"""Train and generate a density-conditioned Nottingham final route."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict
from itertools import cycle, product
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_nottingham_final_retrain import (  # noqa: E402
    CONDITIONED_FIELDS,
    TASK1_PREFIX_FIELDS,
    candidate_settings,
    crop_after,
    full_metrics,
    metrics_from_notes,
    read_note_events_with_velocity,
    rel,
    resolve,
    write_csv,
    write_note_events,
    write_selected_copy,
)
from scripts.train_main import (  # noqa: E402
    encode_files,
    evaluate_loss,
    make_batches,
    parse_number_list,
    set_seed,
    token_label_for_id,
    token_stats,
    token_type_summary,
    try_tokenizer,
    write_dataset_summary,
    write_manifest,
    write_pitch_hist,
    write_skipped_files,
    write_token_lengths,
    write_training_history,
)
from src.data import discover_midi_files, make_lm_windows, pitch_class_histogram, split_files  # noqa: E402
from src.evaluate import analyze_candidate, candidate_reject_reason, max_polyphony, repeated_pitch_bigram_rate  # noqa: E402
from src.markov import NGramModel  # noqa: E402


DENSITY_LABELS = ("low", "med", "high")
DENSITY_TARGETS = {"low": 0.10, "med": 0.60, "high": 0.30}
TASK1_DENSITY = "high"


def token_counts(tokenizer: object, ids: list[int]) -> dict[str, float]:
    counts = {
        "pitch": 0,
        "bar": 0,
        "position": 0,
        "duration": 0,
        "velocity": 0,
        "other": 0,
    }
    for token_id in ids:
        label = token_label_for_id(tokenizer, int(token_id))
        token_type = label.split("_", 1)[0]
        if token_type == "Pitch" or label.startswith("N_"):
            counts["pitch"] += 1
            if label.startswith("N_"):
                counts["duration"] += 1
                counts["velocity"] += 1
        elif token_type == "Bar":
            counts["bar"] += 1
        elif token_type == "Position":
            counts["position"] += 1
        elif token_type == "Duration":
            counts["duration"] += 1
        elif token_type == "Velocity":
            counts["velocity"] += 1
        else:
            counts["other"] += 1
    denom = max(len(ids), 1)
    counts["pitch_per_100_tokens"] = counts["pitch"] * 100.0 / denom
    counts["pitch_token_ratio"] = counts["pitch"] / denom
    counts["bar_count"] = counts["bar"]
    counts["notes_per_bar_proxy"] = counts["pitch"] / max(counts["bar"], 1)
    return counts


def density_label_for_window(tokenizer: object, window: list[int]) -> str:
    stats = token_counts(tokenizer, window)
    pitch_per_100 = float(stats["pitch_per_100_tokens"])
    pitch_count = int(stats["pitch"])
    if pitch_count < 30 or pitch_per_100 < 10.0:
        return "low"
    if pitch_per_100 >= 20.0:
        return "high"
    return "med"


def density_distribution(labels: list[str]) -> dict[str, int]:
    return {label: labels.count(label) for label in DENSITY_LABELS}


def controlled_window(window: list[int], control_id: int) -> list[int]:
    return [control_id, *window]


def make_density_windows(
    tokenizer: object,
    windows: list[list[int]],
    control_ids: dict[str, int],
    balance: bool,
    rng: random.Random,
) -> tuple[list[list[int]], dict[str, object]]:
    buckets = {label: [] for label in DENSITY_LABELS}
    for window in windows:
        label = density_label_for_window(tokenizer, window)
        buckets[label].append(window)

    filtered = {label: list(items) for label, items in buckets.items()}
    selected: list[tuple[str, list[int]]] = []
    if balance:
        available_labels = [label for label, items in filtered.items() if items]
        total = sum(len(items) for items in filtered.values())
        if not available_labels or total == 0:
            raise ValueError("No windows available after density bucketing")
        remaining = total
        for idx, label in enumerate(available_labels):
            if idx == len(available_labels) - 1:
                count = remaining
            else:
                normalized = DENSITY_TARGETS[label] / sum(DENSITY_TARGETS[item] for item in available_labels)
                count = int(round(total * normalized))
                remaining -= count
            for _ in range(max(0, count)):
                selected.append((label, rng.choice(filtered[label])))
        rng.shuffle(selected)
    else:
        for label, items in filtered.items():
            selected.extend((label, item) for item in items)

    controlled = [controlled_window(window, control_ids[label]) for label, window in selected]
    summary = {
        "raw_distribution": {label: len(buckets[label]) for label in DENSITY_LABELS},
        "selected_distribution": density_distribution([label for label, _window in selected]),
        "target_distribution": DENSITY_TARGETS if balance else None,
        "window_count": len(controlled),
    }
    return controlled, summary


def make_weight_vector(tokenizer: object, base_vocab_size: int, control_ids: dict[str, int]) -> torch.Tensor:
    weights = torch.ones(base_vocab_size + len(control_ids), dtype=torch.float32)
    for token_id in range(base_vocab_size):
        label = token_label_for_id(tokenizer, token_id)
        token_type = label.split("_", 1)[0]
        if token_type == "Pitch" or label.startswith("N_"):
            weights[token_id] = 1.5
        elif token_type in {"Velocity", "Duration"}:
            weights[token_id] = 1.2
    for control_id in control_ids.values():
        weights[control_id] = 1.0
    return weights


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    raw = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    )
    flat_labels = labels.reshape(-1)
    token_weights = weights.to(labels.device)[flat_labels]
    return (raw * token_weights).sum() / token_weights.sum().clamp_min(1e-6)


def evaluate_weighted_loss(
    model: GPT2LMHeadModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    weights: torch.Tensor,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in batches:
            result = model(input_ids=inputs.to(device))
            losses.append(float(weighted_loss(result.logits, labels.to(device), weights).detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_density_transformer(
    train_windows: list[list[int]],
    valid_windows: list[list[int]],
    vocab_size: int,
    block_size: int,
    batch_size: int,
    epochs: int,
    max_steps: int,
    lr: float,
    weight_decay: float,
    n_embd: int,
    n_layer: int,
    n_head: int,
    dropout: float,
    grad_clip: float,
    checkpoint_dir: Path,
    eval_interval: int,
    weights: torch.Tensor,
) -> tuple[GPT2LMHeadModel, dict[str, object]]:
    train_batches = make_batches(train_windows, block_size, batch_size)
    valid_batches = make_batches(valid_windows, block_size, batch_size)
    config = GPT2Config(
        vocab_size=max(vocab_size, 8),
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    training_history: list[dict[str, float | int | None]] = []
    eval_history: list[dict[str, float | int]] = []
    best_valid_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best_transformer.pt"
    started = time.perf_counter()
    steps = 0
    model.train()
    for _epoch in range(epochs):
        random.shuffle(train_batches)
        for inputs, labels in train_batches:
            result = model(input_ids=inputs.to(device))
            loss = weighted_loss(result.logits, labels.to(device), weights)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite density transformer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if eval_interval > 0 and steps % eval_interval == 0:
                valid_loss = evaluate_weighted_loss(model, valid_batches, device, weights)
                row = {
                    "step": steps,
                    "total_step": steps,
                    "train_loss": float(loss.detach().cpu()),
                    "valid_loss": valid_loss,
                    "valid_perplexity": math.exp(valid_loss) if valid_loss < 20 else float("inf"),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                training_history.append(row)
                eval_history.append({"step": steps, "valid_loss": valid_loss})
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config.to_dict(),
                            "step": steps,
                            "valid_loss": valid_loss,
                        },
                        best_checkpoint_path,
                    )
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break

    valid_loss = evaluate_weighted_loss(model, valid_batches, device, weights)
    training_history.append(
        {
            "step": steps,
            "total_step": steps,
            "train_loss": losses[-1] if losses else None,
            "valid_loss": valid_loss,
            "valid_perplexity": math.exp(valid_loss) if valid_loss < 20 else float("inf"),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict(),
                "step": steps,
                "valid_loss": valid_loss,
            },
            best_checkpoint_path,
        )
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {
        "model_class": model.__class__.__name__,
        "pretrained_loaded": False,
        "density_conditioned": True,
        "weighted_loss": True,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "block_size": block_size,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "dropout": dropout,
        "grad_clip": grad_clip,
        "epochs_requested": epochs,
        "max_steps": max_steps,
        "steps_completed": steps,
        "total_steps_including_resume": steps,
        "resume_checkpoint": None,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_mean": sum(losses) / max(len(losses), 1),
        "valid_loss": float(checkpoint["valid_loss"]),
        "valid_perplexity": math.exp(float(checkpoint["valid_loss"])) if float(checkpoint["valid_loss"]) < 20 else float("inf"),
        "best_valid_loss": best_valid_loss,
        "best_valid_perplexity": math.exp(best_valid_loss) if best_valid_loss < 20 else float("inf"),
        "best_checkpoint": rel(best_checkpoint_path),
        "eval_history": eval_history,
        "training_history": training_history,
        "runtime_seconds": time.perf_counter() - started,
    }
    return model, metrics


def sample_density_transformer(
    model: GPT2LMHeadModel,
    prefix: list[int],
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_k: int,
    control_ids: dict[str, int],
) -> list[int]:
    device = next(model.parameters()).device
    output = list(prefix)
    blocked = torch.tensor(list(control_ids.values()), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = output[-block_size:]
            inputs = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids=inputs).logits[0, -1, :] / max(temperature, 1e-6)
            logits[blocked] = -float("inf")
            k = min(top_k, logits.numel() - len(control_ids))
            values, indices = torch.topk(logits, k=k)
            probabilities = torch.softmax(values, dim=-1)
            output.append(int(indices[torch.multinomial(probabilities, num_samples=1)].item()))
    return output


def strip_control_tokens(ids: list[int], base_vocab_size: int) -> list[int]:
    return [int(token_id) for token_id in ids if int(token_id) < base_vocab_size]


def decode_density_ids(tokenizer: object, ids: list[int], base_vocab_size: int, path: Path) -> tuple[bool, int, str, list[int]]:
    clean = strip_control_tokens(ids, base_vocab_size)
    try:
        tokenizer.decode_ids(clean, path)
        metrics = analyze_candidate(path)
        if metrics.note_count <= 0:
            return False, 0, "decoded zero-note MIDI", clean
        return True, metrics.note_count, "", clean
    except Exception as exc:  # noqa: BLE001 - reports decode failures in candidate metadata.
        return False, 0, f"{type(exc).__name__}: {exc}", clean


def onset_gap_seconds(notes: list[dict[str, int]], ticks_per_beat: int) -> tuple[float, float]:
    starts = sorted(note["start"] / max(ticks_per_beat, 1) * 0.5 for note in notes)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    if not gaps:
        return 0.0, 0.0
    return max(gaps), sorted(gaps)[min(len(gaps) - 1, int(len(gaps) * 0.95))]


def short_dense_metrics(path: Path, notes: list[dict[str, int]], ticks_per_beat: int, base_row: dict[str, object]) -> dict[str, object]:
    max_gap, p95_gap = onset_gap_seconds(notes, ticks_per_beat)
    row = dict(base_row)
    row["max_onset_gap_seconds"] = max_gap
    row["p95_onset_gap_seconds"] = p95_gap
    return row


def short_dense_reject(row: dict[str, object], *, min_notes: int = 100) -> str:
    reasons = []
    if not row.get("valid", True):
        reasons.append("invalid")
    if int(float(row.get("note_count", 0) or 0)) < min_notes:
        reasons.append(f"note_count_lt_{min_notes}")
    duration = float(row.get("duration_seconds", 0) or 0)
    nps = float(row.get("notes_per_second", 0) or 0)
    if duration < 25:
        reasons.append("duration_lt_25")
    if duration > 70:
        reasons.append("duration_gt_70")
    if nps < 1.5:
        reasons.append("notes_per_second_lt_1p5")
    if nps > 6.0:
        reasons.append("notes_per_second_gt_6")
    if float(row.get("max_onset_gap_seconds", 0) or 0) > 3.0:
        reasons.append("max_gap_gt_3")
    if int(float(row.get("max_simultaneous_notes", 0) or 0)) > 32:
        reasons.append("polyphony_gt_32")
    if int(float(row.get("unique_pitch_count", 0) or 0)) < 6:
        reasons.append("unique_pitch_lt_6")
    if int(float(row.get("pitch_range", 0) or 0)) < 8:
        reasons.append("pitch_range_lt_8")
    return ";".join(reasons)


def short_dense_score(row: dict[str, object]) -> float:
    reject = str(row.get("short_dense_reject_reason", ""))
    if reject:
        return -1_000_000_000.0
    duration = float(row.get("duration_seconds", 0) or 0)
    nps = float(row.get("notes_per_second", 0) or 0)
    notes = float(row.get("note_count", 0) or 0)
    gap = float(row.get("max_onset_gap_seconds", 0) or 0)
    repetition = float(row.get("repeated_pitch_bigram_rate", 0) or 0)
    return (
        2.0
        - abs(duration - 45.0) / 45.0
        - abs(nps - 3.0) / 3.0
        - max(0.0, gap - 1.5) * 0.4
        - max(0.0, repetition - 0.75) * 2.0
        + min(notes, 220.0) / 220.0
    )


def analyze_short_dense(path: Path, min_notes: int = 100) -> dict[str, object]:
    base = asdict(analyze_candidate(path))
    notes, ticks_per_beat = read_note_events_with_velocity(path)
    row = short_dense_metrics(path, notes, ticks_per_beat, base)
    row["short_dense_reject_reason"] = short_dense_reject(row, min_notes=min_notes)
    row["short_dense_usable"] = not row["short_dense_reject_reason"]
    row["short_dense_score"] = short_dense_score(row)
    return row


def continuation_short_dense(
    path: Path,
    split_tick: int,
    prefix: str,
    min_notes: int = 100,
) -> tuple[dict[str, object], list[dict[str, int]], int]:
    notes, ticks_per_beat = read_note_events_with_velocity(path)
    continuation_notes = crop_after(notes, split_tick)
    dataset, task_type, model_type = "nottingham_density_retrain", prefix, "transformer"
    temperature, top_k, candidate_index = None, None, None
    metrics = metrics_from_notes(path, continuation_notes, ticks_per_beat, task_type, model_type, temperature, top_k, candidate_index)
    row = asdict(metrics)
    row = short_dense_metrics(path, continuation_notes, ticks_per_beat, row)
    row["short_dense_reject_reason"] = short_dense_reject(row, min_notes=min_notes)
    row["short_dense_usable"] = not row["short_dense_reject_reason"]
    row["short_dense_score"] = short_dense_score(row)
    row["dataset"] = dataset
    return row, continuation_notes, ticks_per_beat


def first_n_valid_sequences(
    files: list[Path],
    sequence_by_file: dict[Path, list[int]],
    count: int,
    min_tokens: int,
) -> list[tuple[int, Path, list[int]]]:
    selected = []
    for path in files:
        ids = sequence_by_file.get(path, [])
        if len(ids) >= min_tokens:
            selected.append((len(selected), path, ids))
        if len(selected) >= count:
            break
    return selected


def write_generation_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_notes_file(path: Path, notes: list[str]) -> None:
    path.write_text("\n".join(notes).rstrip() + "\n", encoding="utf-8")


def save_task1_run(
    run_dir: Path,
    model: GPT2LMHeadModel,
    tokenizer: object,
    base_vocab_size: int,
    block_size: int,
    control_ids: dict[str, int],
    mode: str,
    prefix_ids: list[int],
    prefix_file: Path | None,
    prefix_token_count: int,
    generate_tokens: int,
    candidate_count: int,
    temperatures: list[float],
    top_ks: list[int],
    seed: int,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    settings = candidate_settings(temperatures, top_ks)
    rows = []
    primer_metrics = None
    primer_path = run_dir / "primer_only.mid"
    if mode == "piece_start_seeded":
        tokenizer.decode_ids(prefix_ids, primer_path)
        primer_metrics = asdict(analyze_candidate(primer_path))
    for idx in range(candidate_count):
        temperature, top_k = next(settings)
        candidate_path = candidates_dir / f"transformer_density_unconditioned_{mode}_temp{str(temperature).replace('.', 'p')}_topk{top_k}_idx{idx:04d}.mid"
        sampled = sample_density_transformer(
            model=model,
            prefix=[control_ids[TASK1_DENSITY], *prefix_ids],
            max_new_tokens=generate_tokens,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k,
            control_ids=control_ids,
        )
        valid, note_count, error, clean_ids = decode_density_ids(tokenizer, sampled, base_vocab_size, candidate_path)
        row = analyze_short_dense(candidate_path) if valid else {"path": rel(candidate_path), "valid": False, "note_count": 0, "short_dense_usable": False, "short_dense_reject_reason": error, "short_dense_score": -1_000_000_000.0}
        row.update(
            {
                "run": run_dir.name,
                "dataset": "nottingham_density_retrain",
                "task_type": "unconditioned",
                "model_type": "transformer",
                "source_path": rel(candidate_path),
                "temperature": temperature,
                "top_k": top_k,
                "candidate_index": idx,
                "density_control": TASK1_DENSITY,
                "generation_mode": mode,
                "prefix_token_count": prefix_token_count,
                "generated_token_count": max(0, len(clean_ids) - prefix_token_count),
                "decode_error": error,
            }
        )
        row.update(token_type_summary(tokenizer, clean_ids[prefix_token_count:], note_count))
        if mode == "piece_start_seeded" and primer_metrics:
            split_tick = round(float(primer_metrics["duration_seconds"]) / 0.5 * read_note_events_with_velocity(candidate_path)[1])
            continuation, _notes, _ticks = continuation_short_dense(candidate_path, split_tick, "unconditioned", min_notes=100)
            for key, value in {
                "primer_token_count": prefix_token_count,
                "generated_token_count": max(0, len(clean_ids) - prefix_token_count),
                "full_token_count": len(clean_ids),
                "primer_duration": primer_metrics["duration_seconds"],
                "full_duration": row.get("duration_seconds", 0),
                "continuation_duration": continuation["duration_seconds"],
                "primer_note_count": primer_metrics["note_count"],
                "full_note_count": row.get("note_count", 0),
                "continuation_note_count": continuation["note_count"],
                "continuation_notes_per_second": continuation["notes_per_second"],
                "continuation_pitch_range": continuation["pitch_range"],
                "continuation_unique_pitch_count": continuation["unique_pitch_count"],
                "continuation_max_simultaneous_notes": continuation["max_simultaneous_notes"],
                "continuation_repeated_pitch_bigram_rate": continuation["repeated_pitch_bigram_rate"],
                "continuation_usable": continuation["short_dense_usable"],
                "continuation_reject_reason": continuation["short_dense_reject_reason"],
                "continuation_score": continuation["short_dense_score"],
                "continuation_max_onset_gap_seconds": continuation["max_onset_gap_seconds"],
            }.items():
                row[key] = value
        rows.append(row)

    if mode == "piece_start_seeded":
        usable = [row for row in rows if str(row.get("continuation_usable")) == "True" or row.get("continuation_usable") is True]
        selected = max(usable, key=lambda row: float(row.get("continuation_score", -1_000_000_000)), default=None)
    else:
        usable = [row for row in rows if row.get("short_dense_usable") is True]
        selected = max(usable, key=lambda row: float(row.get("short_dense_score", -1_000_000_000)), default=None)

    selected_rows = []
    if selected is None:
        selected_rows.append({"run": run_dir.name, "task_type": "unconditioned", "selected_path": "", "reject_reason": "No short-dense usable candidate found."})
    else:
        source = resolve(selected["source_path"])
        selected_path = run_dir / "symbolic_unconditioned.mid"
        write_selected_copy(source, selected_path)
        selected_row = dict(selected)
        selected_row["selected_path"] = rel(selected_path)
        if mode == "piece_start_seeded" and primer_metrics:
            full_with_primer = run_dir / "full_with_primer.mid"
            continuation_only = run_dir / "continuation_only.mid"
            write_selected_copy(source, full_with_primer)
            full_notes, ticks_per_beat = read_note_events_with_velocity(source)
            split_tick = round(float(primer_metrics["duration_seconds"]) / 0.5 * ticks_per_beat)
            write_note_events(continuation_only, crop_after(full_notes, split_tick), ticks_per_beat)
            selected_row["primer_only_path"] = rel(primer_path)
            selected_row["full_with_primer_path"] = rel(full_with_primer)
            selected_row["continuation_only_path"] = rel(continuation_only)
        selected_rows.append(selected_row)

    write_csv(run_dir / "candidate_ranking.csv", rows)
    write_csv(run_dir / "selected_candidates.csv", selected_rows)
    write_generation_config(
        run_dir / "generation_config.json",
        {
            "run": run_dir.name,
            "task": "task1_unconditioned_density",
            "generation_mode": mode,
            "density_control": TASK1_DENSITY,
            "candidate_count": candidate_count,
            "generate_tokens": generate_tokens,
            "temperatures": temperatures,
            "top_ks": top_ks,
            "prefix_token_count": prefix_token_count,
            "prefix_file": "" if prefix_file is None else rel(prefix_file),
            "selection_rule": "short_dense_score; piece_start uses continuation-only short_dense_score",
            "seed": seed,
        },
    )
    write_notes_file(
        run_dir / "notes.txt",
        [
            f"Task 1 density-conditioned run: {mode}",
            "Density control token: high",
            "Selection prefers 25-70s, >=100 notes, 1.5-6 notes/s, max onset gap <=3s.",
            "For piece-start runs, continuation_only.mid is the primary listening file.",
        ],
    )
    return selected_rows[0]


def save_conditioned_bigpool(
    run_dir: Path,
    model: GPT2LMHeadModel,
    tokenizer: object,
    base_vocab_size: int,
    block_size: int,
    control_ids: dict[str, int],
    sequence_by_file: dict[Path, list[int]],
    valid_files: list[Path],
    prefix_count: int,
    prefix_tokens: int,
    candidates_per_prefix: int,
    generate_tokens: int,
    temperatures: list[float],
    top_ks: list[int],
    seed: int,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = run_dir / "candidates"
    prefix_dir = run_dir / "prefixes"
    candidates_dir.mkdir(exist_ok=True)
    prefix_dir.mkdir(exist_ok=True)
    rows = []
    settings = candidate_settings(temperatures, top_ks)
    prefixes = first_n_valid_sequences(valid_files, sequence_by_file, prefix_count, prefix_tokens + 1)
    for prefix_id, prefix_file, ids in prefixes:
        prefix_ids = ids[:prefix_tokens]
        prefix_label = density_label_for_window(tokenizer, prefix_ids)
        density_control = "high" if prefix_label == "high" else "med"
        prefix_path = prefix_dir / f"conditioned_prefix_{prefix_id:03d}.mid"
        tokenizer.decode_ids(prefix_ids, prefix_path)
        prefix_metrics = asdict(analyze_candidate(prefix_path))
        split_tick = round(float(prefix_metrics["duration_seconds"]) / 0.5 * read_note_events_with_velocity(prefix_path)[1])
        for local_idx in range(candidates_per_prefix):
            temperature, top_k = next(settings)
            candidate_index = prefix_id * candidates_per_prefix + local_idx
            candidate_path = candidates_dir / f"transformer_density_conditioned_prefix{prefix_id:03d}_temp{str(temperature).replace('.', 'p')}_topk{top_k}_idx{candidate_index:04d}.mid"
            sampled = sample_density_transformer(
                model=model,
                prefix=[control_ids[density_control], *prefix_ids],
                max_new_tokens=generate_tokens,
                block_size=block_size,
                temperature=temperature,
                top_k=top_k,
                control_ids=control_ids,
            )
            valid, note_count, error, clean_ids = decode_density_ids(tokenizer, sampled, base_vocab_size, candidate_path)
            row = full_metrics(candidate_path, "nottingham_density_retrain") if valid else {"path": rel(candidate_path), "valid": False, "note_count": 0, "score": -1_000_000_000.0, "reject_reason": error}
            row.update(
                {
                    "run": run_dir.name,
                    "task_type": "conditioned",
                    "model_type": "transformer",
                    "source_path": rel(candidate_path),
                    "temperature": temperature,
                    "top_k": top_k,
                    "candidate_index": candidate_index,
                    "conditioned_prefix_id": prefix_id,
                    "conditioned_prefix_file": rel(prefix_file),
                    "conditioned_density_control": density_control,
                    "conditioned_prefix_token_count": prefix_tokens,
                    "conditioned_generated_token_count": max(0, len(clean_ids) - prefix_tokens),
                    "decode_error": error,
                }
            )
            if valid:
                continuation, _notes, _ticks = continuation_short_dense(candidate_path, split_tick, "conditioned", min_notes=100)
                for key, value in {
                    "conditioned_full_token_count": len(clean_ids),
                    "conditioned_prefix_duration": prefix_metrics["duration_seconds"],
                    "conditioned_full_duration": row.get("duration_seconds", 0),
                    "conditioned_continuation_duration": continuation["duration_seconds"],
                    "conditioned_prefix_note_count": prefix_metrics["note_count"],
                    "conditioned_full_note_count": row.get("note_count", 0),
                    "conditioned_continuation_note_count": continuation["note_count"],
                    "conditioned_continuation_notes_per_second": continuation["notes_per_second"],
                    "conditioned_continuation_pitch_range": continuation["pitch_range"],
                    "conditioned_continuation_unique_pitch_count": continuation["unique_pitch_count"],
                    "conditioned_continuation_max_polyphony": continuation["max_simultaneous_notes"],
                    "conditioned_continuation_repeated_pitch_bigram_rate": continuation["repeated_pitch_bigram_rate"],
                    "conditioned_continuation_usable": continuation["short_dense_usable"],
                    "conditioned_continuation_reject_reason": continuation["short_dense_reject_reason"],
                    "conditioned_continuation_score": continuation["short_dense_score"],
                    "conditioned_continuation_max_onset_gap_seconds": continuation["max_onset_gap_seconds"],
                }.items():
                    row[key] = value
            rows.append(row)

    usable = [row for row in rows if row.get("conditioned_continuation_usable") is True]
    selected = max(usable, key=lambda row: float(row.get("conditioned_continuation_score", -1_000_000_000)), default=None)
    selected_rows = []
    if selected is None:
        selected_rows.append({"run": run_dir.name, "task_type": "conditioned", "selected_path": "", "reject_reason": "No short-dense usable conditioned continuation found."})
    else:
        source = resolve(selected["source_path"])
        prefix_id = int(selected["conditioned_prefix_id"])
        prefix_path = prefix_dir / f"conditioned_prefix_{prefix_id:03d}.mid"
        prefix_metrics = asdict(analyze_candidate(prefix_path))
        full_with_prefix = run_dir / "conditioned_full_with_prefix.mid"
        continuation_only = run_dir / "conditioned_continuation_only.mid"
        symbolic_conditioned = run_dir / "symbolic_conditioned.mid"
        write_selected_copy(prefix_path, run_dir / "conditioned_prefix_only.mid")
        write_selected_copy(source, full_with_prefix)
        write_selected_copy(source, symbolic_conditioned)
        full_notes, ticks_per_beat = read_note_events_with_velocity(source)
        split_tick = round(float(prefix_metrics["duration_seconds"]) / 0.5 * ticks_per_beat)
        write_note_events(continuation_only, crop_after(full_notes, split_tick), ticks_per_beat)
        selected_row = dict(selected)
        selected_row["selected_path"] = rel(symbolic_conditioned)
        selected_row["conditioned_prefix_only_path"] = rel(run_dir / "conditioned_prefix_only.mid")
        selected_row["conditioned_full_with_prefix_path"] = rel(full_with_prefix)
        selected_row["conditioned_continuation_only_path"] = rel(continuation_only)
        selected_rows.append(selected_row)

    write_csv(run_dir / "candidate_ranking.csv", rows)
    write_csv(run_dir / "selected_candidates.csv", selected_rows)
    write_generation_config(
        run_dir / "generation_config.json",
        {
            "run": run_dir.name,
            "task": "task2_conditioned_density_bigpool",
            "prefix_count": prefix_count,
            "prefix_tokens": prefix_tokens,
            "candidates_per_prefix": candidates_per_prefix,
            "generate_tokens": generate_tokens,
            "temperatures": temperatures,
            "top_ks": top_ks,
            "selection_rule": "conditioned continuation-only short_dense_score",
            "seed": seed,
        },
    )
    write_notes_file(
        run_dir / "notes.txt",
        [
            "Task 2 density-conditioned bigpool.",
            "Prefix density low/med maps to MED control; high maps to HIGH control.",
            "conditioned_continuation_only.mid is the primary listening/evaluation file.",
        ],
    )
    return selected_rows[0]


def aggregate_tables(indexed_dir: Path, evaluation_dir: Path) -> None:
    ranking_rows = []
    selected_rows = []
    for run_dir in sorted(path for path in indexed_dir.iterdir() if path.is_dir()):
        ranking = run_dir / "candidate_ranking.csv"
        selected = run_dir / "selected_candidates.csv"
        if ranking.exists():
            with ranking.open(newline="", encoding="utf-8") as handle:
                ranking_rows.extend(csv.DictReader(handle))
        if selected.exists():
            with selected.open(newline="", encoding="utf-8") as handle:
                selected_rows.extend(csv.DictReader(handle))
    tables_dir = evaluation_dir / "tables"
    write_csv(tables_dir / "candidate_ranking.csv", ranking_rows)
    write_csv(tables_dir / "selected_candidates.csv", selected_rows)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="nottingham_density_retrain")
    parser.add_argument("--input-dir", default="data/nottingham-dataset-master/MIDI")
    parser.add_argument("--output-dir", default="outputs/candidates/nottingham_density_retrain")
    parser.add_argument("--metrics-dir", default="outputs/metrics/nottingham_density_retrain")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/nottingham_density_retrain")
    parser.add_argument("--indexed-dir", default="outputs/candidates/final/nottingham_density_retrain")
    parser.add_argument("--evaluation-dir", default="outputs/evaluation/nottingham_density_retrain")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--generate-tokens", type=int, default=512)
    parser.add_argument("--task1-candidate-count", type=int, default=100)
    parser.add_argument("--conditioned-prefix-count", type=int, default=20)
    parser.add_argument("--conditioned-prefix-tokens", type=int, default=128)
    parser.add_argument("--conditioned-candidates-per-prefix", type=int, default=10)
    parser.add_argument("--temperatures", default="0.7,0.8,0.9")
    parser.add_argument("--top-ks", default="20,50")
    parser.add_argument("--seed", type=int, default=253)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    metrics_dir = resolve(args.metrics_dir)
    checkpoint_dir = resolve(args.checkpoint_dir)
    indexed_dir = resolve(args.indexed_dir)
    evaluation_dir = resolve(args.evaluation_dir)
    for path in (output_dir, metrics_dir, checkpoint_dir, indexed_dir, evaluation_dir / "tables"):
        path.mkdir(parents=True, exist_ok=True)

    midi_files = discover_midi_files([input_dir])
    if not midi_files:
        raise ValueError(f"No MIDI files found under {input_dir}")
    train_files, valid_files = split_files(midi_files, args.valid_fraction)
    tokenizer, _sequences, tokenizer_mode, tokenizer_detail = try_tokenizer(midi_files, output_dir)
    sequence_by_file, skipped_files = encode_files(tokenizer, midi_files)
    midi_files = [path for path in midi_files if path in sequence_by_file]
    train_files = [path for path in train_files if path in sequence_by_file]
    valid_files = [path for path in valid_files if path in sequence_by_file]
    train_sequences = [sequence_by_file[path] for path in train_files]
    valid_sequences = [sequence_by_file[path] for path in valid_files]
    raw_train_windows = make_lm_windows(train_sequences, args.block_size, args.stride)
    raw_valid_windows = make_lm_windows(valid_sequences, args.block_size, args.stride)
    base_vocab_size = int(tokenizer.vocab_size)
    control_ids = {label: base_vocab_size + index for index, label in enumerate(DENSITY_LABELS)}
    train_windows, train_density_summary = make_density_windows(tokenizer, raw_train_windows, control_ids, True, rng)
    valid_windows, valid_density_summary = make_density_windows(tokenizer, raw_valid_windows, control_ids, False, rng)
    vocab_size = base_vocab_size + len(control_ids)
    weights = make_weight_vector(tokenizer, base_vocab_size, control_ids)

    model, transformer_metrics = train_density_transformer(
        train_windows=train_windows,
        valid_windows=valid_windows,
        vocab_size=vocab_size,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        checkpoint_dir=checkpoint_dir,
        eval_interval=args.eval_interval,
        weights=weights,
    )

    temperatures = [float(value) for value in parse_number_list(args.temperatures, float)]
    top_ks = [int(value) for value in parse_number_list(args.top_ks, int)]
    selected_rows = []
    selected_rows.append(
        save_task1_run(
            indexed_dir / "run_density_uncond_bos",
            model,
            tokenizer,
            base_vocab_size,
            args.block_size,
            control_ids,
            "pure_bos",
            [raw_train_windows[0][0]],
            None,
            1,
            args.generate_tokens,
            args.task1_candidate_count,
            temperatures,
            top_ks,
            args.seed,
        )
    )
    selected_rows.append(
        save_task1_run(
            indexed_dir / "run_density_uncond_structural_seeded",
            model,
            tokenizer,
            base_vocab_size,
            args.block_size,
            control_ids,
            "structural_seeded",
            raw_valid_windows[0][:32],
            valid_files[0],
            32,
            args.generate_tokens,
            args.task1_candidate_count,
            temperatures,
            top_ks,
            args.seed,
        )
    )
    for prefix_tokens in (128, 256):
        selected_rows.append(
            save_task1_run(
                indexed_dir / f"run_density_uncond_piece_start_{prefix_tokens}",
                model,
                tokenizer,
                base_vocab_size,
                args.block_size,
                control_ids,
                "piece_start_seeded",
                sequence_by_file[valid_files[prefix_tokens % len(valid_files)]][:prefix_tokens],
                valid_files[prefix_tokens % len(valid_files)],
                prefix_tokens,
                args.generate_tokens,
                args.task1_candidate_count,
                temperatures,
                top_ks,
                args.seed,
            )
        )
    selected_rows.append(
        save_conditioned_bigpool(
            indexed_dir / "run_density_conditioned_bigpool",
            model,
            tokenizer,
            base_vocab_size,
            args.block_size,
            control_ids,
            sequence_by_file,
            valid_files,
            args.conditioned_prefix_count,
            args.conditioned_prefix_tokens,
            args.conditioned_candidates_per_prefix,
            args.generate_tokens,
            temperatures,
            top_ks,
            args.seed,
        )
    )

    markov = NGramModel(order=3, seed=args.seed)
    markov.fit(raw_train_windows)
    markov_perplexity = markov.perplexity(raw_valid_windows)

    manifest_path = metrics_dir / "manifest.csv"
    splits = {path: "train" for path in train_files}
    splits.update({path: "valid" for path in valid_files})
    write_manifest(manifest_path, midi_files, splits, sequence_by_file, args.dataset_name)
    skipped_path = metrics_dir / "skipped_files.csv"
    write_skipped_files(skipped_path, skipped_files)
    pitch_hist_path = metrics_dir / "pitch_class_histogram.csv"
    token_lengths_path = metrics_dir / "token_length_distribution.csv"
    dataset_summary_path = metrics_dir / "dataset_summary.csv"
    training_history_path = metrics_dir / "training_history.csv"
    write_pitch_hist(pitch_hist_path, pitch_class_histogram(midi_files))
    write_token_lengths(token_lengths_path, midi_files, sequence_by_file)
    write_training_history(training_history_path, transformer_metrics.get("training_history", []))

    summary = {
        "dataset_name": args.dataset_name,
        "input_dir": rel(input_dir),
        "file_count": len(midi_files),
        "skipped_file_count": len(skipped_files),
        "total_bytes": sum(path.stat().st_size for path in midi_files),
        "train_file_count": len(train_files),
        "valid_file_count": len(valid_files),
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_detail": tokenizer_detail,
        "token_length_stats": token_stats(list(sequence_by_file.values())),
        "base_vocab_size": base_vocab_size,
        "vocab_size": vocab_size,
        "density_control_tokens": control_ids,
        "density_bucket_rule": "low if pitch_count<30 or pitch_per_100<10; high if pitch_per_100>=20; otherwise med",
        "train_window_count": len(raw_train_windows),
        "valid_window_count": len(raw_valid_windows),
        "controlled_train_window_count": len(train_windows),
        "controlled_valid_window_count": len(valid_windows),
        "train_density_summary": train_density_summary,
        "valid_density_summary": valid_density_summary,
        "markov": {"order": markov.order, "valid_perplexity": markov_perplexity},
        "transformer": transformer_metrics,
        "generation": {
            "generate_tokens": args.generate_tokens,
            "task1_candidate_count": args.task1_candidate_count,
            "conditioned_prefix_count": args.conditioned_prefix_count,
            "conditioned_prefix_tokens": args.conditioned_prefix_tokens,
            "conditioned_candidates_per_prefix": args.conditioned_candidates_per_prefix,
            "temperatures": temperatures,
            "top_ks": top_ks,
            "selection": "short-dense continuation-only for piece-start and conditioned runs",
        },
        "metrics": {
            "manifest": rel(manifest_path),
            "skipped_files": rel(skipped_path),
            "training_history": rel(training_history_path),
            "pitch_class_histogram": rel(pitch_hist_path),
            "token_length_distribution": rel(token_lengths_path),
            "dataset_summary": rel(dataset_summary_path),
        },
        "selected_runs": selected_rows,
    }
    write_dataset_summary(dataset_summary_path, summary)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    aggregate_tables(indexed_dir, evaluation_dir)
    print(json.dumps({"summary": rel(metrics_dir / "summary.json"), "indexed_dir": rel(indexed_dir), "evaluation_dir": rel(evaluation_dir)}, indent=2))


if __name__ == "__main__":
    main()
