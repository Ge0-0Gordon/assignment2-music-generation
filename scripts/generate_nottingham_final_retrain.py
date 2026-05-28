"""Generate Nottingham final-retrain indexed Transformer and Markov runs."""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from itertools import cycle, product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from split_piece_start_outputs import (  # noqa: E402
    crop_after,
    metrics_from_notes,
    read_note_events_with_velocity,
    tokenizer_from_summary,
    write_note_events,
)
from src.data import make_lm_windows  # noqa: E402
from src.evaluate import analyze_candidate, infer_candidate_labels, infer_sampling_settings  # noqa: E402
from src.markov import NGramModel  # noqa: E402
from scripts.train_main import (  # noqa: E402
    decode_with_retries,
    load_transformer_for_generation,
    sample_transformer,
    token_type_summary,
)


TASK1_PREFIX_FIELDS = [
    "primer_token_count",
    "generated_token_count",
    "full_token_count",
    "primer_duration",
    "full_duration",
    "continuation_duration",
    "primer_note_count",
    "full_note_count",
    "continuation_note_count",
    "continuation_notes_per_second",
    "continuation_pitch_range",
    "continuation_unique_pitch_count",
    "continuation_max_simultaneous_notes",
    "continuation_repeated_pitch_bigram_rate",
    "continuation_usable",
    "continuation_reject_reason",
    "continuation_score",
]

CONDITIONED_FIELDS = [
    "conditioned_prefix_id",
    "conditioned_prefix_file",
    "conditioned_prefix_token_count",
    "conditioned_generated_token_count",
    "conditioned_full_token_count",
    "conditioned_prefix_duration",
    "conditioned_full_duration",
    "conditioned_continuation_duration",
    "conditioned_prefix_note_count",
    "conditioned_full_note_count",
    "conditioned_continuation_note_count",
    "conditioned_continuation_notes_per_second",
    "conditioned_continuation_pitch_range",
    "conditioned_continuation_unique_pitch_count",
    "conditioned_continuation_max_polyphony",
    "conditioned_continuation_repeated_pitch_bigram_rate",
    "conditioned_continuation_usable",
    "conditioned_continuation_reject_reason",
    "conditioned_continuation_score",
]


def resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_csv(path: Path, rows: list[dict[str, object]], preferred: list[str] | None = None) -> None:
    fieldnames = list(preferred or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_summary(metrics_dir: Path) -> dict[str, object]:
    return json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))


def split_files_from_manifest(summary: dict[str, object]) -> tuple[list[Path], list[Path]]:
    manifest = resolve(summary["metrics"]["manifest"])
    train_files: list[Path] = []
    valid_files: list[Path] = []
    for row in read_csv_rows(manifest):
        path = resolve(row["file_path"])
        if row.get("split") == "valid":
            valid_files.append(path)
        else:
            train_files.append(path)
    return train_files, valid_files


def sequences_for_files(tokenizer: object, files: list[Path]) -> dict[Path, list[int]]:
    sequences: dict[Path, list[int]] = {}
    for path in files:
        ids = tokenizer.encode_file(path)
        if ids:
            sequences[path] = ids
    return sequences


def format_temperature(value: float) -> str:
    return str(value).replace(".", "p")


def candidate_settings(temperatures: list[float], top_ks: list[int]):
    return cycle(list(product(temperatures, top_ks)))


def full_metrics(candidate_path: Path, dataset_name: str, source_path: Path | None = None) -> dict[str, object]:
    row = asdict(analyze_candidate(candidate_path))
    row["dataset"] = dataset_name
    if source_path is not None:
        row["source_path"] = rel(source_path)
    return row


def prefix_metrics_for_ids(tokenizer: object, ids: list[int], path: Path) -> tuple[dict[str, object], int]:
    tokenizer.decode_ids(ids, path)
    metrics = asdict(analyze_candidate(path))
    notes, _ticks_per_beat = read_note_events_with_velocity(path)
    duration_ticks = max((note["end"] for note in notes), default=0)
    return metrics, duration_ticks


def continuation_metrics(
    candidate_path: Path,
    prefix_metrics: dict[str, object],
    prefix_token_count: int,
    generated_token_count: int,
    prefix: str,
    prefix_id: int | None = None,
    prefix_file: Path | None = None,
) -> dict[str, object]:
    full = asdict(analyze_candidate(candidate_path))
    notes, ticks_per_beat = read_note_events_with_velocity(candidate_path)
    split_tick = round(float(prefix_metrics["duration_seconds"]) / 0.5 * ticks_per_beat)
    continuation_notes = crop_after(notes, split_tick)
    _dataset, task_type, model_type = infer_candidate_labels(candidate_path)
    temperature, top_k, candidate_index = infer_sampling_settings(candidate_path)
    continuation = metrics_from_notes(
        candidate_path,
        continuation_notes,
        ticks_per_beat,
        task_type,
        model_type,
        temperature,
        top_k,
        candidate_index,
    )
    if prefix == "conditioned":
        return {
            "conditioned_prefix_id": "" if prefix_id is None else prefix_id,
            "conditioned_prefix_file": "" if prefix_file is None else rel(prefix_file),
            "conditioned_prefix_token_count": prefix_token_count,
            "conditioned_generated_token_count": generated_token_count,
            "conditioned_full_token_count": prefix_token_count + generated_token_count,
            "conditioned_prefix_duration": prefix_metrics["duration_seconds"],
            "conditioned_full_duration": full["duration_seconds"],
            "conditioned_continuation_duration": continuation.duration_seconds,
            "conditioned_prefix_note_count": prefix_metrics["note_count"],
            "conditioned_full_note_count": full["note_count"],
            "conditioned_continuation_note_count": continuation.note_count,
            "conditioned_continuation_notes_per_second": continuation.notes_per_second,
            "conditioned_continuation_pitch_range": continuation.pitch_range,
            "conditioned_continuation_unique_pitch_count": continuation.unique_pitch_count,
            "conditioned_continuation_max_polyphony": continuation.max_simultaneous_notes,
            "conditioned_continuation_repeated_pitch_bigram_rate": continuation.repeated_pitch_bigram_rate,
            "conditioned_continuation_usable": continuation.usable,
            "conditioned_continuation_reject_reason": continuation.reject_reason,
            "conditioned_continuation_score": continuation.score,
        }
    return {
        "primer_token_count": prefix_token_count,
        "generated_token_count": generated_token_count,
        "full_token_count": prefix_token_count + generated_token_count,
        "primer_duration": prefix_metrics["duration_seconds"],
        "full_duration": full["duration_seconds"],
        "continuation_duration": continuation.duration_seconds,
        "primer_note_count": prefix_metrics["note_count"],
        "full_note_count": full["note_count"],
        "continuation_note_count": continuation.note_count,
        "continuation_notes_per_second": continuation.notes_per_second,
        "continuation_pitch_range": continuation.pitch_range,
        "continuation_unique_pitch_count": continuation.unique_pitch_count,
        "continuation_max_simultaneous_notes": continuation.max_simultaneous_notes,
        "continuation_repeated_pitch_bigram_rate": continuation.repeated_pitch_bigram_rate,
        "continuation_usable": continuation.usable,
        "continuation_reject_reason": continuation.reject_reason,
        "continuation_score": continuation.score,
    }


def write_selected_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return rel(target)


def best_full(rows: list[dict[str, object]], task_type: str) -> dict[str, object] | None:
    usable = [row for row in rows if row["task_type"] == task_type and str(row.get("usable")).lower() == "true"]
    usable.sort(key=lambda row: float(row.get("score") or -1e9), reverse=True)
    return usable[0] if usable else None


def best_continuation(rows: list[dict[str, object]], field_prefix: str) -> dict[str, object] | None:
    usable_field = f"{field_prefix}_usable"
    score_field = f"{field_prefix}_score"
    usable = [row for row in rows if str(row.get(usable_field)).lower() == "true"]
    usable.sort(key=lambda row: float(row.get(score_field) or -1e9), reverse=True)
    return usable[0] if usable else None


def write_run_common(run_dir: Path, ranking_rows: list[dict[str, object]], selected_rows: list[dict[str, object]], config: dict[str, object], notes: list[str]) -> None:
    write_csv(run_dir / "candidate_ranking.csv", ranking_rows)
    write_csv(run_dir / "selected_candidates.csv", selected_rows)
    (run_dir / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "notes.txt").write_text("\n".join(notes).rstrip() + "\n", encoding="utf-8")


def generate_transformer_ids(
    model,
    prefix_ids: list[int],
    generate_tokens: int,
    vocab_size: int,
    block_size: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    max_new = max(1, generate_tokens - len(prefix_ids))
    return sample_transformer(
        model,
        prefix=prefix_ids,
        max_new_tokens=max_new,
        vocab_size=vocab_size,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k,
    )


def generate_task1_run(
    run_dir: Path,
    model,
    tokenizer: object,
    dataset_name: str,
    mode: str,
    prefix_ids: list[int],
    prefix_file: Path | None,
    generate_tokens: int,
    candidate_count: int,
    temperatures: list[float],
    top_ks: list[int],
    vocab_size: int,
    block_size: int,
    checkpoint: Path,
    seed: int,
) -> None:
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    settings = candidate_settings(temperatures, top_ks)
    ranking_rows: list[dict[str, object]] = []
    prefix_metrics: dict[str, object] | None = None
    primer_path = run_dir / "primer_only.mid"
    if mode == "piece_start_seeded":
        prefix_metrics, _prefix_duration_ticks = prefix_metrics_for_ids(tokenizer, prefix_ids, primer_path)
    for index in range(candidate_count):
        temperature, top_k = next(settings)
        name = (
            f"transformer_unconditioned_{mode}_temp{format_temperature(float(temperature))}_"
            f"topk{int(top_k)}_idx{index:03d}"
        )
        path = candidates_dir / f"{name}.mid"
        ids = generate_transformer_ids(
            model,
            prefix_ids,
            generate_tokens,
            vocab_size,
            block_size,
            float(temperature),
            int(top_k),
        )
        valid, notes, error, selected_ids = decode_with_retries(tokenizer, [ids], path)
        row = full_metrics(path, dataset_name, path)
        row.update(
            {
                "valid": valid,
                "note_count": notes if valid else row["note_count"],
                "temperature": float(temperature),
                "top_k": int(top_k),
                "candidate_index": index,
                "generation_mode": mode,
                "prefix_token_count": len(prefix_ids),
                "continuation_token_count": max(0, len(selected_ids) - len(prefix_ids)),
                "error": error,
            }
        )
        row.update(token_type_summary(tokenizer, selected_ids, row["note_count"]))
        if mode == "piece_start_seeded" and prefix_metrics is not None:
            row.update(
                continuation_metrics(
                    path,
                    prefix_metrics,
                    len(prefix_ids),
                    max(0, len(selected_ids) - len(prefix_ids)),
                    "task1",
                )
            )
        else:
            for field in TASK1_PREFIX_FIELDS:
                row[field] = ""
        ranking_rows.append(row)

    if mode == "piece_start_seeded":
        selected = best_continuation(ranking_rows, "continuation")
    else:
        selected = best_full(ranking_rows, "unconditioned")
    selected_rows: list[dict[str, object]] = []
    if selected is None:
        selected_rows.append(
            {
                "run": run_dir.name,
                "dataset": dataset_name,
                "task_type": "unconditioned",
                "selected_path": "",
                "source_path": "",
                "reject_reason": "No usable Transformer candidate found.",
            }
        )
    else:
        source = resolve(selected["source_path"])
        selected_path = run_dir / "symbolic_unconditioned.mid"
        write_selected_copy(source, selected_path)
        selected_row = {
            "run": run_dir.name,
            "dataset": dataset_name,
            "task_type": "unconditioned",
            "selected_path": rel(selected_path),
            "source_path": rel(source),
            "score": selected["score"],
            "note_count": selected["note_count"],
            "duration_seconds": selected["duration_seconds"],
            "notes_per_second": selected["notes_per_second"],
            "pitch_range": selected["pitch_range"],
            "max_simultaneous_notes": selected["max_simultaneous_notes"],
            "repeated_pitch_bigram_rate": selected["repeated_pitch_bigram_rate"],
            "reject_reason": selected.get("reject_reason", ""),
        }
        for field in TASK1_PREFIX_FIELDS:
            selected_row[field] = selected.get(field, "")
        if mode == "piece_start_seeded":
            full_with_primer = run_dir / "full_with_primer.mid"
            continuation_only = run_dir / "continuation_only.mid"
            write_selected_copy(source, full_with_primer)
            full_notes, ticks_per_beat = read_note_events_with_velocity(source)
            split_tick = round(float(prefix_metrics["duration_seconds"]) / 0.5 * ticks_per_beat) if prefix_metrics else 0
            write_note_events(continuation_only, crop_after(full_notes, split_tick), ticks_per_beat)
            selected_row["primer_only_path"] = rel(primer_path)
            selected_row["full_with_primer_path"] = rel(full_with_primer)
            selected_row["continuation_only_path"] = rel(continuation_only)
        selected_rows.append(selected_row)

    config = {
        "run": run_dir.name,
        "dataset": dataset_name,
        "task": "task1_unconditioned",
        "model_type": "transformer",
        "checkpoint": rel(checkpoint),
        "generation_mode": mode,
        "generate_tokens": generate_tokens,
        "candidate_count_total": candidate_count,
        "temperatures": temperatures,
        "top_ks": top_ks,
        "prefix_token_count": len(prefix_ids),
        "prefix_file": "" if prefix_file is None else rel(prefix_file),
        "seed": seed,
        "selection_rule": "continuation_score for piece_start_seeded; full score for other unconditioned modes",
    }
    notes = [
        f"{run_dir.name}: Task 1 {mode} Transformer run.",
        "Selected candidate must pass hard reject filters.",
    ]
    if mode == "piece_start_seeded":
        notes.append("symbolic_unconditioned.mid and full_with_primer.mid include the real primer; continuation_only.mid is the primary listening/evaluation file.")
    write_run_common(run_dir, ranking_rows, selected_rows, config, notes)


def valid_prefixes(
    tokenizer: object,
    valid_sequences: dict[Path, list[int]],
    prefix_count: int,
    prefix_tokens: int,
    seed: int,
) -> list[tuple[int, Path, list[int]]]:
    candidates = [(path, ids) for path, ids in valid_sequences.items() if len(ids) >= 2]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: list[tuple[int, Path, list[int]]] = []
    for path, ids in candidates:
        prefix = ids[: min(prefix_tokens, len(ids))]
        if prefix:
            selected.append((len(selected), path, prefix))
        if len(selected) >= prefix_count:
            break
    if not selected:
        raise ValueError("No validation prefixes available")
    return selected


def generate_conditioned_bigpool(
    run_dir: Path,
    model,
    tokenizer: object,
    dataset_name: str,
    prefixes: list[tuple[int, Path, list[int]]],
    generate_tokens: int,
    candidates_per_prefix: int,
    temperatures: list[float],
    top_ks: list[int],
    vocab_size: int,
    block_size: int,
    checkpoint: Path,
    seed: int,
) -> None:
    candidates_dir = run_dir / "candidates"
    prefix_dir = run_dir / "prefixes"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    prefix_dir.mkdir(parents=True, exist_ok=True)
    ranking_rows: list[dict[str, object]] = []
    settings = candidate_settings(temperatures, top_ks)
    for prefix_id, prefix_file, prefix_ids in prefixes:
        prefix_path = prefix_dir / f"conditioned_prefix_{prefix_id:03d}.mid"
        prefix_metrics, _duration_ticks = prefix_metrics_for_ids(tokenizer, prefix_ids, prefix_path)
        for local_index in range(candidates_per_prefix):
            temperature, top_k = next(settings)
            global_index = prefix_id * candidates_per_prefix + local_index
            name = (
                f"transformer_conditioned_prefix{prefix_id:03d}_temp{format_temperature(float(temperature))}_"
                f"topk{int(top_k)}_idx{global_index:04d}"
            )
            path = candidates_dir / f"{name}.mid"
            ids = generate_transformer_ids(
                model,
                prefix_ids,
                generate_tokens,
                vocab_size,
                block_size,
                float(temperature),
                int(top_k),
            )
            valid, notes, error, selected_ids = decode_with_retries(tokenizer, [ids], path)
            row = full_metrics(path, dataset_name, path)
            row.update(
                {
                    "valid": valid,
                    "note_count": notes if valid else row["note_count"],
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    "candidate_index": global_index,
                    "generation_mode": "conditioned_bigpool",
                    "prefix_token_count": len(prefix_ids),
                    "continuation_token_count": max(0, len(selected_ids) - len(prefix_ids)),
                    "error": error,
                }
            )
            row.update(token_type_summary(tokenizer, selected_ids, row["note_count"]))
            row.update(
                continuation_metrics(
                    path,
                    prefix_metrics,
                    len(prefix_ids),
                    max(0, len(selected_ids) - len(prefix_ids)),
                    "conditioned",
                    prefix_id=prefix_id,
                    prefix_file=prefix_file,
                )
            )
            ranking_rows.append(row)

    selected = best_continuation(ranking_rows, "conditioned_continuation")
    selected_rows: list[dict[str, object]] = []
    if selected is None:
        selected_rows.append(
            {
                "run": run_dir.name,
                "dataset": dataset_name,
                "task_type": "conditioned",
                "selected_path": "",
                "source_path": "",
                "reject_reason": "No usable conditioned continuation-only Transformer candidate found.",
            }
        )
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
        selected_row = {
            "run": run_dir.name,
            "dataset": dataset_name,
            "task_type": "conditioned",
            "selected_path": rel(symbolic_conditioned),
            "source_path": rel(source),
            "score": selected["score"],
            "note_count": selected["note_count"],
            "duration_seconds": selected["duration_seconds"],
            "notes_per_second": selected["notes_per_second"],
            "pitch_range": selected["pitch_range"],
            "max_simultaneous_notes": selected["max_simultaneous_notes"],
            "repeated_pitch_bigram_rate": selected["repeated_pitch_bigram_rate"],
            "reject_reason": selected.get("reject_reason", ""),
            "conditioned_prefix_only_path": rel(run_dir / "conditioned_prefix_only.mid"),
            "conditioned_full_with_prefix_path": rel(full_with_prefix),
            "conditioned_continuation_only_path": rel(continuation_only),
        }
        for field in CONDITIONED_FIELDS:
            selected_row[field] = selected.get(field, "")
        selected_rows.append(selected_row)

    config = {
        "run": run_dir.name,
        "dataset": dataset_name,
        "task": "task2_conditioned_bigpool",
        "model_type": "transformer",
        "checkpoint": rel(checkpoint),
        "generate_tokens": generate_tokens,
        "conditioned_prefix_count": len(prefixes),
        "conditioned_prefix_tokens": max(len(prefix_ids) for _idx, _path, prefix_ids in prefixes),
        "candidate_count_per_prefix": candidates_per_prefix,
        "temperatures": temperatures,
        "top_ks": top_ks,
        "seed": seed,
        "selection_rule": "highest conditioned_continuation_score among continuation-only usable candidates",
    }
    notes = [
        f"{run_dir.name}: Task 2 conditioned bigpool Transformer run.",
        "Ranking and selection use continuation-only metrics, not full_with_prefix metrics.",
        "symbolic_conditioned.mid and conditioned_full_with_prefix.mid include the selected real prefix.",
    ]
    write_run_common(run_dir, ranking_rows, selected_rows, config, notes)


def generate_markov_baseline(
    run_dir: Path,
    tokenizer: object,
    dataset_name: str,
    train_windows: list[list[int]],
    prefixes: list[tuple[int, Path, list[int]]],
    generate_tokens: int,
    unconditioned_count: int,
    conditioned_count_per_prefix: int,
    seed: int,
) -> None:
    markov = NGramModel(order=3, seed=seed)
    markov.fit(train_windows)
    candidates_dir = run_dir / "candidates"
    prefix_dir = run_dir / "prefixes"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    prefix_dir.mkdir(parents=True, exist_ok=True)
    ranking_rows: list[dict[str, object]] = []
    train_seed = train_windows[0][0]
    for index in range(unconditioned_count):
        ids = markov.sample(generate_tokens, prefix=[train_seed])
        path = candidates_dir / f"markov_unconditioned_idx{index:03d}.mid"
        valid, notes, error, selected_ids = decode_with_retries(tokenizer, [ids], path)
        row = full_metrics(path, dataset_name, path)
        row.update(
            {
                "model_type": "markov",
                "valid": valid,
                "note_count": notes if valid else row["note_count"],
                "candidate_index": index,
                "generation_mode": "markov_unconditioned",
                "prefix_token_count": 1,
                "continuation_token_count": max(0, len(selected_ids) - 1),
                "error": error,
            }
        )
        ranking_rows.append(row)

    for prefix_id, prefix_file, prefix_ids in prefixes:
        prefix_path = prefix_dir / f"conditioned_prefix_{prefix_id:03d}.mid"
        prefix_metrics, _duration_ticks = prefix_metrics_for_ids(tokenizer, prefix_ids, prefix_path)
        for local_index in range(conditioned_count_per_prefix):
            global_index = prefix_id * conditioned_count_per_prefix + local_index
            ids = markov.sample(generate_tokens, prefix=prefix_ids)
            path = candidates_dir / f"markov_conditioned_prefix{prefix_id:03d}_idx{global_index:04d}.mid"
            valid, notes, error, selected_ids = decode_with_retries(tokenizer, [ids], path)
            row = full_metrics(path, dataset_name, path)
            row.update(
                {
                    "model_type": "markov",
                    "valid": valid,
                    "note_count": notes if valid else row["note_count"],
                    "candidate_index": global_index,
                    "generation_mode": "markov_conditioned",
                    "prefix_token_count": len(prefix_ids),
                    "continuation_token_count": max(0, len(selected_ids) - len(prefix_ids)),
                    "error": error,
                }
            )
            row.update(
                continuation_metrics(
                    path,
                    prefix_metrics,
                    len(prefix_ids),
                    max(0, len(selected_ids) - len(prefix_ids)),
                    "conditioned",
                    prefix_id=prefix_id,
                    prefix_file=prefix_file,
                )
            )
            ranking_rows.append(row)

    unconditioned = best_full(ranking_rows, "unconditioned")
    conditioned = best_continuation(ranking_rows, "conditioned_continuation")
    selected_rows: list[dict[str, object]] = []
    if unconditioned is not None:
        source = resolve(unconditioned["source_path"])
        target = run_dir / "symbolic_unconditioned_markov.mid"
        write_selected_copy(source, target)
        selected_rows.append(
            {
                "run": run_dir.name,
                "dataset": dataset_name,
                "task_type": "unconditioned",
                "model_type": "markov",
                "selected_path": rel(target),
                "source_path": rel(source),
                "score": unconditioned["score"],
                "note_count": unconditioned["note_count"],
                "duration_seconds": unconditioned["duration_seconds"],
                "notes_per_second": unconditioned["notes_per_second"],
                "pitch_range": unconditioned["pitch_range"],
                "max_simultaneous_notes": unconditioned["max_simultaneous_notes"],
                "repeated_pitch_bigram_rate": unconditioned["repeated_pitch_bigram_rate"],
                "reject_reason": unconditioned.get("reject_reason", ""),
            }
        )
    else:
        selected_rows.append(
            {
                "run": run_dir.name,
                "dataset": dataset_name,
                "task_type": "unconditioned",
                "model_type": "markov",
                "selected_path": "",
                "source_path": "",
                "reject_reason": "No usable Markov unconditioned candidate found.",
            }
        )

    if conditioned is not None:
        source = resolve(conditioned["source_path"])
        prefix_id = int(conditioned["conditioned_prefix_id"])
        prefix_path = prefix_dir / f"conditioned_prefix_{prefix_id:03d}.mid"
        prefix_metrics = asdict(analyze_candidate(prefix_path))
        full_with_prefix = run_dir / "conditioned_full_with_prefix_markov.mid"
        continuation_only = run_dir / "conditioned_continuation_only_markov.mid"
        symbolic_conditioned = run_dir / "symbolic_conditioned_markov.mid"
        write_selected_copy(prefix_path, run_dir / "conditioned_prefix_only_markov.mid")
        write_selected_copy(source, full_with_prefix)
        write_selected_copy(source, symbolic_conditioned)
        full_notes, ticks_per_beat = read_note_events_with_velocity(source)
        split_tick = round(float(prefix_metrics["duration_seconds"]) / 0.5 * ticks_per_beat)
        write_note_events(continuation_only, crop_after(full_notes, split_tick), ticks_per_beat)
        selected_row = {
            "run": run_dir.name,
            "dataset": dataset_name,
            "task_type": "conditioned",
            "model_type": "markov",
            "selected_path": rel(symbolic_conditioned),
            "source_path": rel(source),
            "score": conditioned["score"],
            "note_count": conditioned["note_count"],
            "duration_seconds": conditioned["duration_seconds"],
            "notes_per_second": conditioned["notes_per_second"],
            "pitch_range": conditioned["pitch_range"],
            "max_simultaneous_notes": conditioned["max_simultaneous_notes"],
            "repeated_pitch_bigram_rate": conditioned["repeated_pitch_bigram_rate"],
            "reject_reason": conditioned.get("reject_reason", ""),
            "conditioned_prefix_only_path": rel(run_dir / "conditioned_prefix_only_markov.mid"),
            "conditioned_full_with_prefix_path": rel(full_with_prefix),
            "conditioned_continuation_only_path": rel(continuation_only),
        }
        for field in CONDITIONED_FIELDS:
            selected_row[field] = conditioned.get(field, "")
        selected_rows.append(selected_row)
    else:
        selected_rows.append(
            {
                "run": run_dir.name,
                "dataset": dataset_name,
                "task_type": "conditioned",
                "model_type": "markov",
                "selected_path": "",
                "source_path": "",
                "reject_reason": "No usable Markov conditioned continuation-only candidate found.",
            }
        )

    config = {
        "run": run_dir.name,
        "dataset": dataset_name,
        "task": "markov_baseline",
        "model_type": "markov",
        "markov_order": 3,
        "generate_tokens": generate_tokens,
        "unconditioned_candidate_count": unconditioned_count,
        "conditioned_prefix_count": len(prefixes),
        "conditioned_candidate_count_per_prefix": conditioned_count_per_prefix,
        "seed": seed,
        "selection_rule": "full score for unconditioned; continuation-only score for conditioned",
    }
    notes = [
        "Markov / n-gram baseline on the same Nottingham split.",
        "This is a comparison baseline and possible fallback, not the Transformer final model.",
        "Conditioned Markov selection uses continuation-only metrics.",
    ]
    write_run_common(run_dir, ranking_rows, selected_rows, config, notes)


def aggregate_tables(indexed_dir: Path, evaluation_dir: Path) -> None:
    ranking_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in indexed_dir.iterdir() if path.is_dir()):
        ranking_path = run_dir / "candidate_ranking.csv"
        selected_path = run_dir / "selected_candidates.csv"
        if ranking_path.exists():
            for row in read_csv_rows(ranking_path):
                row["run"] = run_dir.name
                ranking_rows.append(row)
        if selected_path.exists():
            selected_rows.extend(read_csv_rows(selected_path))
    tables_dir = evaluation_dir / "tables"
    write_csv(tables_dir / "candidate_ranking.csv", ranking_rows)
    write_csv(tables_dir / "selected_candidates.csv", selected_rows)


def main() -> None:
    parser = ArgumentParser(description="Generate Nottingham final retrain runs.")
    parser.add_argument("--metrics-dir", default="outputs/metrics/nottingham_final_retrain")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/nottingham_final_retrain/best_transformer.pt")
    parser.add_argument("--indexed-dir", default="outputs/candidates/final/nottingham_final_retrain")
    parser.add_argument("--evaluation-dir", default="outputs/evaluation/nottingham_final_retrain")
    parser.add_argument("--generate-tokens", type=int, default=768)
    parser.add_argument("--task1-candidate-count", type=int, default=100)
    parser.add_argument("--conditioned-prefix-count", type=int, default=20)
    parser.add_argument("--conditioned-prefix-tokens", type=int, default=128)
    parser.add_argument("--conditioned-candidates-per-prefix", type=int, default=10)
    parser.add_argument("--temperatures", default="0.7,0.8,0.9")
    parser.add_argument("--top-ks", default="20,50")
    parser.add_argument("--seed", type=int, default=253)
    args = parser.parse_args()

    random.seed(args.seed)
    metrics_dir = resolve(args.metrics_dir)
    checkpoint = resolve(args.checkpoint)
    indexed_dir = resolve(args.indexed_dir)
    evaluation_dir = resolve(args.evaluation_dir)
    summary = load_summary(metrics_dir)
    dataset_name = str(summary["dataset_name"])
    tokenizer = tokenizer_from_summary(summary)
    train_files, valid_files = split_files_from_manifest(summary)
    train_sequences = sequences_for_files(tokenizer, train_files)
    valid_sequences = sequences_for_files(tokenizer, valid_files)
    block_size = int(summary["transformer"]["block_size"])
    stride = int(summary.get("stride", 256) or 256)
    train_windows = make_lm_windows(list(train_sequences.values()), block_size, stride)
    valid_windows = make_lm_windows(list(valid_sequences.values()), block_size, stride)
    if not train_windows or not valid_windows:
        raise ValueError("Empty train/valid windows")
    model, _model_metrics = load_transformer_for_generation(
        checkpoint_path=checkpoint,
        fallback_vocab_size=int(summary["vocab_size"]),
        fallback_block_size=block_size,
    )
    temperatures = [float(value.strip()) for value in args.temperatures.split(",") if value.strip()]
    top_ks = [int(value.strip()) for value in args.top_ks.split(",") if value.strip()]
    vocab_size = int(summary["vocab_size"])

    valid_prefix_pool = valid_prefixes(
        tokenizer,
        valid_sequences,
        args.conditioned_prefix_count,
        args.conditioned_prefix_tokens,
        args.seed,
    )
    structural_prefix = valid_windows[0][: min(32, block_size // 2, len(valid_windows[0]))]
    valid_items = [(path, ids) for path, ids in valid_sequences.items() if ids]
    piece_file_128, piece_ids_128 = valid_items[min(args.seed, len(valid_items) - 1)]
    piece_file_256, piece_ids_256 = valid_items[min(args.seed, len(valid_items) - 1)]

    generate_task1_run(
        indexed_dir / "run_uncond_bos",
        model,
        tokenizer,
        dataset_name,
        "pure_bos",
        [train_windows[0][0]],
        None,
        args.generate_tokens,
        args.task1_candidate_count,
        temperatures,
        top_ks,
        vocab_size,
        block_size,
        checkpoint,
        args.seed,
    )
    generate_task1_run(
        indexed_dir / "run_uncond_structural_seeded",
        model,
        tokenizer,
        dataset_name,
        "structural_seeded",
        structural_prefix,
        valid_files[0],
        args.generate_tokens,
        args.task1_candidate_count,
        temperatures,
        top_ks,
        vocab_size,
        block_size,
        checkpoint,
        args.seed,
    )
    generate_task1_run(
        indexed_dir / "run_uncond_piece_start_128",
        model,
        tokenizer,
        dataset_name,
        "piece_start_seeded",
        piece_ids_128[: min(128, len(piece_ids_128))],
        piece_file_128,
        args.generate_tokens,
        args.task1_candidate_count,
        temperatures,
        top_ks,
        vocab_size,
        block_size,
        checkpoint,
        args.seed,
    )
    generate_task1_run(
        indexed_dir / "run_uncond_piece_start_256",
        model,
        tokenizer,
        dataset_name,
        "piece_start_seeded",
        piece_ids_256[: min(256, len(piece_ids_256))],
        piece_file_256,
        args.generate_tokens,
        args.task1_candidate_count,
        temperatures,
        top_ks,
        vocab_size,
        block_size,
        checkpoint,
        args.seed,
    )
    generate_conditioned_bigpool(
        indexed_dir / "run_conditioned_bigpool",
        model,
        tokenizer,
        dataset_name,
        valid_prefix_pool,
        args.generate_tokens,
        args.conditioned_candidates_per_prefix,
        temperatures,
        top_ks,
        vocab_size,
        block_size,
        checkpoint,
        args.seed,
    )
    generate_markov_baseline(
        indexed_dir / "baseline_markov",
        tokenizer,
        dataset_name,
        train_windows,
        valid_prefix_pool,
        args.generate_tokens,
        args.task1_candidate_count,
        args.conditioned_candidates_per_prefix,
        args.seed,
    )
    aggregate_tables(indexed_dir, evaluation_dir)
    print(
        json.dumps(
            {
                "indexed_dir": rel(indexed_dir),
                "evaluation_tables": rel(evaluation_dir / "tables"),
                "runs": [
                    "run_uncond_bos",
                    "run_uncond_structural_seeded",
                    "run_uncond_piece_start_128",
                    "run_uncond_piece_start_256",
                    "run_conditioned_bigpool",
                    "baseline_markov",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
