"""Run a bounded symbolic MIDI pipeline on a small MIDI subset."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    DEFAULT_GENERATE_TOKENS,
    DEFAULT_MARKOV_ORDER,
    DEFAULT_STRIDE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TRANSFORMER_BATCH_SIZE,
    DEFAULT_TRANSFORMER_EMBED,
    DEFAULT_TRANSFORMER_EPOCHS,
    DEFAULT_TRANSFORMER_HEADS,
    DEFAULT_TRANSFORMER_LAYERS,
    DEFAULT_TRANSFORMER_LR,
    DEFAULT_TRANSFORMER_MAX_STEPS,
    DEFAULT_VALID_FRACTION,
    SEED,
)
from src.data import (  # noqa: E402
    count_midi_notes,
    discover_midi_files,
    make_lm_windows,
    pitch_class_histogram,
    split_files,
)
from src.markov import NGramModel  # noqa: E402
from src.tokenizers import REMITokenizerSmoke, SimpleMIDITokenizer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def try_tokenizer(midi_files: list[Path], output_dir: Path) -> tuple[object, list[list[int]], str, str]:
    last_error = ""
    for num_velocities in (8, 4, 1):
        try:
            tokenizer = REMITokenizerSmoke(num_velocities=num_velocities)
            tokenizer.fit(midi_files)
            sequence_by_file, _skipped = encode_files(tokenizer, midi_files)
            sequences = list(sequence_by_file.values())
            if not sequences:
                raise ValueError("no files encoded with REMI")
            roundtrip = output_dir / f"roundtrip_remi_v{num_velocities}.mid"
            tokenizer.decode_ids(sequences[0], roundtrip)
            if count_midi_notes(roundtrip) == 0:
                raise ValueError("roundtrip decoded MIDI has zero notes")
            return tokenizer, sequences, "remi", f"remi_num_velocities_{num_velocities}"
        except Exception as exc:  # noqa: BLE001 - one focused fallback path.
            last_error = f"{type(exc).__name__}: {exc}"
    tokenizer = SimpleMIDITokenizer()
    tokenizer.fit(midi_files)
    sequences = [tokenizer.encode_file(path) for path in midi_files]
    return tokenizer, sequences, "simple_event", f"fallback_after_remi_error={last_error}"


def parse_label_set(raw: str) -> set[str]:
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def load_manifest_split(
    manifest_csv: Path,
    path_column: str,
    split_column: str,
    train_labels: set[str],
    valid_labels: set[str],
) -> tuple[list[Path], list[Path], list[dict[str, object]]]:
    train_files: list[Path] = []
    valid_files: list[Path] = []
    skipped: list[dict[str, object]] = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_path = row.get(path_column, "")
            split = row.get(split_column, "").strip().lower()
            if not raw_path:
                skipped.append({"file_path": "", "reason": "missing_manifest_path", "detail": ""})
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = ROOT / path
            if split in train_labels:
                train_files.append(path)
            elif split in valid_labels:
                valid_files.append(path)
            else:
                skipped.append(
                    {
                        "file_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                        "reason": "split_excluded",
                        "detail": split,
                    }
                )
    return train_files, valid_files, skipped


def validate_midi_files(midi_files: list[Path]) -> tuple[list[Path], list[dict[str, object]]]:
    valid_files: list[Path] = []
    skipped: list[dict[str, object]] = []
    for midi_path in midi_files:
        try:
            notes = count_midi_notes(midi_path)
            if notes <= 0:
                skipped.append(
                    {
                        "file_path": str(midi_path.relative_to(ROOT)) if midi_path.is_relative_to(ROOT) else str(midi_path),
                        "reason": "zero_note_midi",
                        "detail": "note_count=0",
                    }
                )
                continue
            valid_files.append(midi_path)
        except Exception as exc:  # noqa: BLE001 - record bad input files and keep the run moving.
            skipped.append(
                {
                    "file_path": str(midi_path.relative_to(ROOT)) if midi_path.is_relative_to(ROOT) else str(midi_path),
                    "reason": "midi_parse_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return valid_files, skipped


def encode_files(
    tokenizer: object,
    midi_files: list[Path],
) -> tuple[dict[Path, list[int]], list[dict[str, object]]]:
    sequences: dict[Path, list[int]] = {}
    skipped: list[dict[str, object]] = []
    for midi_path in midi_files:
        try:
            ids = tokenizer.encode_file(midi_path)
            if len(ids) < 2:
                skipped.append(
                    {
                        "file_path": str(midi_path.relative_to(ROOT)) if midi_path.is_relative_to(ROOT) else str(midi_path),
                        "reason": "too_few_tokens",
                        "detail": f"token_length={len(ids)}",
                    }
                )
                continue
            sequences[midi_path] = ids
        except Exception as exc:  # noqa: BLE001 - report tokenizer failures per file.
            skipped.append(
                {
                    "file_path": str(midi_path.relative_to(ROOT)) if midi_path.is_relative_to(ROOT) else str(midi_path),
                    "reason": "tokenize_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return sequences, skipped


def write_manifest(
    path: Path,
    midi_files: list[Path],
    splits: dict[Path, str],
    sequences: dict[Path, list[int]],
    source_dataset: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_path", "file_name", "source_dataset", "split", "note_count", "token_length"],
        )
        writer.writeheader()
        for midi_path in midi_files:
            writer.writerow(
                {
                    "file_path": str(midi_path.relative_to(ROOT)),
                    "file_name": midi_path.name,
                    "source_dataset": source_dataset,
                    "split": splits[midi_path],
                    "note_count": count_midi_notes(midi_path),
                    "token_length": len(sequences[midi_path]),
                }
            )


def write_skipped_files(path: Path, skipped: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "reason", "detail"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in skipped:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_batches(windows: list[list[int]], block_size: int, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    examples = []
    for window in windows:
        trimmed = window[: block_size + 1]
        if len(trimmed) < block_size + 1:
            trimmed = trimmed + [trimmed[-1]] * (block_size + 1 - len(trimmed))
        examples.append(trimmed)
    batches = []
    for start in range(0, len(examples), batch_size):
        batch = torch.tensor(examples[start : start + batch_size], dtype=torch.long)
        batches.append((batch[:, :-1], batch[:, 1:]))
    return batches


def evaluate_loss(model: GPT2LMHeadModel, batches: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in batches:
            result = model(input_ids=inputs.to(device), labels=labels.to(device))
            losses.append(float(result.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_transformer(
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
    checkpoint_dir: Path | None,
    eval_interval: int,
    resume_checkpoint: Path | None,
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
    eval_history: list[dict[str, float | int]] = []
    training_history: list[dict[str, float | int | None]] = []
    best_valid_loss = float("inf")
    best_checkpoint_path: Path | None = None
    resumed_from = None
    resumed_step = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        best_valid_loss = float(checkpoint.get("valid_loss", best_valid_loss))
        resumed_step = int(checkpoint.get("step", 0))
        resumed_from = str(resume_checkpoint.relative_to(ROOT))
    started = time.perf_counter()
    steps = 0
    model.train()
    for _epoch in range(epochs):
        random.shuffle(train_batches)
        for inputs, labels in train_batches:
            result = model(input_ids=inputs.to(device), labels=labels.to(device))
            loss = result.loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite transformer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if eval_interval > 0 and steps % eval_interval == 0:
                valid_loss = evaluate_loss(model, valid_batches, device)
                row = {
                    "step": steps,
                    "total_step": resumed_step + steps,
                    "train_loss": float(loss.detach().cpu()),
                    "valid_loss": valid_loss,
                    "valid_perplexity": math.exp(valid_loss) if valid_loss < 20 else float("inf"),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                eval_history.append({"step": steps, "valid_loss": valid_loss})
                training_history.append(row)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if checkpoint_dir is not None:
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        best_checkpoint_path = checkpoint_dir / "best_transformer.pt"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "config": config.to_dict(),
                                "step": resumed_step + steps,
                                "valid_loss": valid_loss,
                            },
                            best_checkpoint_path,
                        )
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
    valid_loss = evaluate_loss(model, valid_batches, device)
    training_history.append(
        {
            "step": steps,
            "total_step": resumed_step + steps,
            "train_loss": losses[-1] if losses else None,
            "valid_loss": valid_loss,
            "valid_perplexity": math.exp(valid_loss) if valid_loss < 20 else float("inf"),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = checkpoint_dir / "best_transformer.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "step": resumed_step + steps,
                    "valid_loss": valid_loss,
                },
                best_checkpoint_path,
            )
    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        valid_loss = float(checkpoint["valid_loss"])
    metrics = {
        "model_class": model.__class__.__name__,
        "pretrained_loaded": False,
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
        "total_steps_including_resume": resumed_step + steps,
        "resume_checkpoint": resumed_from,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_mean": sum(losses) / max(len(losses), 1),
        "valid_loss": valid_loss,
        "valid_perplexity": math.exp(valid_loss) if valid_loss < 20 else float("inf"),
        "losses_finite": all(math.isfinite(loss) for loss in losses) and math.isfinite(valid_loss),
        "best_valid_loss": best_valid_loss,
        "best_valid_perplexity": math.exp(best_valid_loss) if best_valid_loss < 20 else float("inf"),
        "best_checkpoint": str(best_checkpoint_path.relative_to(ROOT)) if best_checkpoint_path else None,
        "eval_history": eval_history,
        "training_history": training_history,
        "runtime_seconds": time.perf_counter() - started,
    }
    return model, metrics


def load_transformer_for_generation(
    checkpoint_path: Path,
    fallback_vocab_size: int,
    fallback_block_size: int,
) -> tuple[GPT2LMHeadModel, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get("config")
    if config_dict is None:
        config = GPT2Config(
            vocab_size=max(fallback_vocab_size, 8),
            n_positions=fallback_block_size,
            n_ctx=fallback_block_size,
            n_embd=128,
            n_layer=2,
            n_head=4,
            bos_token_id=0,
            eos_token_id=1,
        )
    else:
        config = GPT2Config(**config_dict)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    metrics = {
        "model_class": model.__class__.__name__,
        "pretrained_loaded": False,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "block_size": int(config.n_positions),
        "batch_size": None,
        "n_embd": int(config.n_embd),
        "n_layer": int(config.n_layer),
        "n_head": int(config.n_head),
        "dropout": float(config.resid_pdrop),
        "checkpoint_loaded": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "checkpoint_valid_loss": checkpoint.get("valid_loss"),
    }
    return model, metrics


def sample_transformer(
    model: GPT2LMHeadModel,
    prefix: list[int],
    max_new_tokens: int,
    vocab_size: int,
    block_size: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    device = next(model.parameters()).device
    output = list(prefix)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = output[-block_size:]
            inputs = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids=inputs).logits[0, -1, :] / max(temperature, 1e-6)
            k = min(top_k, vocab_size)
            values, indices = torch.topk(logits, k=k)
            probabilities = torch.softmax(values, dim=-1)
            next_id = int(indices[torch.multinomial(probabilities, num_samples=1)].item())
            output.append(next_id)
    return output


def decode_with_retries(tokenizer: object, candidates: list[list[int]], output_path: Path) -> tuple[bool, int, str]:
    errors = []
    for ids in candidates:
        try:
            tokenizer.decode_ids(ids, output_path)
            notes = count_midi_notes(output_path)
            if notes > 0:
                return True, notes, ""
            errors.append("decoded zero-note MIDI")
        except Exception as exc:  # noqa: BLE001 - reports candidate decode failures.
            errors.append(f"{type(exc).__name__}: {exc}")
    return False, 0, " | ".join(errors[-3:])


def parse_number_list(raw: str, cast: object) -> list[object]:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def format_temperature(value: float) -> str:
    return str(value).replace(".", "p")


def generate_transformer_candidates(
    model: GPT2LMHeadModel,
    tokenizer: object,
    train_seed: int,
    prefix: list[int],
    output_dir: Path,
    vocab_size: int,
    block_size: int,
    generate_tokens: int,
    temperatures: list[float],
    top_ks: list[int],
    candidate_count: int,
) -> dict[str, dict[str, object]]:
    outputs: dict[str, dict[str, object]] = {}
    for task_name, seed_prefix in {
        "unconditioned": [train_seed],
        "conditioned": prefix,
    }.items():
        for temperature in temperatures:
            for top_k in top_ks:
                for index in range(candidate_count):
                    max_new = generate_tokens - len(seed_prefix) if task_name == "conditioned" else generate_tokens - 1
                    ids = sample_transformer(
                        model,
                        prefix=seed_prefix,
                        max_new_tokens=max(1, max_new),
                        vocab_size=vocab_size,
                        block_size=block_size,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    name = (
                        f"transformer_{task_name}_temp{format_temperature(temperature)}_"
                        f"topk{top_k}_idx{index:02d}"
                    )
                    path = output_dir / f"{name}.mid"
                    valid, notes, error = decode_with_retries(tokenizer, [ids], path)
                    outputs[name] = {
                        "path": str(path.relative_to(ROOT)),
                        "valid": valid,
                        "note_count": notes,
                        "temperature": temperature,
                        "top_k": top_k,
                        "candidate_index": index,
                        "error": error,
                    }
    return outputs


def token_stats(sequences: list[list[int]]) -> dict[str, float]:
    lengths = [len(sequence) for sequence in sequences]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def write_pitch_hist(path: Path, histogram: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pitch_class", "count"])
        for pitch_class, count in enumerate(histogram):
            writer.writerow([pitch_class, count])


def write_token_lengths(path: Path, midi_files: list[Path], sequences: dict[Path, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "token_length"])
        for midi_path in midi_files:
            writer.writerow([midi_path.name, len(sequences[midi_path])])


def write_dataset_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "dataset_name",
        "file_count",
        "total_bytes",
        "train_file_count",
        "valid_file_count",
        "vocab_size",
        "train_window_count",
        "valid_window_count",
        "markov_valid_perplexity",
        "transformer_train_loss_last",
        "transformer_valid_loss",
        "transformer_valid_perplexity",
    ]
    row = {
        "dataset_name": summary["dataset_name"],
        "file_count": summary["file_count"],
        "total_bytes": summary["total_bytes"],
        "train_file_count": summary["train_file_count"],
        "valid_file_count": summary["valid_file_count"],
        "vocab_size": summary["vocab_size"],
        "train_window_count": summary["train_window_count"],
        "valid_window_count": summary["valid_window_count"],
        "markov_valid_perplexity": summary["markov"]["valid_perplexity"],
        "transformer_train_loss_last": summary["transformer"]["train_loss_last"],
        "transformer_valid_loss": summary["transformer"]["valid_loss"],
        "transformer_valid_perplexity": summary["transformer"]["valid_perplexity"],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerow(row)


def write_training_history(path: Path, history: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "total_step", "train_loss", "valid_loss", "valid_perplexity", "learning_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in fieldnames})


def parse_args() -> object:
    parser = ArgumentParser(description="Run a bounded symbolic MIDI training pipeline.")
    parser.add_argument("--mode", choices=["train_generate", "generate"], default="train_generate")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--manifest-path-column", default="file_path")
    parser.add_argument("--manifest-split-column", default="split")
    parser.add_argument("--train-split-labels", default="train")
    parser.add_argument("--valid-split-labels", default="validation,valid")
    parser.add_argument("--max-files", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--valid-fraction", type=float, default=DEFAULT_VALID_FRACTION)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRANSFORMER_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_TRANSFORMER_MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRANSFORMER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRANSFORMER_LR)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_TRANSFORMER_EMBED)
    parser.add_argument("--n-layer", type=int, default=DEFAULT_TRANSFORMER_LAYERS)
    parser.add_argument("--n-head", type=int, default=DEFAULT_TRANSFORMER_HEADS)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--generate-tokens", type=int, default=DEFAULT_GENERATE_TOKENS)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--temperatures", default=str(DEFAULT_TEMPERATURE))
    parser.add_argument("--top-ks", default=str(DEFAULT_TOP_K))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_absolute():
        metrics_dir = ROOT / metrics_dir
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None and not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume_checkpoint is not None and not resume_checkpoint.is_absolute():
        resume_checkpoint = ROOT / resume_checkpoint
    if args.mode == "generate" and resume_checkpoint is None:
        raise ValueError("--mode generate requires --resume-checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    skipped_files: list[dict[str, object]] = []
    manifest_source = None
    if args.manifest_csv:
        manifest_csv = Path(args.manifest_csv)
        if not manifest_csv.is_absolute():
            manifest_csv = ROOT / manifest_csv
        manifest_source = str(manifest_csv.relative_to(ROOT))
        train_files, valid_files, manifest_skipped = load_manifest_split(
            manifest_csv=manifest_csv,
            path_column=args.manifest_path_column,
            split_column=args.manifest_split_column,
            train_labels=parse_label_set(args.train_split_labels),
            valid_labels=parse_label_set(args.valid_split_labels),
        )
        skipped_files.extend(manifest_skipped)
        midi_files = train_files + valid_files
        if args.max_files > 0:
            train_limit = max(1, round(args.max_files * (1.0 - args.valid_fraction)))
            valid_limit = max(1, args.max_files - train_limit)
            train_files = train_files[:train_limit]
            valid_files = valid_files[:valid_limit]
            midi_files = train_files + valid_files
    else:
        discovered_files = discover_midi_files([input_dir])
        midi_files = discovered_files if args.max_files <= 0 else discovered_files[: args.max_files]
        if len(midi_files) < 2:
            raise ValueError("Need at least two MIDI files for train/validation split")
        train_files, valid_files = split_files(midi_files, args.valid_fraction)

    midi_files, validation_skipped = validate_midi_files(midi_files)
    skipped_files.extend(validation_skipped)
    midi_file_set = set(midi_files)
    train_files = [path for path in train_files if path in midi_file_set]
    valid_files = [path for path in valid_files if path in midi_file_set]
    if len(train_files) < 1 or len(valid_files) < 1:
        raise ValueError("Need at least one valid train and validation MIDI file")

    tokenizer, _initial_sequences, tokenizer_mode, tokenizer_detail = try_tokenizer(midi_files, output_dir)
    sequence_by_file, tokenization_skipped = encode_files(tokenizer, midi_files)
    skipped_files.extend(tokenization_skipped)
    midi_files = [path for path in midi_files if path in sequence_by_file]
    train_files = [path for path in train_files if path in sequence_by_file]
    valid_files = [path for path in valid_files if path in sequence_by_file]
    if len(train_files) < 1 or len(valid_files) < 1:
        raise ValueError("Need at least one tokenized train and validation MIDI file")
    splits = {path: "train" for path in train_files}
    splits.update({path: "valid" for path in valid_files})

    train_sequences = [sequence_by_file[path] for path in train_files]
    valid_sequences = [sequence_by_file[path] for path in valid_files]
    train_windows = make_lm_windows(train_sequences, args.block_size, args.stride)
    valid_windows = make_lm_windows(valid_sequences, args.block_size, args.stride)
    if not train_windows or not valid_windows:
        raise ValueError("Train/validation windows are empty")

    manifest_path = metrics_dir / "manifest.csv"
    write_manifest(manifest_path, midi_files, splits, sequence_by_file, args.dataset_name)
    skipped_files_path = metrics_dir / "skipped_files.csv"
    write_skipped_files(skipped_files_path, skipped_files)

    markov = NGramModel(order=DEFAULT_MARKOV_ORDER, seed=SEED)
    markov.fit(train_windows)
    markov_perplexity = markov.perplexity(valid_windows)

    if args.mode == "generate":
        model, transformer_metrics = load_transformer_for_generation(
            checkpoint_path=resume_checkpoint,
            fallback_vocab_size=tokenizer.vocab_size,
            fallback_block_size=args.block_size,
        )
        valid_batches = make_batches(valid_windows, int(transformer_metrics["block_size"]), args.batch_size)
        valid_loss = evaluate_loss(model, valid_batches, next(model.parameters()).device)
        transformer_metrics.update(
            {
                "steps_completed": 0,
                "total_steps_including_resume": transformer_metrics["checkpoint_step"],
                "train_loss_first": None,
                "train_loss_last": None,
                "train_loss_mean": None,
                "valid_loss": valid_loss,
                "valid_perplexity": math.exp(valid_loss) if valid_loss < 20 else float("inf"),
                "losses_finite": math.isfinite(valid_loss),
                "best_checkpoint": str(resume_checkpoint.relative_to(ROOT)),
                "runtime_seconds": 0.0,
            }
        )
    else:
        model, transformer_metrics = train_transformer(
            train_windows=train_windows,
            valid_windows=valid_windows,
            vocab_size=tokenizer.vocab_size,
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
            resume_checkpoint=resume_checkpoint,
        )

    generation_block_size = int(transformer_metrics["block_size"])
    prefix = valid_windows[0][: min(generation_block_size // 2, len(valid_windows[0]))]
    markov_uncond = markov.sample(args.generate_tokens, prefix=[train_windows[0][0]])
    markov_cond = markov.sample(args.generate_tokens, prefix=prefix)
    temperatures = [float(value) for value in parse_number_list(args.temperatures, float)]
    top_ks = [int(value) for value in parse_number_list(args.top_ks, int)]

    outputs = {}
    for name, candidates in {
        "markov_unconditioned": [markov_uncond],
        "markov_conditioned": [markov_cond],
    }.items():
        path = output_dir / f"{name}.mid"
        valid, notes, error = decode_with_retries(tokenizer, candidates, path)
        outputs[name] = {"path": str(path.relative_to(ROOT)), "valid": valid, "note_count": notes, "error": error}
    outputs.update(
        generate_transformer_candidates(
            model=model,
            tokenizer=tokenizer,
            train_seed=train_windows[0][0],
            prefix=prefix,
            output_dir=output_dir,
            vocab_size=tokenizer.vocab_size,
            block_size=generation_block_size,
            generate_tokens=args.generate_tokens,
            temperatures=temperatures,
            top_ks=top_ks,
            candidate_count=args.candidate_count,
        )
    )

    pitch_hist = pitch_class_histogram(midi_files)
    pitch_hist_path = metrics_dir / "pitch_class_histogram.csv"
    token_lengths_path = metrics_dir / "token_length_distribution.csv"
    dataset_summary_path = metrics_dir / "dataset_summary.csv"
    training_history_path = metrics_dir / "training_history.csv"
    write_pitch_hist(pitch_hist_path, pitch_hist)
    write_token_lengths(token_lengths_path, midi_files, sequence_by_file)
    training_history = transformer_metrics.get("training_history", [])
    if training_history or not training_history_path.exists():
        write_training_history(training_history_path, training_history)

    summary = {
        "dataset_name": args.dataset_name,
        "input_dir": str(input_dir.relative_to(ROOT)),
        "manifest_source": manifest_source,
        "file_count": len(midi_files),
        "skipped_file_count": len(skipped_files),
        "skipped_file_reasons": {
            reason: sum(1 for row in skipped_files if row.get("reason") == reason)
            for reason in sorted({str(row.get("reason")) for row in skipped_files})
        },
        "total_bytes": sum(path.stat().st_size for path in midi_files),
        "train_file_count": len(train_files),
        "valid_file_count": len(valid_files),
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_detail": tokenizer_detail,
        "vocab_size": tokenizer.vocab_size,
        "token_length_stats": token_stats(list(sequence_by_file.values())),
        "train_window_count": len(train_windows),
        "valid_window_count": len(valid_windows),
        "markov": {
            "order": markov.order,
            "valid_perplexity": markov_perplexity,
        },
        "transformer": transformer_metrics,
        "outputs": outputs,
        "generation": {
            "generate_tokens": args.generate_tokens,
            "temperatures": temperatures,
            "top_ks": top_ks,
            "candidate_count_per_setting": args.candidate_count,
            "transformer_candidate_count": len([key for key in outputs if key.startswith("transformer_")]),
        },
        "metrics": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "skipped_files": str(skipped_files_path.relative_to(ROOT)),
            "training_history": str(training_history_path.relative_to(ROOT)),
            "pitch_class_histogram": str(pitch_hist_path.relative_to(ROOT)),
            "token_length_distribution": str(token_lengths_path.relative_to(ROOT)),
            "dataset_summary": str(dataset_summary_path.relative_to(ROOT)),
        },
    }
    write_dataset_summary(dataset_summary_path, summary)
    summary_path = metrics_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
