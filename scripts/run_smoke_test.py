"""Run the minimal Phase 1 symbolic MIDI smoke test."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    count_midi_notes,
    create_synthetic_smoke_midis,
    discover_midi_files,
    make_lm_windows,
    split_train_valid,
)
from src.markov import NGramModel  # noqa: E402
from src.tokenizers import REMITokenizerSmoke, SimpleMIDITokenizer  # noqa: E402


def _try_remi(midi_files: list[Path], output_dir: Path) -> tuple[object, list[list[int]], str]:
    last_error = ""
    for num_velocities in (4, 1):
        try:
            tokenizer = REMITokenizerSmoke(num_velocities=num_velocities)
            tokenizer.fit(midi_files)
            sequences = [tokenizer.encode_file(path) for path in midi_files]
            if not any(len(sequence) > 1 for sequence in sequences):
                raise ValueError("REMI produced no usable token sequences")
            roundtrip_path = output_dir / f"remi_roundtrip_v{num_velocities}.mid"
            tokenizer.decode_ids(sequences[0], roundtrip_path)
            if count_midi_notes(roundtrip_path) == 0:
                raise ValueError("REMI round-trip MIDI contains zero notes")
            return tokenizer, sequences, f"remi_num_velocities_{num_velocities}"
        except Exception as exc:  # noqa: BLE001 - smoke test reports and falls back.
            last_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last_error)


def _use_simple_tokenizer(midi_files: list[Path]) -> tuple[SimpleMIDITokenizer, list[list[int]]]:
    tokenizer = SimpleMIDITokenizer()
    tokenizer.fit(midi_files)
    sequences = [tokenizer.encode_file(path) for path in midi_files]
    return tokenizer, sequences


def _run_tiny_gpt_smoke(train_windows: list[list[int]], vocab_size: int) -> dict[str, object]:
    block_size = min(16, max(len(window) - 1 for window in train_windows))
    examples = []
    for window in train_windows[:2]:
        trimmed = window[: block_size + 1]
        if len(trimmed) < block_size + 1:
            trimmed = trimmed + [trimmed[-1]] * (block_size + 1 - len(trimmed))
        examples.append(trimmed)
    batch = torch.tensor(examples, dtype=torch.long)
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    config = GPT2Config(
        vocab_size=max(vocab_size, 8),
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(input_ids=inputs, labels=labels)
    loss_before = float(output.loss.detach().cpu())
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    output_after = model(input_ids=inputs, labels=labels)
    return {
        "model_class": model.__class__.__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "block_size": block_size,
        "loss_before": loss_before,
        "loss_after": float(output_after.loss.detach().cpu()),
        "loss_is_finite": bool(torch.isfinite(output_after.loss).item()),
        "pretrained_loaded": False,
    }


def parse_args() -> object:
    parser = ArgumentParser(description="Run a minimal symbolic MIDI smoke test.")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="MIDI directory to scan. Can be passed more than once. Defaults to data and docs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "smoke_test"),
        help="Directory for smoke-test outputs.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=12,
        help="Maximum number of MIDI files to use.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Create synthetic MIDI files if no input MIDI files are found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(path) for path in args.input_dir] if args.input_dir else [ROOT / "data", ROOT / "docs"]
    roots = [path if path.is_absolute() else ROOT / path for path in roots]
    midi_files = discover_midi_files(roots)
    midi_files = [
        path
        for path in midi_files
        if "data\\samples\\smoke_test" not in str(path.relative_to(ROOT))
    ][: args.max_files]
    used_synthetic = False
    if not midi_files:
        if not args.allow_synthetic:
            raise FileNotFoundError(
                "No MIDI files found. Pass --allow-synthetic for synthetic smoke data."
            )
        midi_files = create_synthetic_smoke_midis(ROOT / "data" / "samples" / "smoke_test")
        used_synthetic = True

    tokenizer_mode = "remi"
    tokenizer_detail = ""
    remi_error = ""
    try:
        tokenizer, sequences, tokenizer_detail = _try_remi(midi_files, output_dir)
    except Exception as exc:  # noqa: BLE001 - requested fallback behavior.
        remi_error = f"{type(exc).__name__}: {exc}"
        tokenizer, sequences = _use_simple_tokenizer(midi_files)
        tokenizer_mode = tokenizer.name
        tokenizer_detail = "fallback_simple_event"

    block_size = 16
    windows = make_lm_windows(sequences, block_size=block_size, stride=8)
    train_windows, valid_windows = split_train_valid(windows)

    markov = NGramModel(order=3, seed=7)
    markov.fit(train_windows)
    valid_perplexity = markov.perplexity(valid_windows)

    unconditioned_ids = markov.sample(length=64, prefix=[train_windows[0][0]])
    prefix = valid_windows[0][: min(8, len(valid_windows[0]))]
    conditioned_ids = markov.sample(length=64, prefix=prefix)

    unconditioned_path = output_dir / "smoke_unconditioned.mid"
    conditioned_path = output_dir / "smoke_conditioned.mid"
    tokenizer.decode_ids(unconditioned_ids, unconditioned_path)
    tokenizer.decode_ids(conditioned_ids, conditioned_path)

    unconditioned_notes = count_midi_notes(unconditioned_path)
    conditioned_notes = count_midi_notes(conditioned_path)
    if unconditioned_notes == 0 or conditioned_notes == 0:
        raise RuntimeError("Smoke-test MIDI output contains zero notes")

    gpt_smoke = _run_tiny_gpt_smoke(train_windows, tokenizer.vocab_size)

    summary = {
        "midi_source": "synthetic_smoke_test" if used_synthetic else "local_existing",
        "midi_files": [str(path.relative_to(ROOT)) for path in midi_files],
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_detail": tokenizer_detail,
        "remi_error": remi_error,
        "vocab_size": tokenizer.vocab_size,
        "sequence_lengths": [len(sequence) for sequence in sequences],
        "window_count": len(windows),
        "train_window_count": len(train_windows),
        "valid_window_count": len(valid_windows),
        "markov_order": markov.order,
        "markov_valid_perplexity": valid_perplexity,
        "gpt_smoke": gpt_smoke,
        "outputs": {
            "unconditioned_midi": str(unconditioned_path.relative_to(ROOT)),
            "conditioned_midi": str(conditioned_path.relative_to(ROOT)),
            "unconditioned_note_count": unconditioned_notes,
            "conditioned_note_count": conditioned_notes,
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
