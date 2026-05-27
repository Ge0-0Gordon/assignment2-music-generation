"""Diagnose MAESTRO candidate MIDI quality and write a candidate diagnostics table."""

from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import mido


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import analyze_candidate  # noqa: E402


def resolve(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def infer_source_group(path: Path) -> str:
    parts = path.relative_to(ROOT).parts if path.is_relative_to(ROOT) else path.parts
    if "final" in parts:
        idx = parts.index("final")
        if idx + 2 < len(parts):
            return "/".join(parts[: idx + 3])
    if "selected" in parts:
        idx = parts.index("selected")
        if idx + 1 < len(parts):
            return "/".join(parts[: idx + 2])
    if "maestro_full" in parts:
        idx = parts.index("maestro_full")
        return "/".join(parts[: idx + 1])
    return str(path.parent)


def midi_event_distribution(path: Path) -> dict[str, int]:
    counts = {
        "pitch_token_count": 0,
        "bar_token_count": 0,
        "position_token_count": 0,
        "duration_token_count": 0,
        "velocity_token_count": 0,
        "other_token_count": 0,
    }
    try:
        midi = mido.MidiFile(path)
    except Exception:
        return counts
    ticks_per_bar = max(1, (midi.ticks_per_beat or 480) * 4)
    positions: set[int] = set()
    bars: set[int] = set()
    durations: list[int] = []
    for track in midi.tracks:
        active: dict[int, list[tuple[int, int]]] = {}
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
                counts["pitch_token_count"] += 1
                counts["velocity_token_count"] += 1
                positions.add(absolute % ticks_per_bar)
                bars.add(absolute // ticks_per_bar)
                active.setdefault(int(message.note), []).append((absolute, int(message.velocity)))
            elif message.type in {"note_off", "note_on"}:
                starts = active.get(int(message.note), [])
                if starts:
                    start, _velocity = starts.pop(0)
                    durations.append(max(1, absolute - start))
            elif not message.is_meta:
                counts["other_token_count"] += 1
    counts["bar_token_count"] = len(bars)
    counts["position_token_count"] = len(positions)
    counts["duration_token_count"] = len(durations)
    return counts


def candidate_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        root = resolve(raw)
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*.mid") if path.is_file() and not path.name.startswith("roundtrip_"))
        files.extend(path for path in root.rglob("*.midi") if path.is_file() and not path.name.startswith("roundtrip_"))
    return sorted(set(files))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    default_fieldnames = [
        "path",
        "source_group",
        "dataset",
        "task_type",
        "model_type",
        "temperature",
        "top_k",
        "candidate_index",
        "valid",
        "note_count",
        "duration_seconds",
        "notes_per_second",
        "pitch_min",
        "pitch_max",
        "pitch_range",
        "unique_pitch_count",
        "max_simultaneous_notes",
        "repeated_pitch_bigram_rate",
        "usable",
        "reject_reason",
        "score",
        "pitch_token_count",
        "bar_token_count",
        "position_token_count",
        "duration_token_count",
        "velocity_token_count",
        "other_token_count",
        "decoded_note_count",
        "generated_token_count",
        "pitch_token_ratio",
        "decoded_notes_per_100_tokens",
    ]
    fieldnames = default_fieldnames
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_generation_summaries(paths: list[str]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for raw in paths:
        path = resolve(raw)
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        for record in summary.get("outputs", {}).values():
            candidate_path = record.get("path")
            if not candidate_path:
                continue
            records[str(candidate_path).replace("\\", "/")] = record
    return records


def main() -> None:
    parser = ArgumentParser(description="Diagnose MAESTRO candidate MIDI files.")
    parser.add_argument(
        "--candidate-dirs",
        nargs="+",
        default=[
            "outputs/candidates/maestro_full",
            "outputs/candidates/final/maestro",
            "outputs/candidates/selected/maestro_full",
        ],
    )
    parser.add_argument("--output-csv", default="outputs/evaluation/maestro_full/tables/candidate_diagnostics.csv")
    parser.add_argument("--metrics-summaries", nargs="*", default=["outputs/metrics/maestro_full/summary.json"])
    args = parser.parse_args()

    generation_records = load_generation_summaries(args.metrics_summaries)
    rows = []
    for path in candidate_files(args.candidate_dirs):
        metrics = asdict(analyze_candidate(path))
        metrics["path"] = str(path.relative_to(ROOT))
        metrics["source_group"] = infer_source_group(path)
        distribution = midi_event_distribution(path)
        metrics.update(distribution)
        metrics["decoded_note_count"] = metrics["note_count"]
        generation_record = generation_records.get(metrics["path"].replace("\\", "/"))
        if generation_record:
            for key in (
                "generated_token_count",
                "pitch_token_count",
                "bar_token_count",
                "position_token_count",
                "duration_token_count",
                "velocity_token_count",
                "other_token_count",
                "decoded_note_count",
                "pitch_token_ratio",
                "decoded_notes_per_100_tokens",
            ):
                if key in generation_record:
                    metrics[key] = generation_record[key]
        rows.append(metrics)
    rows.sort(key=lambda row: (row["task_type"], str(row["source_group"]), -float(row["score"])))

    output_csv = resolve(args.output_csv)
    write_csv(output_csv, rows)
    summary = {
        "candidate_count": len(rows),
        "usable_count": sum(1 for row in rows if row["usable"]),
        "unconditioned_count": sum(1 for row in rows if row["task_type"] == "unconditioned"),
        "unconditioned_usable": sum(1 for row in rows if row["task_type"] == "unconditioned" and row["usable"]),
        "conditioned_count": sum(1 for row in rows if row["task_type"] == "conditioned"),
        "conditioned_usable": sum(1 for row in rows if row["task_type"] == "conditioned" and row["usable"]),
        "output_csv": str(output_csv.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
