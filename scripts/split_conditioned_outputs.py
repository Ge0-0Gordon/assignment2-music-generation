"""Split conditioned indexed runs into prefix/full/continuation MIDI files."""

from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from split_piece_start_outputs import (  # noqa: E402
    crop_after,
    load_generation_records,
    metrics_from_notes,
    read_csv_rows,
    read_note_events_with_velocity,
    resolve,
    safe_copy,
    tokenizer_from_summary,
    write_csv,
    write_note_events,
)
from src.data import make_lm_windows  # noqa: E402
from src.evaluate import analyze_candidate, infer_candidate_labels, infer_sampling_settings  # noqa: E402


CONDITIONED_FIELDS = [
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


def first_valid_file_from_manifest(summary: dict[str, object]) -> Path:
    manifest = resolve(summary["metrics"]["manifest"])
    for row in read_csv_rows(manifest):
        if row.get("split") == "valid":
            path = resolve(row["file_path"])
            if path.exists():
                return path
    raise ValueError(f"No valid split file found in {manifest}")


def conditioned_prefix_ids(summary: dict[str, object], prefix_token_count: int) -> tuple[list[int], Path]:
    tokenizer = tokenizer_from_summary(summary)
    valid_file = first_valid_file_from_manifest(summary)
    block_size = int(summary["transformer"]["block_size"])
    ids = tokenizer.encode_file(valid_file)
    windows = make_lm_windows([ids], block_size=block_size, stride=block_size)
    if not windows:
        raise ValueError(f"No validation windows could be made from {valid_file}")
    prefix_ids = windows[0][: max(1, min(prefix_token_count, len(windows[0])))]
    return prefix_ids, valid_file


def default_conditioned_prefix_count(summary: dict[str, object]) -> int:
    outputs = summary.get("outputs", {})
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if str(key).startswith("transformer_conditioned") and isinstance(value, dict):
                raw_count = value.get("prefix_token_count")
                if raw_count:
                    return int(raw_count)
    return max(1, int(summary["transformer"]["block_size"]) // 2)


def continuation_metrics_for_candidate(
    candidate: Path,
    prefix_metrics: dict[str, object],
    prefix_token_count: int,
    generated_token_count: int,
) -> dict[str, object]:
    full = asdict(analyze_candidate(candidate))
    full_notes, ticks_per_beat = read_note_events_with_velocity(candidate)
    split_tick = round(float(prefix_metrics["duration_seconds"]) / 0.5 * ticks_per_beat)
    continuation_notes = crop_after(full_notes, split_tick)
    _dataset, task_type, model_type = infer_candidate_labels(candidate)
    temperature, top_k, candidate_index = infer_sampling_settings(candidate)
    continuation = metrics_from_notes(
        candidate,
        continuation_notes,
        ticks_per_beat,
        task_type,
        model_type,
        temperature,
        top_k,
        candidate_index,
    )
    return {
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


def update_aggregate_tables(evaluation_dir: Path, run_dirs: list[Path]) -> None:
    tables_dir = evaluation_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ranking_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        for row in read_csv_rows(run_dir / "candidate_ranking.csv"):
            row["run"] = run_dir.name
            ranking_rows.append(row)
        selected_rows.extend(read_csv_rows(run_dir / "selected_candidates.csv"))
    if ranking_rows:
        write_csv(tables_dir / "candidate_ranking.csv", ranking_rows)
    if selected_rows:
        write_csv(tables_dir / "selected_candidates.csv", selected_rows)


def process_run(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "generation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = resolve(config["metrics_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generation_records = load_generation_records(summary_path)

    prefix_token_count = default_conditioned_prefix_count(summary)
    prefix_ids, prefix_file = conditioned_prefix_ids(summary, prefix_token_count)
    tokenizer = tokenizer_from_summary(summary)
    prefix_path = run_dir / "conditioned_prefix_only.mid"
    tokenizer.decode_ids(prefix_ids, prefix_path)
    prefix_metrics = asdict(analyze_candidate(prefix_path))

    source_dir = resolve(config["source_dir"])
    ranking_rows: list[dict[str, object]] = []
    for existing in read_csv_rows(run_dir / "candidate_ranking.csv"):
        row = dict(existing)
        candidate_path = resolve(row.get("source_path") or row.get("path") or "")
        if row.get("task_type") == "conditioned" and candidate_path.exists():
            record = generation_records.get(str(candidate_path.relative_to(ROOT)).replace("\\", "/"))
            row_prefix_count = int(record.get("prefix_token_count") if record else prefix_token_count)
            row_generated_count = int(record.get("continuation_token_count") if record else 0)
            row.update(
                continuation_metrics_for_candidate(
                    candidate_path,
                    prefix_metrics,
                    row_prefix_count,
                    row_generated_count,
                )
            )
        else:
            for field in CONDITIONED_FIELDS:
                row.setdefault(field, "")
        ranking_rows.append(row)

    if not ranking_rows:
        for candidate_path in sorted(source_dir.glob("transformer_*.mid*")):
            row = asdict(analyze_candidate(candidate_path))
            row["dataset"] = config["dataset"]
            row["source_path"] = str(candidate_path.relative_to(ROOT))
            if row["task_type"] == "conditioned":
                record = generation_records.get(str(candidate_path.relative_to(ROOT)).replace("\\", "/"))
                row_prefix_count = int(record.get("prefix_token_count") if record else prefix_token_count)
                row_generated_count = int(record.get("continuation_token_count") if record else 0)
                row.update(
                    continuation_metrics_for_candidate(
                        candidate_path,
                        prefix_metrics,
                        row_prefix_count,
                        row_generated_count,
                    )
                )
            else:
                for field in CONDITIONED_FIELDS:
                    row[field] = ""
            ranking_rows.append(row)

    write_csv(run_dir / "candidate_ranking.csv", ranking_rows)

    selected_rows: list[dict[str, object]] = []
    conditioned_selected: dict[str, object] | None = None
    for row in read_csv_rows(run_dir / "selected_candidates.csv"):
        selected = dict(row)
        if selected.get("task_type") == "conditioned" and selected.get("source_path"):
            source = resolve(selected["source_path"])
            matched = next(
                (
                    candidate
                    for candidate in ranking_rows
                    if resolve(candidate.get("source_path") or candidate.get("path") or "") == source
                ),
                None,
            )
            if matched:
                for field in CONDITIONED_FIELDS:
                    selected[field] = matched.get(field, "")
            full_path = run_dir / "conditioned_full_with_prefix.mid"
            continuation_path = run_dir / "conditioned_continuation_only.mid"
            safe_copy(source, full_path)
            full_notes, ticks_per_beat = read_note_events_with_velocity(full_path)
            split_tick = round(float(prefix_metrics["duration_seconds"]) / 0.5 * ticks_per_beat)
            continuation_notes = crop_after(full_notes, split_tick)
            write_note_events(continuation_path, continuation_notes, ticks_per_beat)
            selected["conditioned_prefix_only_path"] = str(prefix_path.relative_to(ROOT))
            selected["conditioned_full_with_prefix_path"] = str(full_path.relative_to(ROOT))
            selected["conditioned_continuation_only_path"] = str(continuation_path.relative_to(ROOT))
            conditioned_selected = selected
        selected_rows.append(selected)
    write_csv(run_dir / "selected_candidates.csv", selected_rows)

    config["symbolic_conditioned_contains_prefix"] = True
    config["conditioned_prefix_split_outputs"] = {
        "prefix_source_file": str(prefix_file.relative_to(ROOT)) if prefix_file.is_relative_to(ROOT) else str(prefix_file),
        "prefix_token_count": prefix_token_count,
        "prefix_only": str(prefix_path.relative_to(ROOT)),
        "full_with_prefix": str((run_dir / "conditioned_full_with_prefix.mid").relative_to(ROOT)),
        "continuation_only": str((run_dir / "conditioned_continuation_only.mid").relative_to(ROOT)),
        "selection_rule_conditioned": "existing selected conditioned candidate; continuation metrics are diagnostic",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    selected = conditioned_selected or {}
    notes_path = run_dir / "notes.txt"
    existing_notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    conditioned_notes = [
        "",
        "Conditioned prefix-aware diagnostic:",
        "symbolic_conditioned.mid and conditioned_full_with_prefix.mid include the real validation prefix.",
        "conditioned_continuation_only.mid is cropped after the decoded prefix end time and is the key file for judging model continuation.",
        f"conditioned_prefix_file={config['conditioned_prefix_split_outputs']['prefix_source_file']}",
        f"conditioned_prefix_token_count={prefix_token_count}",
        (
            "conditioned_continuation_note_count="
            f"{selected.get('conditioned_continuation_note_count', '')}, "
            f"conditioned_continuation_duration={selected.get('conditioned_continuation_duration', '')}, "
            f"conditioned_continuation_notes_per_second={selected.get('conditioned_continuation_notes_per_second', '')}, "
            f"conditioned_continuation_usable={selected.get('conditioned_continuation_usable', '')}, "
            f"conditioned_continuation_reject_reason={selected.get('conditioned_continuation_reject_reason', '')}"
        ),
    ]
    notes_path.write_text(existing_notes.rstrip() + "\n" + "\n".join(conditioned_notes).rstrip() + "\n", encoding="utf-8")

    return {
        "run": run_dir.name,
        "prefix_token_count": prefix_token_count,
        "prefix_file": config["conditioned_prefix_split_outputs"]["prefix_source_file"],
        "conditioned_prefix_only": str(prefix_path.relative_to(ROOT)),
        "conditioned_full_with_prefix": str((run_dir / "conditioned_full_with_prefix.mid").relative_to(ROOT)),
        "conditioned_continuation_only": str((run_dir / "conditioned_continuation_only.mid").relative_to(ROOT)),
        "selected_conditioned": conditioned_selected,
    }


def main() -> None:
    parser = ArgumentParser(description="Split conditioned indexed runs into prefix/full/continuation files.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--evaluation-dir", default=None)
    args = parser.parse_args()

    run_dirs = [resolve(run_dir) for run_dir in args.run_dirs]
    summaries = [process_run(run_dir) for run_dir in run_dirs]
    if args.evaluation_dir:
        update_aggregate_tables(resolve(args.evaluation_dir), run_dirs)
    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
