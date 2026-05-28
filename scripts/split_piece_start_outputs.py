"""Split piece-start indexed runs into primer/full/continuation MIDI files."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import mido


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    CandidateMetrics,
    analyze_candidate,
    candidate_reject_reason,
    infer_candidate_labels,
    infer_sampling_settings,
    max_polyphony,
    repeated_pitch_bigram_rate,
    score_candidate,
)
from src.tokenizers import REMITokenizerSmoke  # noqa: E402


PREFIX_FIELDS = [
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


def resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def load_generation_records(summary_path: Path) -> dict[str, dict[str, object]]:
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = {}
    for record in summary.get("outputs", {}).values():
        candidate_path = record.get("path")
        if candidate_path:
            records[str(candidate_path).replace("\\", "/")] = record
    return records


def manifest_files(summary: dict[str, object]) -> list[Path]:
    manifest = resolve(summary["metrics"]["manifest"])
    files = []
    for row in read_csv_rows(manifest):
        path = resolve(row["file_path"])
        if path.exists():
            files.append(path)
    return files


def tokenizer_from_summary(summary: dict[str, object]) -> REMITokenizerSmoke:
    detail = str(summary.get("tokenizer_detail", "remi_num_velocities_8"))
    try:
        num_velocities = int(detail.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        num_velocities = 8
    tokenizer = REMITokenizerSmoke(num_velocities=num_velocities)
    tokenizer.fit(manifest_files(summary))
    return tokenizer


def read_note_events_with_velocity(path: Path) -> tuple[list[dict[str, int]], int]:
    midi = mido.MidiFile(path)
    notes: list[dict[str, int]] = []
    for track in midi.tracks:
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
                channel = int(getattr(message, "channel", 0))
                key = (channel, int(message.note))
                active.setdefault(key, []).append((absolute, int(message.velocity)))
            elif message.type in {"note_off", "note_on"}:
                channel = int(getattr(message, "channel", 0))
                key = (channel, int(message.note))
                starts = active.get(key, [])
                if not starts:
                    continue
                start, velocity = starts.pop(0)
                notes.append(
                    {
                        "start": start,
                        "end": max(start + 1, absolute),
                        "pitch": int(message.note),
                        "velocity": max(1, min(velocity, 127)),
                        "channel": channel,
                    }
                )
    notes.sort(key=lambda note: (note["start"], note["pitch"], note["end"]))
    return notes, midi.ticks_per_beat or 480


def write_note_events(path: Path, notes: list[dict[str, int]], ticks_per_beat: int) -> None:
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("program_change", program=0, time=0))
    events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        start = max(0, int(note["start"]))
        end = max(start + 1, int(note["end"]))
        channel = max(0, min(int(note.get("channel", 0)), 15))
        pitch = max(0, min(int(note["pitch"]), 127))
        velocity = max(1, min(int(note.get("velocity", 64)), 127))
        events.append((start, 1, mido.Message("note_on", note=pitch, velocity=velocity, channel=channel, time=0)))
        events.append((end, 0, mido.Message("note_off", note=pitch, velocity=0, channel=channel, time=0)))
    events.sort(key=lambda item: (item[0], item[1]))
    previous = 0
    for absolute, _order, message in events:
        message.time = max(0, absolute - previous)
        track.append(message)
        previous = absolute
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def safe_copy(source: Path, target: Path) -> bool:
    try:
        shutil.copy2(source, target)
        return True
    except PermissionError:
        return False


def duration_seconds(notes: list[dict[str, int]], ticks_per_beat: int) -> float:
    if not notes:
        return 0.0
    return max(note["end"] for note in notes) / max(ticks_per_beat, 1) * 0.5


def crop_after(notes: list[dict[str, int]], split_tick: int) -> list[dict[str, int]]:
    cropped = []
    for note in notes:
        if note["end"] <= split_tick:
            continue
        shifted = dict(note)
        shifted["start"] = max(note["start"], split_tick) - split_tick
        shifted["end"] = note["end"] - split_tick
        if shifted["end"] > shifted["start"]:
            cropped.append(shifted)
    return cropped


def metrics_from_notes(
    path: Path,
    notes: list[dict[str, int]],
    ticks_per_beat: int,
    task_type: str,
    model_type: str,
    temperature: float | None,
    top_k: int | None,
    candidate_index: int | None,
) -> CandidateMetrics:
    if notes:
        starts = [note["start"] for note in notes]
        ends = [note["end"] for note in notes]
        pitches = [note["pitch"] for note in notes]
        dur = max(ends) / max(ticks_per_beat, 1) * 0.5
        pitch_min = min(pitches)
        pitch_max = max(pitches)
        pitch_range = pitch_max - pitch_min
        unique_pitch_count = len(set(pitches))
        notes_per_second = len(notes) / max(dur, 1e-6)
        note_tuples = [(note["start"], note["end"], note["pitch"]) for note in notes]
        simultaneous = max_polyphony(note_tuples)
        repetition = repeated_pitch_bigram_rate(note_tuples)
        valid = True
    else:
        dur = 0.0
        pitch_min = None
        pitch_max = None
        pitch_range = 0
        unique_pitch_count = 0
        notes_per_second = 0.0
        simultaneous = 0
        repetition = 0.0
        valid = False
    metrics = CandidateMetrics(
        path=str(path),
        dataset=infer_candidate_labels(path)[0],
        task_type=task_type,
        model_type=model_type,
        temperature=temperature,
        top_k=top_k,
        candidate_index=candidate_index,
        valid=valid,
        note_count=len(notes),
        duration_seconds=dur,
        notes_per_second=notes_per_second,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        pitch_range=pitch_range,
        unique_pitch_count=unique_pitch_count,
        max_simultaneous_notes=simultaneous,
        repeated_pitch_bigram_rate=repetition,
        usable=False,
        reject_reason="",
        score=0.0,
    )
    metrics.reject_reason = candidate_reject_reason(metrics)
    metrics.usable = not metrics.reject_reason
    metrics.score = score_candidate(metrics)
    return metrics


def prefix_metrics_for_candidate(
    candidate: Path,
    primer_duration_ticks: int,
    primer_metrics: CandidateMetrics,
    primer_token_count: int,
    generated_token_count: int,
) -> dict[str, object]:
    full = asdict(analyze_candidate(candidate))
    notes, ticks_per_beat = read_note_events_with_velocity(candidate)
    split_tick = round(primer_metrics.duration_seconds / 0.5 * ticks_per_beat)
    if primer_metrics.duration_seconds <= 0:
        split_tick = primer_duration_ticks
    continuation_notes = crop_after(notes, split_tick)
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
        "primer_token_count": primer_token_count,
        "generated_token_count": generated_token_count,
        "full_token_count": primer_token_count + generated_token_count,
        "primer_duration": primer_metrics.duration_seconds,
        "full_duration": full["duration_seconds"],
        "continuation_duration": continuation.duration_seconds,
        "primer_note_count": primer_metrics.note_count,
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


def best_unconditioned_by_continuation(rows: list[dict[str, object]]) -> dict[str, object] | None:
    usable = [
        row
        for row in rows
        if row["task_type"] == "unconditioned"
        and str(row.get("continuation_usable")).lower() == "true"
    ]
    usable.sort(key=lambda row: float(row.get("continuation_score", -1e9)), reverse=True)
    return usable[0] if usable else None


def best_full_by_task(rows: list[dict[str, object]], task_type: str) -> dict[str, object] | None:
    usable = [
        row for row in rows if row["task_type"] == task_type and row["valid"] and row.get("usable", False)
    ]
    usable.sort(key=lambda row: float(row["score"]), reverse=True)
    return usable[0] if usable else None


def process_run(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "generation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    generation = config.get("generation", {})
    primer = generation.get("unconditioned_primer")
    if generation.get("unconditioned_mode") != "piece_start_seeded" or not primer:
        raise ValueError(f"{run_dir} is not a piece_start_seeded run")

    summary_path = resolve(config["metrics_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_summary(summary)
    primer_file = resolve(primer["primer_file"])
    primer_token_count = int(primer["actual_prefix_tokens"])
    generated_token_count = int(generation.get("unconditioned_continuation_tokens") or 0)
    primer_ids = tokenizer.encode_file(primer_file)[:primer_token_count]
    primer_path = run_dir / "primer_only.mid"
    tokenizer.decode_ids(primer_ids, primer_path)
    primer_metrics = analyze_candidate(primer_path)
    primer_notes, primer_ticks_per_beat = read_note_events_with_velocity(primer_path)
    primer_duration_ticks = max((note["end"] for note in primer_notes), default=0)

    source_dir = resolve(config["source_dir"])
    candidates = sorted(source_dir.glob("transformer_*.mid*"))
    generation_records = load_generation_records(summary_path)
    ranking_rows: list[dict[str, object]] = []
    for candidate in candidates:
        row = asdict(analyze_candidate(candidate))
        row["dataset"] = config["dataset"]
        row["source_path"] = str(candidate.relative_to(ROOT))
        record = generation_records.get(str(candidate.relative_to(ROOT)).replace("\\", "/"))
        if record:
            generated_token_count_for_row = int(record.get("continuation_token_count") or generated_token_count)
            primer_token_count_for_row = int(record.get("prefix_token_count") or primer_token_count)
        else:
            generated_token_count_for_row = generated_token_count
            primer_token_count_for_row = primer_token_count
        if row["task_type"] == "unconditioned":
            row.update(
                prefix_metrics_for_candidate(
                    candidate,
                    primer_duration_ticks,
                    primer_metrics,
                    primer_token_count_for_row,
                    generated_token_count_for_row,
                )
            )
        else:
            for field in PREFIX_FIELDS:
                row[field] = ""
        ranking_rows.append(row)
    ranking_rows.sort(
        key=lambda row: (
            row["task_type"],
            -float(row.get("continuation_score") or row["score"] or -1e9),
        )
    )
    write_csv(run_dir / "candidate_ranking.csv", ranking_rows)

    selected_rows: list[dict[str, object]] = []
    unconditioned = best_unconditioned_by_continuation(ranking_rows)
    conditioned = best_full_by_task(ranking_rows, "conditioned")
    for task_type, selected in (("unconditioned", unconditioned), ("conditioned", conditioned)):
        if selected is None:
            selected_rows.append(
                {
                    "run": run_dir.name,
                    "dataset": config["dataset"],
                    "task_type": task_type,
                    "selected_path": "",
                    "source_path": "",
                    "reject_reason": "No usable candidate found after prefix-aware scoring.",
                }
            )
            continue
        source = resolve(str(selected["source_path"]))
        target = run_dir / ("symbolic_unconditioned.mid" if task_type == "unconditioned" else "symbolic_conditioned.mid")
        selected_path = target
        if task_type == "unconditioned":
            full_with_primer = run_dir / "full_with_primer.mid"
            safe_copy(source, full_with_primer)
            if not safe_copy(source, target):
                selected_path = full_with_primer
        elif not safe_copy(source, target):
            selected_path = target
        selected_row = {
            "run": run_dir.name,
            "dataset": config["dataset"],
            "task_type": task_type,
            "selected_path": str(selected_path.relative_to(ROOT)),
            "source_path": str(source.relative_to(ROOT)),
            "score": selected["score"],
            "note_count": selected["note_count"],
            "duration_seconds": selected["duration_seconds"],
            "notes_per_second": selected["notes_per_second"],
            "pitch_range": selected["pitch_range"],
            "max_simultaneous_notes": selected["max_simultaneous_notes"],
            "repeated_pitch_bigram_rate": selected["repeated_pitch_bigram_rate"],
            "reject_reason": "",
        }
        for field in PREFIX_FIELDS:
            selected_row[field] = selected.get(field, "")
        if task_type == "unconditioned":
            continuation_only = run_dir / "continuation_only.mid"
            full_notes, ticks_per_beat = read_note_events_with_velocity(full_with_primer)
            split_tick = round(primer_duration_ticks * ticks_per_beat / max(primer_ticks_per_beat, 1))
            continuation_notes = crop_after(full_notes, split_tick)
            write_note_events(continuation_only, continuation_notes, ticks_per_beat)
            selected_row["primer_only_path"] = str(primer_path.relative_to(ROOT))
            selected_row["full_with_primer_path"] = str(full_with_primer.relative_to(ROOT))
            selected_row["continuation_only_path"] = str(continuation_only.relative_to(ROOT))
        selected_rows.append(selected_row)
    write_csv(run_dir / "selected_candidates.csv", selected_rows)

    config["symbolic_unconditioned_contains_primer"] = True
    config["prefix_split_outputs"] = {
        "primer_only": str(primer_path.relative_to(ROOT)),
        "full_with_primer": str((run_dir / "full_with_primer.mid").relative_to(ROOT)),
        "continuation_only": str((run_dir / "continuation_only.mid").relative_to(ROOT)),
        "selection_rule_unconditioned": "highest continuation_score among usable continuation-only candidates",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    uncond_row = next((row for row in selected_rows if row["task_type"] == "unconditioned"), {})
    notes = [
        f"{run_dir.name}: piece_start_seeded prefix-aware reprocess.",
        "symbolic_unconditioned.mid and full_with_primer.mid include the real primer prefix.",
        "continuation_only.mid is cropped after the decoded primer end time and is the key file for judging model output.",
        f"primer: {primer['primer_file']}",
        f"primer_token_count={primer_token_count}, generated_token_count={generated_token_count}",
        f"primer_note_count={uncond_row.get('primer_note_count', '')}, primer_duration={uncond_row.get('primer_duration', '')}",
        f"full_note_count={uncond_row.get('full_note_count', '')}, full_duration={uncond_row.get('full_duration', '')}",
        (
            "continuation_note_count="
            f"{uncond_row.get('continuation_note_count', '')}, "
            f"continuation_duration={uncond_row.get('continuation_duration', '')}, "
            f"continuation_notes_per_second={uncond_row.get('continuation_notes_per_second', '')}, "
            f"continuation_usable={uncond_row.get('continuation_usable', '')}, "
            f"continuation_reject_reason={uncond_row.get('continuation_reject_reason', '')}"
        ),
    ]
    (run_dir / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return {
        "run": run_dir.name,
        "primer_only": str(primer_path.relative_to(ROOT)),
        "full_with_primer": str((run_dir / "full_with_primer.mid").relative_to(ROOT)),
        "continuation_only": str((run_dir / "continuation_only.mid").relative_to(ROOT)),
        "selected": selected_rows,
    }


def main() -> None:
    parser = ArgumentParser(description="Split piece_start_seeded indexed runs into primer/full/continuation files.")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    summaries = [process_run(resolve(run_dir)) for run_dir in args.run_dirs]
    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
