"""Prepare a local MAESTRO v3 MIDI-only full manifest from the existing zip."""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from argparse import ArgumentParser
from io import TextIOWrapper
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import max_polyphony, read_note_events  # noqa: E402

DEFAULT_ZIP = ROOT / "data" / "maestro-v3.0.0-midi.zip"
METADATA_CSV = "maestro-v3.0.0/maestro-v3.0.0.csv"
METADATA_JSON = "maestro-v3.0.0/maestro-v3.0.0.json"


def resolve(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def quality_reject_reason(midi_path: Path, args: object) -> tuple[str, dict[str, object]]:
    if not any(
        value is not None
        for value in (
            args.min_notes,
            args.max_notes,
            args.min_notes_per_second,
            args.max_notes_per_second,
            args.max_token_length,
            args.max_polyphony,
        )
    ):
        return "", {}
    try:
        notes, ticks_per_beat = read_note_events(midi_path)
    except Exception as exc:  # noqa: BLE001 - manifest records file-level skip reasons.
        return f"quality_parse_error={type(exc).__name__}: {exc}", {}
    note_count = len(notes)
    duration_seconds = 0.0
    notes_per_second = 0.0
    polyphony = 0
    if notes:
        duration_seconds = max(end for _start, end, _pitch in notes) / max(ticks_per_beat, 1) * 0.5
        notes_per_second = note_count / max(duration_seconds, 1e-6)
        polyphony = max_polyphony(notes)
    estimated_token_length = note_count * 4
    reasons = []
    if args.min_notes is not None and note_count < args.min_notes:
        reasons.append(f"note_count_lt_{args.min_notes}")
    if args.max_notes is not None and note_count > args.max_notes:
        reasons.append(f"note_count_gt_{args.max_notes}")
    if args.min_notes_per_second is not None and notes_per_second < args.min_notes_per_second:
        reasons.append(f"notes_per_second_lt_{args.min_notes_per_second}")
    if args.max_notes_per_second is not None and notes_per_second > args.max_notes_per_second:
        reasons.append(f"notes_per_second_gt_{args.max_notes_per_second}")
    if args.max_token_length is not None and estimated_token_length > args.max_token_length:
        reasons.append(f"estimated_token_length_gt_{args.max_token_length}")
    if args.max_polyphony is not None and polyphony > args.max_polyphony:
        reasons.append(f"polyphony_gt_{args.max_polyphony}")
    metrics = {
        "note_count": note_count,
        "duration_seconds": duration_seconds,
        "notes_per_second": notes_per_second,
        "estimated_token_length": estimated_token_length,
        "max_polyphony": polyphony,
    }
    return ";".join(reasons), metrics


def main() -> None:
    parser = ArgumentParser(description="Extract local MAESTRO MIDI-only files and write an official-split manifest.")
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP.relative_to(ROOT)))
    parser.add_argument("--output-dir", default="data/raw/maestro_full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-notes", type=int, default=None)
    parser.add_argument("--max-notes", type=int, default=None)
    parser.add_argument("--min-notes-per-second", type=float, default=None)
    parser.add_argument("--max-notes-per-second", type=float, default=None)
    parser.add_argument("--max-token-length", type=int, default=None)
    parser.add_argument("--max-polyphony", type=int, default=None)
    args = parser.parse_args()

    zip_path = resolve(args.zip_path)
    output_dir = resolve(args.output_dir)
    midi_dir = output_dir / "midi"
    metadata_dir = output_dir / "metadata"
    midi_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    rows_out: list[dict[str, object]] = []
    missing_members: list[str] = []
    skipped_rows: list[dict[str, object]] = []
    extracted_count = 0
    reused_count = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        with archive.open(METADATA_CSV) as raw_handle:
            rows = list(csv.DictReader(TextIOWrapper(raw_handle, encoding="utf-8")))
        (metadata_dir / "maestro-v3.0.0.csv").write_bytes(archive.read(METADATA_CSV))
        if METADATA_JSON in members:
            (metadata_dir / "maestro-v3.0.0.json").write_bytes(archive.read(METADATA_JSON))

        for row in rows:
            member = "maestro-v3.0.0/" + row["midi_filename"]
            if member not in members:
                missing_members.append(row["midi_filename"])
                continue
            target = midi_dir / row["midi_filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.force or not target.exists() or target.stat().st_size == 0:
                target.write_bytes(archive.read(member))
                extracted_count += 1
            else:
                reused_count += 1
            reject_reason, quality_metrics = quality_reject_reason(target, args)
            if reject_reason:
                skipped_rows.append(
                    {
                        "file_path": str(target.relative_to(ROOT)),
                        "original_midi_filename": row["midi_filename"],
                        "split": row["split"],
                        "skipped_reason": reject_reason,
                        **quality_metrics,
                    }
                )
                continue
            rows_out.append(
                {
                    "file_path": str(target.relative_to(ROOT)),
                    "original_midi_filename": row["midi_filename"],
                    "split": row["split"],
                    "year": row["year"],
                    "duration": row["duration"],
                    "canonical_composer": row["canonical_composer"],
                    "canonical_title": row["canonical_title"],
                    "skipped_reason": "",
                    **quality_metrics,
                }
            )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "file_path",
            "original_midi_filename",
            "split",
            "year",
            "duration",
            "canonical_composer",
            "canonical_title",
            "skipped_reason",
            "note_count",
            "duration_seconds",
            "notes_per_second",
            "estimated_token_length",
            "max_polyphony",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    skipped_path = output_dir / "skipped_manifest_rows.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "file_path",
            "original_midi_filename",
            "split",
            "skipped_reason",
            "note_count",
            "duration_seconds",
            "notes_per_second",
            "estimated_token_length",
            "max_polyphony",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped_rows)

    split_counts: dict[str, int] = {}
    for row in rows_out:
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1

    summary = {
        "dataset": "maestro_v3_midi_only_full",
        "zip_path": str(zip_path.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "midi_dir": str(midi_dir.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "metadata_csv": str((metadata_dir / "maestro-v3.0.0.csv").relative_to(ROOT)),
        "metadata_json": str((metadata_dir / "maestro-v3.0.0.json").relative_to(ROOT))
        if (metadata_dir / "maestro-v3.0.0.json").exists()
        else None,
        "metadata_rows": len(rows_out),
        "midi_files": len(rows_out),
        "skipped_file_count": len(skipped_rows),
        "skipped_manifest_rows": str(skipped_path.relative_to(ROOT)),
        "split_counts": split_counts,
        "extracted_count": extracted_count,
        "reused_count": reused_count,
        "missing_member_count": len(missing_members),
        "missing_members": missing_members,
    }
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
