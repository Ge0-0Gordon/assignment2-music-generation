"""Prepare a local MAESTRO v3 MIDI-only full manifest from the existing zip."""

from __future__ import annotations

import csv
import json
import zipfile
from argparse import ArgumentParser
from io import TextIOWrapper
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "data" / "maestro-v3.0.0-midi.zip"
METADATA_CSV = "maestro-v3.0.0/maestro-v3.0.0.csv"
METADATA_JSON = "maestro-v3.0.0/maestro-v3.0.0.json"


def resolve(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def main() -> None:
    parser = ArgumentParser(description="Extract local MAESTRO MIDI-only files and write an official-split manifest.")
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP.relative_to(ROOT)))
    parser.add_argument("--output-dir", default="data/raw/maestro_full")
    parser.add_argument("--force", action="store_true")
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
            rows_out.append(
                {
                    "file_path": str(target.relative_to(ROOT)),
                    "original_midi_filename": row["midi_filename"],
                    "split": row["split"],
                    "year": row["year"],
                    "duration": row["duration"],
                    "canonical_composer": row["canonical_composer"],
                    "canonical_title": row["canonical_title"],
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
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

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
