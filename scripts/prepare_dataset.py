"""Prepare tiny MIDI-only dataset subsets for bounded experiments."""

from __future__ import annotations

import json
import zipfile
from argparse import ArgumentParser
from csv import DictReader
from io import TextIOWrapper
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
NOTTINGHAM_API = "https://api.github.com/repos/jukedeck/nottingham-dataset/contents/MIDI"
MAESTRO_MIDI_ZIP = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
LOCAL_MAESTRO_ZIP_CANDIDATES = [
    ROOT / "data" / "maestro-v3.0.0-midi.zip",
    ROOT / "data" / "raw" / "maestro_subset" / "maestro-v3.0.0-midi.zip",
]


def download_url(url: str, output_path: Path) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, ""
    errors = []
    for _attempt in range(2):
        try:
            with urlopen(url, timeout=45) as response:
                output_path.write_bytes(response.read())
            return True, ""
        except Exception as exc:  # noqa: BLE001 - keep dataset preparation moving.
            errors.append(f"{type(exc).__name__}: {exc}")
    return False, " | ".join(errors)


def prepare_nottingham(output_dir: Path, max_files: int) -> dict[str, object]:
    with urlopen(NOTTINGHAM_API, timeout=30) as response:
        entries = json.loads(response.read().decode("utf-8"))
    midi_entries = [
        entry
        for entry in entries
        if entry.get("type") == "file" and entry.get("name", "").lower().endswith(".mid")
    ]
    midi_entries = sorted(midi_entries, key=lambda item: item["name"])[:max_files]
    downloaded = []
    failed = []
    for entry in midi_entries:
        path = output_dir / entry["name"]
        ok, error = download_url(entry["download_url"], path)
        if ok:
            downloaded.append({"name": entry["name"], "path": str(path), "bytes": path.stat().st_size})
        else:
            failed.append({"name": entry["name"], "url": entry["download_url"], "error": error})
    return {
        "dataset": "nottingham",
        "source": "https://github.com/jukedeck/nottingham-dataset/tree/master/MIDI",
        "license": "GPL-3.0",
        "output_dir": str(output_dir),
        "file_count": len(downloaded),
        "total_bytes": sum(item["bytes"] for item in downloaded),
        "failed_count": len(failed),
        "failed": failed,
        "files": downloaded,
    }


def prepare_maestro_midi(output_dir: Path, max_files: int) -> dict[str, object]:
    archive_path = output_dir / "maestro-v3.0.0-midi.zip"
    files_dir = output_dir / "midi"
    archive_source = "local"
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        local_archive = next((path for path in LOCAL_MAESTRO_ZIP_CANDIDATES if path.exists() and path.stat().st_size > 0), None)
        if local_archive is None:
            ok, error = download_url(MAESTRO_MIDI_ZIP, archive_path)
            archive_source = MAESTRO_MIDI_ZIP
            if not ok:
                raise RuntimeError(error)
        else:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(local_archive.read_bytes())
            archive_source = str(local_archive.relative_to(ROOT))
    files_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        midi_members = [
            member
            for member in archive.namelist()
            if member.lower().endswith((".mid", ".midi")) and not member.endswith("/")
        ]
        midi_by_suffix = {member.removeprefix("maestro-v3.0.0/"): member for member in midi_members}
        metadata_member = next(
            (member for member in archive.namelist() if member.lower().endswith("maestro-v3.0.0.csv")),
            None,
        )
        selected_members = []
        if metadata_member is not None:
            with archive.open(metadata_member) as raw_handle:
                rows = list(DictReader(TextIOWrapper(raw_handle, encoding="utf-8")))
            rows = sorted(rows, key=lambda row: (float(row["duration"]), row["midi_filename"]))
            for row in rows:
                if float(row["duration"]) > 600.0:
                    continue
                member = midi_by_suffix.get(row["midi_filename"])
                if member is not None:
                    selected_members.append(member)
                if len(selected_members) >= max_files:
                    break
        midi_members = selected_members or sorted(midi_members)[:max_files]
        for index, member in enumerate(midi_members, start=1):
            suffix = Path(member).suffix or ".midi"
            name = f"maestro_{index:03d}_{Path(member).name}"
            target = files_dir / name
            if not target.exists() or target.stat().st_size == 0:
                target.write_bytes(archive.read(member))
            extracted.append(
                {
                    "archive_member": member,
                    "name": target.name,
                    "path": str(target),
                    "bytes": target.stat().st_size,
                }
            )
    return {
        "dataset": "maestro_v3_midi",
        "source": MAESTRO_MIDI_ZIP,
        "archive_source": archive_source,
        "license": "CC BY-NC-SA 4.0",
        "output_dir": str(output_dir),
        "archive_bytes": archive_path.stat().st_size,
        "file_count": len(extracted),
        "total_bytes": sum(item["bytes"] for item in extracted),
        "selection": f"up to {max_files} shortest MIDI members with metadata duration <= 600s",
        "files": extracted,
    }


def parse_args() -> object:
    parser = ArgumentParser(description="Prepare small MIDI-only dataset subsets.")
    parser.add_argument("--dataset", choices=["nottingham", "maestro-midi"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if args.dataset == "nottingham":
        summary = prepare_nottingham(output_dir, args.max_files)
    elif args.dataset == "maestro-midi":
        summary = prepare_maestro_midi(output_dir, args.max_files)
    else:
        raise ValueError(args.dataset)
    summary_path = output_dir / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
