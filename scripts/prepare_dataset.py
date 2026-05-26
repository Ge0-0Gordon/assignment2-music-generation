"""Prepare tiny MIDI-only dataset subsets for bounded experiments."""

from __future__ import annotations

import json
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
NOTTINGHAM_API = "https://api.github.com/repos/jukedeck/nottingham-dataset/contents/MIDI"
MAESTRO_MIDI_ZIP = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with urlopen(url, timeout=30) as response:
        output_path.write_bytes(response.read())


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
    for entry in midi_entries:
        path = output_dir / entry["name"]
        download_url(entry["download_url"], path)
        downloaded.append({"name": entry["name"], "path": str(path), "bytes": path.stat().st_size})
    return {
        "dataset": "nottingham",
        "source": "https://github.com/jukedeck/nottingham-dataset/tree/master/MIDI",
        "license": "GPL-3.0",
        "output_dir": str(output_dir),
        "file_count": len(downloaded),
        "total_bytes": sum(item["bytes"] for item in downloaded),
        "files": downloaded,
    }


def prepare_maestro_midi(output_dir: Path, max_files: int) -> dict[str, object]:
    archive_path = output_dir / "maestro-v3.0.0-midi.zip"
    files_dir = output_dir / "midi"
    download_url(MAESTRO_MIDI_ZIP, archive_path)
    files_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        midi_members = sorted(
            member
            for member in archive.namelist()
            if member.lower().endswith((".mid", ".midi")) and not member.endswith("/")
        )[:max_files]
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
        "license": "CC BY-NC-SA 4.0",
        "output_dir": str(output_dir),
        "archive_bytes": archive_path.stat().st_size,
        "file_count": len(extracted),
        "total_bytes": sum(item["bytes"] for item in extracted),
        "selection": f"first {max_files} MIDI members in sorted archive order",
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
