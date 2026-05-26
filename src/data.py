"""Small data helpers for the Phase 1 symbolic MIDI smoke test."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mido
from midiutil import MIDIFile


MIDI_EXTENSIONS = {".mid", ".midi"}


def discover_midi_files(roots: Iterable[Path]) -> list[Path]:
    """Return MIDI files under the given roots, sorted for determinism."""
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS:
                files.append(path)
    return sorted(files)


def create_synthetic_smoke_midis(output_dir: Path) -> list[Path]:
    """Create tiny MIDI files for smoke testing only, not for final training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces = [
        ("smoke_ascending.mid", [60, 62, 64, 65, 67, 69, 71, 72]),
        ("smoke_arpeggio.mid", [60, 64, 67, 72, 67, 64, 60, 55]),
        ("smoke_minor.mid", [57, 60, 64, 69, 67, 64, 60, 57]),
    ]
    paths: list[Path] = []
    for name, pitches in pieces:
        midi = MIDIFile(1)
        midi.addTempo(track=0, time=0, tempo=100)
        for idx, pitch in enumerate(pitches):
            midi.addNote(
                track=0,
                channel=0,
                pitch=pitch,
                time=idx * 0.5,
                duration=0.45,
                volume=80,
            )
        path = output_dir / name
        with path.open("wb") as handle:
            midi.writeFile(handle)
        paths.append(path)
    return paths


def count_midi_notes(path: Path) -> int:
    """Count note-on events with nonzero velocity in a MIDI file."""
    midi = mido.MidiFile(path)
    count = 0
    for track in midi.tracks:
        for message in track:
            if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
                count += 1
    return count


def make_lm_windows(
    sequences: Iterable[list[int]],
    block_size: int,
    stride: int | None = None,
) -> list[list[int]]:
    """Create short language-model windows containing input and next-token labels."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    step = stride or block_size
    windows: list[list[int]] = []
    target_len = block_size + 1
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        if len(sequence) <= target_len:
            windows.append(sequence)
            continue
        for start in range(0, len(sequence) - target_len + 1, step):
            windows.append(sequence[start : start + target_len])
    return windows


def split_train_valid(windows: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """Make a deterministic tiny train/validation split."""
    if len(windows) < 2:
        return windows, windows
    return windows[:-1], windows[-1:]


def split_files(
    midi_files: list[Path],
    valid_fraction: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    """Deterministically split files into train/validation groups."""
    if not midi_files:
        return [], []
    if len(midi_files) == 1:
        return midi_files, midi_files
    valid_count = max(1, int(round(len(midi_files) * valid_fraction)))
    valid_count = min(valid_count, len(midi_files) - 1)
    return midi_files[:-valid_count], midi_files[-valid_count:]


def pitch_class_histogram(paths: Iterable[Path]) -> list[int]:
    """Count note-on pitch classes across MIDI files."""
    counts = [0] * 12
    for path in paths:
        midi = mido.MidiFile(path)
        for track in midi.tracks:
            for message in track:
                if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
                    counts[int(message.note) % 12] += 1
    return counts
