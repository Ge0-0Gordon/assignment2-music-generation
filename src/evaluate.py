"""Evaluation helpers for generated MIDI candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mido


@dataclass
class CandidateMetrics:
    path: str
    dataset: str
    task_type: str
    model_type: str
    temperature: float | None
    top_k: int | None
    candidate_index: int | None
    valid: bool
    note_count: int
    duration_seconds: float
    notes_per_second: float
    pitch_min: int | None
    pitch_max: int | None
    pitch_range: int
    unique_pitch_count: int
    max_simultaneous_notes: int
    repeated_pitch_bigram_rate: float
    score: float


def read_note_events(path: Path) -> tuple[list[tuple[int, int, int]], int]:
    """Return (start_tick, end_tick, pitch) note events and ticks_per_beat."""
    midi = mido.MidiFile(path)
    notes: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        active: dict[int, list[int]] = {}
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
                active.setdefault(int(message.note), []).append(absolute)
            elif message.type in {"note_off", "note_on"}:
                starts = active.get(int(message.note), [])
                if not starts:
                    continue
                start = starts.pop(0)
                notes.append((start, max(start + 1, absolute), int(message.note)))
    return sorted(notes), midi.ticks_per_beat or 480


def pitch_class_histogram(path: Path) -> list[int]:
    notes, _ticks_per_beat = read_note_events(path)
    counts = [0] * 12
    for _start, _end, pitch in notes:
        counts[pitch % 12] += 1
    return counts


def max_polyphony(notes: list[tuple[int, int, int]]) -> int:
    events: list[tuple[int, int]] = []
    for start, end, _pitch in notes:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    max_seen = 0
    for _tick, delta in events:
        current += delta
        max_seen = max(max_seen, current)
    return max_seen


def repeated_pitch_bigram_rate(notes: list[tuple[int, int, int]]) -> float:
    pitches = [pitch for _start, _end, pitch in notes]
    if len(pitches) < 3:
        return 0.0
    bigrams = list(zip(pitches, pitches[1:]))
    counts = Counter(bigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(bigrams), 1)


def infer_candidate_labels(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    dataset = "unknown"
    if "candidates" in parts:
        idx = parts.index("candidates")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
            if dataset == "selected" and idx + 2 < len(parts):
                dataset = parts[idx + 2]
    for suffix in ("_subset", "_smoke"):
        if dataset.endswith(suffix):
            dataset = dataset[: -len(suffix)]
    stem = path.stem
    model_type = "transformer" if "transformer" in stem else "markov"
    task_type = "conditioned" if "conditioned" in stem and "unconditioned" not in stem else "unconditioned"
    return dataset, task_type, model_type


def infer_sampling_settings(path: Path) -> tuple[float | None, int | None, int | None]:
    temperature = None
    top_k = None
    candidate_index = None
    for part in path.stem.split("_"):
        if part.startswith("temp"):
            try:
                temperature = float(part.removeprefix("temp").replace("p", "."))
            except ValueError:
                temperature = None
        elif part.startswith("topk"):
            try:
                top_k = int(part.removeprefix("topk"))
            except ValueError:
                top_k = None
        elif part.startswith("idx"):
            try:
                candidate_index = int(part.removeprefix("idx"))
            except ValueError:
                candidate_index = None
    return temperature, top_k, candidate_index


def score_candidate(metrics: CandidateMetrics) -> float:
    if not metrics.valid:
        return -1e9
    density_penalty = abs(metrics.notes_per_second - 2.0)
    range_bonus = min(metrics.pitch_range, 36) / 36.0
    duration_bonus = min(metrics.duration_seconds, 90.0) / 90.0
    polyphony_penalty = max(0, metrics.max_simultaneous_notes - 8) * 0.25
    repetition_penalty = metrics.repeated_pitch_bigram_rate * 2.0
    note_bonus = min(metrics.note_count, 160) / 160.0
    return note_bonus + range_bonus + duration_bonus - density_penalty * 0.15 - polyphony_penalty - repetition_penalty


def analyze_candidate(path: Path) -> CandidateMetrics:
    dataset, task_type, model_type = infer_candidate_labels(path)
    temperature, top_k, candidate_index = infer_sampling_settings(path)
    try:
        notes, ticks_per_beat = read_note_events(path)
        valid = len(notes) > 0
    except Exception:
        notes = []
        ticks_per_beat = 480
        valid = False
    if notes:
        starts = [start for start, _end, _pitch in notes]
        ends = [end for _start, end, _pitch in notes]
        pitches = [pitch for _start, _end, pitch in notes]
        duration_seconds = max(ends) / ticks_per_beat * 0.5
        pitch_min = min(pitches)
        pitch_max = max(pitches)
        pitch_range = pitch_max - pitch_min
        unique_pitch_count = len(set(pitches))
        notes_per_second = len(notes) / max(duration_seconds, 1e-6)
        simultaneous = max_polyphony(notes)
        repetition = repeated_pitch_bigram_rate(notes)
    else:
        duration_seconds = 0.0
        pitch_min = None
        pitch_max = None
        pitch_range = 0
        unique_pitch_count = 0
        notes_per_second = 0.0
        simultaneous = 0
        repetition = 0.0
    metrics = CandidateMetrics(
        path=str(path),
        dataset=dataset,
        task_type=task_type,
        model_type=model_type,
        temperature=temperature,
        top_k=top_k,
        candidate_index=candidate_index,
        valid=valid,
        note_count=len(notes),
        duration_seconds=duration_seconds,
        notes_per_second=notes_per_second,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        pitch_range=pitch_range,
        unique_pitch_count=unique_pitch_count,
        max_simultaneous_notes=simultaneous,
        repeated_pitch_bigram_rate=repetition,
        score=0.0,
    )
    metrics.score = score_candidate(metrics)
    return metrics
