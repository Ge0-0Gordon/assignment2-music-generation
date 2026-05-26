"""Minimal MIDI tokenizers for Phase 1 smoke testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido
from mido import Message, MidiFile, MidiTrack


def _write_symusic_score(score: object, output_path: Path) -> None:
    if hasattr(score, "dump_midi"):
        score.dump_midi(str(output_path))
        return
    if hasattr(score, "write"):
        score.write(str(output_path))
        return
    raise TypeError("Decoded symusic score does not expose a known MIDI writer")


class REMITokenizerSmoke:
    """Small MidiTok REMI wrapper used only for smoke testing."""

    name = "remi"

    def __init__(self, num_velocities: int = 4):
        from miditok import REMI, TokenizerConfig

        config = TokenizerConfig(
            num_velocities=num_velocities,
            use_chords=False,
            use_programs=False,
        )
        self.tokenizer = REMI(config)

    def fit(self, midi_files: list[Path]) -> None:
        vocab_size = max(512, len(self.tokenizer) + 64)
        self.tokenizer.train(vocab_size=vocab_size, files_paths=[str(path) for path in midi_files])

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode_file(self, midi_path: Path) -> list[int]:
        from symusic import Score

        encoded = self.tokenizer(Score(str(midi_path)))
        sequence = encoded[0] if isinstance(encoded, list) else encoded
        ids = getattr(sequence, "ids", sequence)
        return [int(token_id) for token_id in ids]

    def decode_ids(self, ids: list[int], output_path: Path) -> None:
        score = self.tokenizer.decode([ids])
        _write_symusic_score(score, output_path)


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start: int
    duration: int
    velocity: int


class SimpleMIDITokenizer:
    """Simple note-event tokenizer for smoke-test fallback."""

    name = "simple_event"

    def __init__(self, ticks_per_step: int = 120):
        self.ticks_per_step = ticks_per_step
        self.token_to_id = {"BOS": 0, "EOS": 1}
        self.id_to_token = {0: "BOS", 1: "EOS"}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def fit(self, midi_files: list[Path]) -> None:
        for path in midi_files:
            for token in self._tokens_for_file(path):
                self._add_token(token)

    def encode_file(self, midi_path: Path) -> list[int]:
        tokens = ["BOS", *self._tokens_for_file(midi_path), "EOS"]
        return [self._add_token(token) for token in tokens]

    def decode_ids(self, ids: list[int], output_path: Path) -> None:
        notes = self._ids_to_notes(ids)
        midi = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        midi.tracks.append(track)
        track.append(Message("program_change", program=0, time=0))
        events: list[tuple[int, Message]] = []
        for note in notes:
            velocity = max(1, min(note.velocity, 127))
            start = max(0, note.start)
            end = max(start + 1, note.start + note.duration)
            events.append((start, Message("note_on", note=note.pitch, velocity=velocity, time=0)))
            events.append((end, Message("note_off", note=note.pitch, velocity=0, time=0)))
        events.sort(key=lambda item: (item[0], 0 if item[1].type == "note_off" else 1))
        previous_tick = 0
        for absolute_tick, message in events:
            message.time = max(0, absolute_tick - previous_tick)
            track.append(message)
            previous_tick = absolute_tick
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.save(output_path)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def _tokens_for_file(self, midi_path: Path) -> list[str]:
        notes = self._read_notes(midi_path)
        tokens: list[str] = []
        previous_start_step = 0
        for note in notes:
            start_step = round(note.start / self.ticks_per_step)
            duration_step = max(1, round(note.duration / self.ticks_per_step))
            delta_step = max(0, start_step - previous_start_step)
            velocity_bin = min(3, max(0, note.velocity // 32))
            tokens.append(f"N_{note.pitch}_{delta_step}_{duration_step}_{velocity_bin}")
            previous_start_step = start_step
        return tokens

    def _read_notes(self, midi_path: Path) -> list[NoteEvent]:
        midi = mido.MidiFile(midi_path)
        active: dict[tuple[int, int], tuple[int, int]] = {}
        notes: list[NoteEvent] = []
        for track_index, track in enumerate(midi.tracks):
            absolute = 0
            for message in track:
                absolute += message.time
                if message.type == "note_on" and message.velocity > 0:
                    active[(track_index, message.note)] = (absolute, message.velocity)
                elif message.type in {"note_off", "note_on"}:
                    key = (track_index, message.note)
                    if key not in active:
                        continue
                    start, velocity = active.pop(key)
                    notes.append(
                        NoteEvent(
                            pitch=max(0, min(int(message.note), 127)),
                            start=start,
                            duration=max(1, absolute - start),
                            velocity=velocity,
                        )
                    )
        return sorted(notes, key=lambda note: (note.start, note.pitch, note.duration))

    def _ids_to_notes(self, ids: list[int]) -> list[NoteEvent]:
        notes: list[NoteEvent] = []
        current_step = 0
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), "")
            if not token.startswith("N_"):
                continue
            _, pitch, delta, duration, velocity_bin = token.split("_")
            current_step += int(delta)
            notes.append(
                NoteEvent(
                    pitch=max(0, min(int(pitch), 127)),
                    start=current_step * self.ticks_per_step,
                    duration=max(1, int(duration)) * self.ticks_per_step,
                    velocity=min(127, 32 + int(velocity_bin) * 32),
                )
            )
        return notes
