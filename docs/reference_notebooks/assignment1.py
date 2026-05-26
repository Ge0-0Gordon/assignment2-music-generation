import os
import ast
import copy
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import librosa
import numpy as np
import miditoolkit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import random

def resolve_dataroot(*parts):
    candidates = [
        os.path.join("student_files", *parts),
        os.path.join("student_files_updated", "student_files", *parts),
        os.path.join(".", "student_files", *parts),
        os.path.join(".", "student_files_updated", "student_files", *parts),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

def read_literal(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read())

def write_submission_predictions(predictions, outpath, normalize_audio_paths=False):
    # The autograder reads these files with eval(...), so write Python literals.
    serializable = predictions
    if normalize_audio_paths:
        serializable = {
            (k[2:] if isinstance(k, str) and k.startswith('./') else k): v
            for k, v in predictions.items()
        }
    with open(outpath, "w") as z:
        z.write(repr(serializable) + '\n')
    return serializable


def accuracy1(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)


def accuracy2(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)


TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']


def accuracy3(groundtruth, predictions):
    preds, targets = [], []
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        prediction = [predictions[k][tag] for tag in TAGS]
        target = [1 if tag in groundtruth[k] else 0 for tag in TAGS]
        preds.append(prediction)
        targets.append(target)

    mAP = average_precision_score(targets, preds, average='macro')
    return mAP


dataroot1 = resolve_dataroot("task1_composer_classification")


class model1():
    def __init__(self):
        self.model = None

    def _midi_path(self, path):
        return os.path.join(dataroot1, path)

    def _all_notes(self, midi_obj):
        notes = []
        for inst in getattr(midi_obj, "instruments", []):
            notes.extend(getattr(inst, "notes", []))
        notes.sort(key=lambda n: (n.start, n.pitch, n.end))
        return notes

    def _normalized_hist(self, values, bins):
        hist, _ = np.histogram(values, bins=bins)
        hist = hist.astype(float)
        return [float(x) for x in hist / max(float(np.sum(hist)), 1.0)]

    def features(self, path):
        try:
            midi_obj = miditoolkit.midi.parser.MidiFile(self._midi_path(path))
            notes = self._all_notes(midi_obj)
        except Exception:
            midi_obj = None
            notes = []

        if len(notes) == 0:
            # Fixed-length vector for MIDI files with no notes.
            return [0.0] * 96

        ticks_per_beat = float(getattr(midi_obj, "ticks_per_beat", 480) or 480)
        pitches = np.asarray([note.pitch for note in notes], dtype=float)
        durations = np.asarray([max(0, note.end - note.start) for note in notes], dtype=float)
        duration_beats = durations / ticks_per_beat
        velocities = np.asarray([note.velocity for note in notes], dtype=float)
        starts = np.asarray([note.start for note in notes], dtype=float)
        ends = np.asarray([note.end for note in notes], dtype=float)

        order = np.argsort(starts, kind="mergesort")
        sorted_pitches = pitches[order]
        sorted_starts = starts[order]
        intervals = np.diff(sorted_pitches)
        abs_intervals = np.abs(intervals)
        onset_gaps = np.diff(sorted_starts)
        onset_gap_beats = onset_gaps[onset_gaps >= 0] / ticks_per_beat

        total_ticks = float(max(np.max(ends) - np.min(starts), 1.0))
        total_beats = total_ticks / ticks_per_beat
        note_count = float(len(notes))
        unique_pitch_count = float(len(set(pitches.astype(int))))
        note_density = note_count / max(total_beats, 1e-6)

        if len(intervals) > 0:
            ascending_ratio = float(np.mean(intervals > 0))
            descending_ratio = float(np.mean(intervals < 0))
            repeated_note_ratio = float(np.mean(intervals == 0))
            interval_mean = float(np.mean(intervals) / 24.0)
            interval_std = float(np.std(intervals) / 24.0)
            abs_interval_mean = float(np.mean(abs_intervals) / 24.0)
            abs_interval_std = float(np.std(abs_intervals) / 24.0)
        else:
            ascending_ratio = descending_ratio = repeated_note_ratio = 0.0
            interval_mean = interval_std = abs_interval_mean = abs_interval_std = 0.0

        dur_q25, dur_q50, dur_q75 = np.quantile(duration_beats, [0.25, 0.5, 0.75])
        gap_mean = float(np.mean(onset_gap_beats)) if len(onset_gap_beats) else 0.0
        gap_std = float(np.std(onset_gap_beats)) if len(onset_gap_beats) else 0.0

        pitch_hist = self._normalized_hist(pitches, np.linspace(0, 128, 17))
        pitch_class_hist = np.bincount((pitches.astype(int) % 12), minlength=12).astype(float)
        pitch_class_hist = [float(x) for x in pitch_class_hist / max(float(np.sum(pitch_class_hist)), 1.0)]
        duration_hist = self._normalized_hist(duration_beats, np.array([0, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, np.inf]))
        velocity_hist = self._normalized_hist(velocities, np.linspace(0, 128, 9))
        signed_interval_hist = self._normalized_hist(intervals, np.array([-np.inf, -24, -12, -7, -2, 0, 2, 7, 12, 24, np.inf]))
        abs_interval_hist = self._normalized_hist(abs_intervals, np.array([0, 1, 2, 3, 5, 7, 12, 24, np.inf]))
        beat_positions = ((starts / ticks_per_beat) % 1.0)
        onset_position_hist = self._normalized_hist(beat_positions, np.linspace(0, 1, 9))

        features = [
            float(np.log1p(note_count)),
            float(np.log1p(total_beats)),
            float(np.log1p(note_density)),
            float(np.log1p(len(getattr(midi_obj, "instruments", [])) if midi_obj is not None else 0)),
            float(unique_pitch_count / max(note_count, 1.0)),
            float(np.mean(pitches) / 127.0),
            float(np.std(pitches) / 64.0),
            float(np.min(pitches) / 127.0),
            float(np.max(pitches) / 127.0),
            float((np.max(pitches) - np.min(pitches)) / 127.0),
            float(np.log1p(np.mean(duration_beats))),
            float(np.log1p(np.std(duration_beats))),
            float(np.log1p(dur_q25)),
            float(np.log1p(dur_q50)),
            float(np.log1p(dur_q75)),
            float(np.mean(velocities) / 127.0),
            float(np.std(velocities) / 64.0),
            float(np.log1p(gap_mean)),
            float(np.log1p(gap_std)),
            interval_mean,
            interval_std,
            abs_interval_mean,
            abs_interval_std,
            ascending_ratio,
            descending_ratio,
            repeated_note_ratio,
        ]
        features.extend(pitch_hist)
        features.extend(pitch_class_hist)
        features.extend(duration_hist)
        features.extend(velocity_hist)
        features.extend(signed_interval_hist)
        features.extend(abs_interval_hist)
        features.extend(onset_position_hist)
        return features

    def predict(self, path, outpath=None):
        d = read_literal(path)
        predictions = {}
        paths = d.keys() if isinstance(d, dict) else d
        for k in paths:
            pred = self.model.predict([self.features(k)])
            predictions[k] = int(pred[0])
        if outpath:
            predictions = write_submission_predictions(predictions, outpath)
        return predictions

    # Train your model. Note that this function will not be called from the autograder:
    # instead you should upload your saved model using save()
    def train(self, path):
        train_json = read_literal(path)
        X_train = np.asarray([self.features(k) for k in train_json], dtype=float)
        y_train = np.asarray([int(train_json[k]) for k in train_json])

        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            val_model = ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=fold,
                n_jobs=-1,
            )
            val_model.fit(X_train[tr_idx], y_train[tr_idx])
            val_pred = val_model.predict(X_train[val_idx])
            cv_scores.append(accuracy_score(y_train[val_idx], val_pred))
        print("Task 1 5-fold validation accuracy = "
              + str(float(np.mean(cv_scores))) + " +/- " + str(float(np.std(cv_scores))))

        self.model = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)


dataroot2 = resolve_dataroot("task2_next_sequence_prediction")


class model2():
    def __init__(self):
        self.model = None
        self._feature_cache = {}

    def _midi_path(self, path):
        return os.path.join(dataroot2, path)

    def _all_notes(self, midi_obj):
        notes = []
        for inst in getattr(midi_obj, "instruments", []):
            notes.extend(getattr(inst, "notes", []))
        notes.sort(key=lambda n: (n.start, n.pitch, n.end))
        return notes

    def _safe_stats(self, values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values))]

    def _region_summary(self, notes):
        if len(notes) == 0:
            return {
                "pitch_mean": 0.0,
                "duration_mean": 0.0,
                "velocity_mean": 0.0,
                "interval_mean": 0.0,
                "interval_std": 0.0,
                "chroma": [0.0] * 12,
                "interval_hist": [0.0] * 8,
            }

        pitches = np.asarray([note.pitch for note in notes], dtype=float)
        durations = np.asarray([max(0, note.end - note.start) for note in notes], dtype=float)
        velocities = np.asarray([note.velocity for note in notes], dtype=float)
        intervals = np.diff(pitches)

        chroma = np.bincount((pitches.astype(int) % 12), minlength=12).astype(float)
        chroma = chroma / max(float(np.sum(chroma)), 1.0)

        interval_bins = np.array([-np.inf, -12, -7, -2, 0, 2, 7, 12, np.inf])
        interval_hist, _ = np.histogram(intervals, bins=interval_bins)
        interval_hist = interval_hist.astype(float) / max(float(np.sum(interval_hist)), 1.0)

        return {
            "pitch_mean": float(np.mean(pitches)),
            "duration_mean": float(np.mean(durations)),
            "velocity_mean": float(np.mean(velocities)),
            "interval_mean": float(np.mean(intervals)) if len(intervals) else 0.0,
            "interval_std": float(np.std(intervals)) if len(intervals) else 0.0,
            "chroma": [float(x) for x in chroma],
            "interval_hist": [float(x) for x in interval_hist],
        }

    def features(self, path):
        if path in self._feature_cache:
            return self._feature_cache[path]

        try:
            midi_obj = miditoolkit.midi.parser.MidiFile(self._midi_path(path))
            notes = self._all_notes(midi_obj)
            ticks_per_beat = float(getattr(midi_obj, "ticks_per_beat", 480) or 480)
        except Exception:
            notes = []
            ticks_per_beat = 480.0

        if len(notes) == 0:
            feat = {
                "vector": [0.0] * 42,
                "begin_pitch": 0.0,
                "end_pitch": 0.0,
                "begin_duration": 0.0,
                "end_duration": 0.0,
                "begin_velocity": 0.0,
                "end_velocity": 0.0,
                "first_gap": 0.0,
                "last_gap": 0.0,
                "slope": 0.0,
                "density": 0.0,
                "note_count": 0.0,
                "first_n": self._region_summary([]),
                "last_n": self._region_summary([]),
                "first_half": self._region_summary([]),
                "second_half": self._region_summary([]),
            }
            self._feature_cache[path] = feat
            return feat

        pitches = np.asarray([note.pitch for note in notes], dtype=float)
        durations = np.asarray([max(0, note.end - note.start) for note in notes], dtype=float)
        velocities = np.asarray([note.velocity for note in notes], dtype=float)
        starts = np.asarray([note.start for note in notes], dtype=float)
        ends = np.asarray([note.end for note in notes], dtype=float)
        order = np.argsort(starts, kind="mergesort")
        sorted_notes = [notes[i] for i in order]
        sorted_pitches = pitches[order]
        sorted_starts = starts[order]

        min_tick = float(np.min(starts))
        max_tick = float(np.max(starts))
        begin_mask = starts == min_tick
        end_mask = starts == max_tick
        begin_pitch = float(np.mean(pitches[begin_mask]))
        end_pitch = float(np.mean(pitches[end_mask]))
        begin_duration = float(np.mean(durations[begin_mask]))
        end_duration = float(np.mean(durations[end_mask]))
        begin_velocity = float(np.mean(velocities[begin_mask]))
        end_velocity = float(np.mean(velocities[end_mask]))

        onset_gaps = np.diff(sorted_starts)
        intervals = np.diff(sorted_pitches)
        total_ticks = float(max(np.max(ends) - np.min(starts), 1.0))
        total_beats = total_ticks / ticks_per_beat
        density = len(notes) / total_beats if total_beats > 0 else 0.0
        slope = float((sorted_pitches[-1] - sorted_pitches[0]) / max(len(sorted_pitches) - 1, 1))

        pitch_class_hist = np.bincount((pitches.astype(int) % 12), minlength=12).astype(float)
        pitch_class_hist = pitch_class_hist / max(float(np.sum(pitch_class_hist)), 1.0)

        n_boundary = min(12, len(sorted_notes))
        half = max(1, len(sorted_notes) // 2)
        first_n = self._region_summary(sorted_notes[:n_boundary])
        last_n = self._region_summary(sorted_notes[-n_boundary:])
        first_half = self._region_summary(sorted_notes[:half])
        second_half = self._region_summary(sorted_notes[half:])

        vector = []
        vector.extend([
            float(len(notes)),
            float(len(set(pitches.astype(int)))),
            float(np.max(pitches) - np.min(pitches)),
            float(total_beats),
            float(density),
            begin_pitch,
            end_pitch,
            begin_duration,
            end_duration,
            begin_velocity,
            end_velocity,
            float(onset_gaps[0]) if len(onset_gaps) else 0.0,
            float(onset_gaps[-1]) if len(onset_gaps) else 0.0,
            slope,
        ])
        vector.extend(self._safe_stats(pitches))
        vector.extend(self._safe_stats(durations))
        vector.extend(self._safe_stats(onset_gaps))
        vector.extend(self._safe_stats(intervals))
        vector.extend([float(x) for x in pitch_class_hist])

        feat = {
            "vector": vector,
            "begin_pitch": begin_pitch,
            "end_pitch": end_pitch,
            "begin_duration": begin_duration,
            "end_duration": end_duration,
            "begin_velocity": begin_velocity,
            "end_velocity": end_velocity,
            "first_gap": float(onset_gaps[0]) if len(onset_gaps) else 0.0,
            "last_gap": float(onset_gaps[-1]) if len(onset_gaps) else 0.0,
            "slope": slope,
            "density": float(density),
            "note_count": float(len(notes)),
            "first_n": first_n,
            "last_n": last_n,
            "first_half": first_half,
            "second_half": second_half,
        }
        self._feature_cache[path] = feat
        return feat

    def _boundary_features(self, first, second):
        return [
            abs(first["end_pitch"] - second["begin_pitch"]),
            first["end_pitch"] - second["begin_pitch"],
            abs(first["end_duration"] - second["begin_duration"]),
            abs(first["end_velocity"] - second["begin_velocity"]),
            abs(first["last_gap"] - second["first_gap"]),
            abs(first["density"] - second["density"]),
            abs(first["slope"] - second["slope"]),
        ]

    def _region_distance(self, left, right):
        chroma_dist = sum(abs(a - b) for a, b in zip(left["chroma"], right["chroma"]))
        interval_dist = sum(abs(a - b) for a, b in zip(left["interval_hist"], right["interval_hist"]))
        return [
            abs(left["pitch_mean"] - right["pitch_mean"]),
            abs(left["duration_mean"] - right["duration_mean"]),
            abs(left["velocity_mean"] - right["velocity_mean"]),
            abs(left["interval_mean"] - right["interval_mean"]),
            abs(left["interval_std"] - right["interval_std"]),
            chroma_dist,
            interval_dist,
        ]

    def pair_features(self, path1, path2):
        f1 = self.features(path1)
        f2 = self.features(path2)
        forward = self._boundary_features(f1, f2)
        backward = self._boundary_features(f2, f1)
        forward_n = self._region_distance(f1["last_n"], f2["first_n"])
        backward_n = self._region_distance(f2["last_n"], f1["first_n"])
        forward_half = self._region_distance(f1["second_half"], f2["first_half"])
        backward_half = self._region_distance(f2["second_half"], f1["first_half"])
        global_abs_diff = [abs(a - b) for a, b in zip(f1["vector"], f2["vector"])]
        global_signed_diff = [a - b for a, b in zip(f1["vector"], f2["vector"])]
        continuity_margin = [
            forward[0] - backward[0],
            forward[2] - backward[2],
            forward[4] - backward[4],
            sum(forward[:5]) - sum(backward[:5]),
            sum(forward_n) - sum(backward_n),
            sum(forward_half) - sum(backward_half),
        ]
        region_margins = [a - b for a, b in zip(forward_n + forward_half, backward_n + backward_half)]
        return (
            forward + backward + forward_n + backward_n + forward_half + backward_half +
            continuity_margin + region_margins + global_abs_diff + global_signed_diff
        )

    def train(self, path):
        train_json = read_literal(path)
        X_train = [self.pair_features(k[0], k[1]) for k in train_json]
        y_train = [bool(train_json[k]) for k in train_json]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
        )
        val_model = ExtraTreesClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=3,
            n_jobs=-1,
        )
        val_model.fit(X_tr, y_tr)
        val_pred = val_model.predict(X_val)
        print("Task 2 validation accuracy = " + str(accuracy_score(y_val, val_pred)))

        self.model = ExtraTreesClassifier(
            n_estimators=350,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=2,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

    def predict(self, path, outpath=None):
        d = read_literal(path)
        predictions = {}
        pairs = d.keys() if isinstance(d, dict) else d
        for k in tqdm(pairs):
            path1, path2 = k # Keys are pairs of paths
            x = self.pair_features(path1, path2)
            predictions[k] = bool(self.model.predict([x])[0])
        if outpath:
            predictions = write_submission_predictions(predictions, outpath)
        return predictions


SAMPLE_RATE = 22050
N_MELS = 64
N_CLASSES = 10
AUDIO_DURATION = 10
BATCH_SIZE = 32
TASK3_EPOCHS = 50 
TASK3_LR = 3e-4


dataroot3 = resolve_dataroot("task3_audio_classification")


def extract_waveform(path, crop="first"):
    clean_path = path[2:] if isinstance(path, str) and path.startswith("./") else path
    waveform, sr = librosa.load(os.path.join(dataroot3, clean_path), sr=SAMPLE_RATE, mono=True)
    waveform = torch.FloatTensor(np.array([waveform]))  # Convert to tensor first
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # Pad so that everything is the right length
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        max_start = waveform.shape[1] - target_len
        if crop == "middle":
            start = max_start // 2
        elif crop == "last":
            start = max_start
        elif crop == "random":
            start = random.randint(0, max_start)
        else:
            start = 0
        waveform = waveform[:, start:start + target_len]
    return waveform


class AudioDataset(Dataset):
    def __init__(self, meta, preload = True, crop = "first"):
        self.meta = meta
        ks = list(meta.keys())
        self.idToPath = dict(zip(range(len(ks)), ks))
        self.pathToFeat = {}
        self.crop = crop

        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        self.db = AmplitudeToDB()

        self.preload = preload and crop != "random" # Random crop should be recomputed each epoch.
                                                    # Preloading uses more memory but is faster for fixed crops.
        if self.preload:
            for path in tqdm(ks, desc="Preloading audio"):
                self.pathToFeat[path] = self._extract_mel(path)

    def _extract_mel(self, path):
        waveform = extract_waveform(path, crop=self.crop)
        mel_spec = self.db(self.mel(waveform)).squeeze(0)
        return (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Faster version, preloads the features
        path = self.idToPath[idx]
        tags = self.meta[path]
        bin_label = torch.tensor([1 if tag in tags else 0 for tag in TAGS], dtype=torch.float32)

        if self.preload:
            mel_spec = self.pathToFeat[path]
        else:
            mel_spec = self._extract_mel(path)

        return mel_spec.unsqueeze(0), bin_label, path


class Loaders():
    def __init__(self, train_path, test_path, split_ratio=0.9, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)

        meta_train = read_literal(train_path)
        l_test = read_literal(test_path)
        meta_test = dict([(x,[]) for x in l_test])

        # Split paths first so training can use random crops while validation stays deterministic.
        ks = list(meta_train.keys())
        rng = random.Random(seed)
        rng.shuffle(ks)
        train_len = int(len(ks) * split_ratio)
        train_keys = ks[:train_len]
        valid_keys = ks[train_len:]
        meta_train_split = {k: meta_train[k] for k in train_keys}
        meta_valid_split = {k: meta_train[k] for k in valid_keys}

        print("Loading train set...")
        train_set = AudioDataset(meta_train_split, preload=False, crop="random")
        print("Loading valid set...")
        valid_set = AudioDataset(meta_valid_split, preload=True, crop="first")
        print("Loading test set...")
        test_set = AudioDataset(meta_test, preload=True, crop="first")

        self.loaderTrain = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.loaderValid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderTest = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)  # raw logits; use sigmoid only for probabilities


class Pipeline():
    def __init__(self, model, learning_rate, seed = 0):
        # These two lines will (mostly) make things deterministic.
        # You're welcome to modify them to try to get a better solution.
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Autograder may use CPU
        self.model = model.to(self.device) #model.cuda() # Also uncomment these lines for GPU
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        self.eval_mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        self.eval_db = AmplitudeToDB()

    def _mel_for_crop(self, path, crop):
        waveform = extract_waveform(path, crop=crop)
        mel_spec = self.eval_db(self.eval_mel(waveform)).squeeze(0)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        return mel_spec.unsqueeze(0)

    def _predict_multicrop(self, paths):
        crop_names = ["first", "middle", "last"]
        all_crops = []
        for path in paths:
            for crop in crop_names:
                all_crops.append(self._mel_for_crop(path, crop))
        x = torch.stack(all_crops).to(self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits).view(len(paths), len(crop_names), -1)
        return probs.mean(dim=1)

    def evaluate(self, loader, threshold=0.5, outpath=None, multi_crop=False):
        self.model.eval()
        preds, targets, paths = [], [], []
        with torch.no_grad():
            for x, y, ps in tqdm(loader, desc="Evaluating"):
                y = y.to(self.device) #y.cuda()
                if multi_crop:
                    outputs = self._predict_multicrop(list(ps))
                else:
                    x = x.to(self.device) #x.cuda()
                    logits = self.model(x)
                    outputs = torch.sigmoid(logits)
                preds.append(outputs.cpu())
                targets.append(y.cpu())
                paths += list(ps)

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        predictions = {}
        for i in range(preds.shape[0]):
            predictions[paths[i]] = {TAGS[j]: float(preds[i][j]) for j in range(len(TAGS))}

        mAP = None
        if outpath: # Save predictions
            predictions = write_submission_predictions(predictions, outpath, normalize_audio_paths=True)
        else: # Only compute accuracy if we're *not* saving predictions, since we can't compute test accuracy
            mAP = average_precision_score(targets, preds, average='macro')
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs):
        best_mAP = -1.0
        best_epoch = 0
        best_state = None

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for x, y, path in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_predictions, mAP = self.evaluate(val_loader, multi_crop=True)
            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
            print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Val multi-crop mAP: {mAP:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model from epoch {best_epoch} with Val multi-crop mAP = {best_mAP:.4f}")


def run1():
    model = model1()
    model.train(dataroot1 + "/train.json")
    train_preds = model.predict(dataroot1 + "/train.json")
    test_preds = model.predict(dataroot1 + "/test.json", "predictions1.json")

    train_labels = read_literal(dataroot1 + "/train.json")
    acc1 = accuracy1(train_labels, train_preds)
    print("Task 1 training accuracy = " + str(acc1))


def run2():
    model = model2()
    model.train(dataroot2 + "/train.json")
    train_preds = model.predict(dataroot2 + "/train.json")
    test_preds = model.predict(dataroot2 + "/test.json", "predictions2.json")

    train_labels = read_literal(dataroot2 + "/train.json")
    acc2 = accuracy2(train_labels, train_preds)
    print("Task 2 training accuracy = " + str(acc2))


def run3():
    loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")
    model = CNNClassifier()
    pipeline = Pipeline(model, TASK3_LR)

    pipeline.train(loaders.loaderTrain, loaders.loaderValid, TASK3_EPOCHS)
    train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
    valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5, multi_crop=True)
    test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json", multi_crop=True)

    all_train = read_literal(dataroot3 + "/train.json")
    for k in valid_preds:
        # We split our training set into train+valid
        # so need to remove validation instances from the training set for evaluation
        all_train.pop(k, None)
    acc3 = accuracy3(all_train, train_preds)
    print("Task 3 training mAP = " + str(acc3))


if __name__ == "__main__":
    run1()
    run2()
    run3()
