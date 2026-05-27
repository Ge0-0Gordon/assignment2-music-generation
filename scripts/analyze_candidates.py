"""Rank MIDI candidates and create lightweight evaluation artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import analyze_candidate, pitch_class_histogram  # noqa: E402


def read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dir(base: Path, dataset: str) -> Path:
    direct = base / dataset
    if direct.exists():
        return direct
    subset = base / f"{dataset}_subset"
    if subset.exists():
        return subset
    return direct


def candidate_files(dataset: str) -> list[Path]:
    root = run_dir(ROOT / "outputs" / "candidates", dataset)
    return sorted(
        path
        for path in root.glob("*.mid*")
        if not path.name.startswith("roundtrip_")
    )


def select_candidates(
    metrics: list[dict[str, object]],
    dataset: str,
    transformer_only: bool = False,
) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for task in ("unconditioned", "conditioned"):
        task_rows = [
            row
            for row in metrics
            if row["dataset"] == dataset and row["task_type"] == task and row["valid"]
        ]
        if transformer_only:
            task_rows = [row for row in task_rows if row["model_type"] == "transformer"]
        task_rows.sort(key=lambda row: (row["score"], row["model_type"] == "transformer"), reverse=True)
        if task_rows:
            selected[task] = task_rows[0]
    return selected


def copy_selected(selected: dict[str, dict[str, object]], dataset: str) -> list[dict[str, object]]:
    selected_dir = ROOT / "outputs" / "candidates" / "selected" / dataset
    selected_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for task, row in selected.items():
        source = Path(str(row["path"]))
        if not source.is_absolute():
            source = ROOT / source
        target = selected_dir / f"{task}_{row['model_type']}.mid"
        shutil.copy2(source, target)
        copied.append({"dataset": dataset, "task_type": task, "source": str(source.relative_to(ROOT)), "selected_path": str(target.relative_to(ROOT))})
    return copied


def plot_token_lengths(dataset: str, figures_dir: Path) -> str:
    csv_path = run_dir(ROOT / "outputs" / "metrics", dataset) / "token_length_distribution.csv"
    lengths = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lengths.append(int(row["token_length"]))
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=24, color="#4c78a8", edgecolor="white")
    plt.title(f"{dataset.title()} token length distribution")
    plt.xlabel("REMI token length")
    plt.ylabel("MIDI file count")
    plt.tight_layout()
    out = figures_dir / f"{dataset}_token_lengths.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return str(out.relative_to(ROOT))


def plot_pitch_histogram(dataset: str, selected: dict[str, dict[str, object]], figures_dir: Path) -> str | None:
    train_csv = run_dir(ROOT / "outputs" / "metrics", dataset) / "pitch_class_histogram.csv"
    if not selected:
        return None
    candidate = selected.get("unconditioned") or next(iter(selected.values()))
    candidate_path = Path(str(candidate["path"]))
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    train_counts = []
    with train_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            train_counts.append(int(row["count"]))
    generated_counts = pitch_class_histogram(candidate_path)
    total_train = max(sum(train_counts), 1)
    total_generated = max(sum(generated_counts), 1)
    train_norm = [value / total_train for value in train_counts]
    generated_norm = [value / total_generated for value in generated_counts]
    xs = list(range(12))
    plt.figure(figsize=(8, 4))
    plt.bar([x - 0.2 for x in xs], train_norm, width=0.4, label="train")
    plt.bar([x + 0.2 for x in xs], generated_norm, width=0.4, label="selected generated")
    plt.title(f"{dataset.title()} pitch-class histogram")
    plt.xlabel("Pitch class")
    plt.ylabel("Fraction")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    out = figures_dir / f"{dataset}_pitch_class_histogram.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return str(out.relative_to(ROOT))


def main() -> None:
    parser = ArgumentParser(description="Analyze and rank generated MIDI candidates.")
    parser.add_argument("--datasets", nargs="+", default=["nottingham", "maestro"])
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--transformer-only-selected", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    model_rows = []
    candidate_rows = []
    selected_rows = []
    figure_rows = []

    for dataset in args.datasets:
        summary_path = run_dir(ROOT / "outputs" / "metrics", dataset) / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_summary(summary_path)
        dataset_rows.append(
            {
                "dataset": dataset,
                "file_count": summary["file_count"],
                "train_files": summary["train_file_count"],
                "valid_files": summary["valid_file_count"],
                "token_min": summary["token_length_stats"]["min"],
                "token_max": summary["token_length_stats"]["max"],
                "token_mean": summary["token_length_stats"]["mean"],
                "train_windows": summary["train_window_count"],
                "valid_windows": summary["valid_window_count"],
                "vocab_size": summary["vocab_size"],
            }
        )
        model_rows.append(
            {
                "dataset": dataset,
                "markov_valid_perplexity": summary["markov"]["valid_perplexity"],
                "transformer_train_loss_last": summary["transformer"]["train_loss_last"],
                "transformer_valid_loss": summary["transformer"]["valid_loss"],
                "transformer_valid_perplexity": summary["transformer"]["valid_perplexity"],
                "transformer_params": summary["transformer"]["parameter_count"],
                "block_size": summary["transformer"]["block_size"],
                "steps_completed": summary["transformer"]["steps_completed"],
            }
        )
        for candidate_path in candidate_files(dataset):
            candidate_rows.append(asdict(analyze_candidate(candidate_path)))
        selected = select_candidates(candidate_rows, dataset, transformer_only=args.transformer_only_selected)
        selected_rows.extend(copy_selected(selected, dataset))
        token_fig = plot_token_lengths(dataset, figures_dir)
        figure_rows.append({"dataset": dataset, "figure": token_fig, "kind": "token_length_distribution"})
        pitch_fig = plot_pitch_histogram(dataset, selected, figures_dir)
        if pitch_fig:
            figure_rows.append({"dataset": dataset, "figure": pitch_fig, "kind": "pitch_class_histogram"})

    write_csv(tables_dir / "dataset_summary.csv", dataset_rows, list(dataset_rows[0].keys()))
    write_csv(tables_dir / "model_metrics.csv", model_rows, list(model_rows[0].keys()))
    write_csv(tables_dir / "candidate_ranking.csv", candidate_rows, list(candidate_rows[0].keys()))
    write_csv(tables_dir / "selected_candidates.csv", selected_rows, ["dataset", "task_type", "source", "selected_path"])
    write_csv(tables_dir / "figures.csv", figure_rows, ["dataset", "figure", "kind"])
    summary = {
        "dataset_summary": str((tables_dir / "dataset_summary.csv").relative_to(ROOT)),
        "model_metrics": str((tables_dir / "model_metrics.csv").relative_to(ROOT)),
        "candidate_ranking": str((tables_dir / "candidate_ranking.csv").relative_to(ROOT)),
        "selected_candidates": str((tables_dir / "selected_candidates.csv").relative_to(ROOT)),
        "figures": figure_rows,
        "selected": selected_rows,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
