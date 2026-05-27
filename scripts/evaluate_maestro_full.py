"""Write MAESTRO-full evaluation tables and figures."""

from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import analyze_candidate, pitch_class_histogram  # noqa: E402


def resolve(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_row(summary: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": summary["dataset_name"],
        "input_dir": summary["input_dir"],
        "manifest_source": summary.get("manifest_source"),
        "file_count": summary["file_count"],
        "skipped_file_count": summary.get("skipped_file_count", 0),
        "train_files": summary["train_file_count"],
        "valid_files": summary["valid_file_count"],
        "token_min": summary["token_length_stats"]["min"],
        "token_max": summary["token_length_stats"]["max"],
        "token_mean": summary["token_length_stats"]["mean"],
        "train_windows": summary["train_window_count"],
        "valid_windows": summary["valid_window_count"],
        "vocab_size": summary["vocab_size"],
    }


def model_row(summary: dict[str, object]) -> dict[str, object]:
    transformer = summary["transformer"]
    return {
        "dataset": summary["dataset_name"],
        "markov_valid_perplexity": summary["markov"]["valid_perplexity"],
        "transformer_train_loss_last": transformer["train_loss_last"],
        "transformer_valid_loss": transformer["valid_loss"],
        "transformer_valid_perplexity": transformer["valid_perplexity"],
        "transformer_params": transformer["parameter_count"],
        "block_size": transformer["block_size"],
        "batch_size": transformer["batch_size"],
        "n_embd": transformer["n_embd"],
        "n_layer": transformer["n_layer"],
        "n_head": transformer["n_head"],
        "steps_completed": transformer["steps_completed"],
        "total_steps_including_resume": transformer["total_steps_including_resume"],
        "best_checkpoint": transformer["best_checkpoint"],
    }


def plot_token_lengths(token_csv: Path, out: Path) -> str:
    lengths = [int(row["token_length"]) for row in read_csv_rows(token_csv)]
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=32, color="#4c78a8", edgecolor="white")
    plt.title("MAESTRO full token length distribution")
    plt.xlabel("REMI token length")
    plt.ylabel("MIDI file count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return str(out.relative_to(ROOT))


def plot_pitch_histogram(train_csv: Path, selected_path: Path | None, out: Path) -> str | None:
    if selected_path is None or not selected_path.exists():
        return None
    train_counts = [int(row["count"]) for row in read_csv_rows(train_csv)]
    generated_counts = pitch_class_histogram(selected_path)
    total_train = max(sum(train_counts), 1)
    total_generated = max(sum(generated_counts), 1)
    train_norm = [value / total_train for value in train_counts]
    generated_norm = [value / total_generated for value in generated_counts]
    xs = list(range(12))
    plt.figure(figsize=(8, 4))
    plt.bar([x - 0.2 for x in xs], train_norm, width=0.4, label="train")
    plt.bar([x + 0.2 for x in xs], generated_norm, width=0.4, label="selected generated")
    plt.title("MAESTRO full pitch-class histogram")
    plt.xlabel("Pitch class")
    plt.ylabel("Fraction")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return str(out.relative_to(ROOT))


def indexed_candidate_rows(indexed_dir: Path, dataset_name: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranking_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for run_dir in sorted(indexed_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        ranking_rows.extend(read_csv_rows(run_dir / "candidate_ranking.csv"))
        for path in sorted(run_dir.glob("symbolic_*.mid")):
            row = asdict(analyze_candidate(path))
            row["dataset"] = dataset_name
            row["model_type"] = "transformer"
            row["run"] = run_dir.name
            row["selected_path"] = str(path.relative_to(ROOT))
            selected_rows.append(row)
    return ranking_rows, selected_rows


def nottingham_selected_rows(selected_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(selected_dir.glob("*.mid")):
        row = asdict(analyze_candidate(path))
        row["dataset"] = "nottingham_final"
        row["run"] = "selected"
        row["selected_path"] = str(path.relative_to(ROOT))
        rows.append(row)
    return rows


def main() -> None:
    parser = ArgumentParser(description="Evaluate MAESTRO full indexed candidates.")
    parser.add_argument("--metrics-dir", default="outputs/metrics/maestro_full")
    parser.add_argument("--indexed-dir", default="outputs/candidates/final/maestro")
    parser.add_argument("--output-dir", default="outputs/evaluation/maestro_full")
    parser.add_argument("--nottingham-summary", default="outputs/metrics/nottingham_final/summary.json")
    parser.add_argument("--nottingham-selected-dir", default="outputs/candidates/selected/nottingham_final")
    args = parser.parse_args()

    metrics_dir = resolve(args.metrics_dir)
    indexed_dir = resolve(args.indexed_dir)
    output_dir = resolve(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(metrics_dir / "summary.json")
    dataset_rows = [dataset_row(summary)]
    model_rows = [model_row(summary)]

    ranking_rows, selected_rows = indexed_candidate_rows(indexed_dir, str(summary["dataset_name"]))
    nottingham_summary_path = resolve(args.nottingham_summary)
    comparison_rows = selected_rows + nottingham_selected_rows(resolve(args.nottingham_selected_dir))
    if nottingham_summary_path.exists():
        nottingham_summary = read_summary(nottingham_summary_path)
        dataset_rows.append(dataset_row(nottingham_summary))
        model_rows.append(model_row(nottingham_summary))

    write_csv(tables_dir / "dataset_summary.csv", dataset_rows, list(dataset_rows[0].keys()))
    write_csv(tables_dir / "model_metrics.csv", model_rows, list(model_rows[0].keys()))
    if ranking_rows:
        write_csv(tables_dir / "candidate_ranking.csv", ranking_rows, list(ranking_rows[0].keys()))
    if selected_rows:
        write_csv(tables_dir / "selected_candidates.csv", selected_rows, list(selected_rows[0].keys()))
    if comparison_rows:
        write_csv(tables_dir / "nottingham_vs_maestro_selected.csv", comparison_rows, list(comparison_rows[0].keys()))

    figure_rows = []
    token_fig = plot_token_lengths(
        metrics_dir / "token_length_distribution.csv",
        figures_dir / "maestro_full_token_lengths.png",
    )
    figure_rows.append({"dataset": summary["dataset_name"], "figure": token_fig, "kind": "token_length_distribution"})
    selected_unconditioned = next(
        (resolve(str(row["selected_path"])) for row in selected_rows if row.get("task_type") == "unconditioned"),
        None,
    )
    pitch_fig = plot_pitch_histogram(
        metrics_dir / "pitch_class_histogram.csv",
        selected_unconditioned,
        figures_dir / "maestro_full_pitch_class_histogram.png",
    )
    if pitch_fig:
        figure_rows.append({"dataset": summary["dataset_name"], "figure": pitch_fig, "kind": "pitch_class_histogram"})
    write_csv(tables_dir / "figures.csv", figure_rows, ["dataset", "figure", "kind"])

    out_summary = {
        "dataset_summary": str((tables_dir / "dataset_summary.csv").relative_to(ROOT)),
        "model_metrics": str((tables_dir / "model_metrics.csv").relative_to(ROOT)),
        "candidate_ranking": str((tables_dir / "candidate_ranking.csv").relative_to(ROOT)),
        "selected_candidates": str((tables_dir / "selected_candidates.csv").relative_to(ROOT)),
        "nottingham_vs_maestro_selected": str((tables_dir / "nottingham_vs_maestro_selected.csv").relative_to(ROOT)),
        "figures": figure_rows,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()
