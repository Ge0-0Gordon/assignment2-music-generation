"""Build an interactive MAESTRO training-control notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "maestro_training_control.ipynb"


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (cse253)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nb.cells = [
        markdown_cell(
            """
# MAESTRO MIDI-Only Training Control

This notebook is an experiment controller for MAESTRO MIDI-only training, resume, candidate generation, ranking, and indexed run saving. It is not the final submission workbook and does not create `submission/` files or export HTML.

Long training cells are guarded by booleans. Review the command that is printed, then set the relevant guard to `True` when you want to run it.
            """
        ),
        markdown_cell("## 1. Project Setup"),
        code_cell(
            r"""
from pathlib import Path
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "scripts" / "train_main.py").exists() and (path / "src").exists():
            return path
    raise RuntimeError("Could not find project root containing scripts/train_main.py and src/")


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("project_root:", PROJECT_ROOT)
print("python:", sys.executable)
print("python_version:", sys.version.split()[0])
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import src.evaluate as evaluate
import scripts.train_main as train_main
import scripts.prepare_maestro_full as prepare_maestro_full

print("project imports: ok")
            """
        ),
        markdown_cell("## 2. Experiment Configuration"),
        code_cell(
            r"""
CONFIG = {
    "dataset_name": "maestro_full",
    "input_dir": "data/raw/maestro_full/midi",
    "manifest_csv": "data/raw/maestro_full/manifest.csv",
    "output_dir": "outputs/candidates/maestro_full",
    "metrics_dir": "outputs/metrics/maestro_full",
    "checkpoint_dir": "outputs/checkpoints/maestro_full",
    "max_files": 0,
    "block_size": 256,
    "stride": 256,
    "valid_fraction": 0.2,
    "batch_size": 16,
    "lr": 0.0002,
    "weight_decay": 0.01,
    "n_embd": 256,
    "n_layer": 4,
    "n_head": 4,
    "dropout": 0.1,
    "grad_clip": 1.0,
    "epochs": 100,
    "max_steps": 5000,
    "eval_interval": 250,
    "generate_tokens": 512,
    "candidate_count": 5,
    "temperatures": [0.7, 0.8, 0.9, 1.0],
    "top_ks": [20, 50],
    "seed": 253,
    "resume_checkpoint": "outputs/checkpoints/maestro_full/best_transformer.pt",
}

CONFIG
            """
        ),
        markdown_cell("## 3. Data Inspection"),
        code_cell(
            r"""
def run_command(args, *, check=True):
    print(">", " ".join(str(arg) for arg in args))
    result = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}")
    return result


manifest_path = PROJECT_ROOT / CONFIG["manifest_csv"]
input_dir = PROJECT_ROOT / CONFIG["input_dir"]

if not manifest_path.exists():
    print("Manifest missing; preparing MAESTRO full MIDI-only manifest from local zip.")
    run_command([
        sys.executable,
        "scripts/prepare_maestro_full.py",
        "--zip-path",
        "data/maestro-v3.0.0-midi.zip",
        "--output-dir",
        str(Path(CONFIG["manifest_csv"]).parent),
    ])
else:
    print("manifest exists:", manifest_path)

midi_files = sorted(input_dir.rglob("*.mid")) + sorted(input_dir.rglob("*.midi"))
print("midi_file_count:", len(midi_files))

if manifest_path.exists():
    manifest_df = pd.read_csv(manifest_path)
    display(manifest_df["split"].value_counts().rename_axis("split").reset_index(name="files"))
    display(manifest_df.head())

summary_path = PROJECT_ROOT / CONFIG["metrics_dir"] / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("existing metrics summary:", summary_path)
    display(pd.DataFrame([{
        "dataset": summary["dataset_name"],
        "files": summary["file_count"],
        "skipped": summary.get("skipped_file_count", 0),
        "train_files": summary["train_file_count"],
        "valid_files": summary["valid_file_count"],
        "token_min": summary["token_length_stats"]["min"],
        "token_max": summary["token_length_stats"]["max"],
        "token_mean": summary["token_length_stats"]["mean"],
        "train_windows": summary["train_window_count"],
        "valid_windows": summary["valid_window_count"],
        "vocab_size": summary["vocab_size"],
    }]))
else:
    print("No existing metrics summary yet. Run training to create token/window statistics.")
            """
        ),
        markdown_cell("## 4. Training / Resume"),
        code_cell(
            r"""
def list_arg(values):
    return ",".join(str(value) for value in values)


def train_command(config):
    cmd = [
        sys.executable,
        "scripts/train_main.py",
        "--dataset-name", config["dataset_name"],
        "--input-dir", config["input_dir"],
        "--manifest-csv", config["manifest_csv"],
        "--output-dir", config["output_dir"],
        "--metrics-dir", config["metrics_dir"],
        "--max-files", str(config["max_files"]),
        "--block-size", str(config["block_size"]),
        "--stride", str(config["stride"]),
        "--valid-fraction", str(config["valid_fraction"]),
        "--epochs", str(config["epochs"]),
        "--max-steps", str(config["max_steps"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--n-embd", str(config["n_embd"]),
        "--n-layer", str(config["n_layer"]),
        "--n-head", str(config["n_head"]),
        "--dropout", str(config["dropout"]),
        "--grad-clip", str(config["grad_clip"]),
        "--checkpoint-dir", config["checkpoint_dir"],
        "--eval-interval", str(config["eval_interval"]),
        "--generate-tokens", str(config["generate_tokens"]),
        "--candidate-count", str(config["candidate_count"]),
        "--temperatures", list_arg(config["temperatures"]),
        "--top-ks", list_arg(config["top_ks"]),
        "--seed", str(config["seed"]),
    ]
    if config.get("resume_checkpoint"):
        cmd.extend(["--resume-checkpoint", config["resume_checkpoint"]])
    return cmd


RUN_TRAINING = False  # Change to True when you are ready to launch a long training/resume run.
cmd = train_command(CONFIG)
print("Training command:")
print(" ".join(cmd))

if RUN_TRAINING:
    run_command(cmd)
    summary = json.loads((PROJECT_ROOT / CONFIG["metrics_dir"] / "summary.json").read_text(encoding="utf-8"))
    print("best_checkpoint:", summary["transformer"]["best_checkpoint"])
    print("valid_loss:", summary["transformer"]["valid_loss"])
    print("valid_perplexity:", summary["transformer"]["valid_perplexity"])
else:
    print("Training not started. Set RUN_TRAINING = True in this cell to run it.")
            """
        ),
        markdown_cell("## 5. Loss Visualization"),
        code_cell(
            r"""
metrics_dir = PROJECT_ROOT / CONFIG["metrics_dir"]
summary_path = metrics_dir / "summary.json"
history_path = metrics_dir / "training_history.csv"

if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("best_checkpoint:", summary["transformer"].get("best_checkpoint"))
    print("total_steps:", summary["transformer"].get("total_steps_including_resume"))
else:
    print("No summary.json found yet.")

if history_path.exists():
    history = pd.read_csv(history_path)
    display(history.tail())
    if len(history) >= 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if "train_loss" in history and history["train_loss"].notna().any():
            axes[0].plot(history["total_step"], history["train_loss"], marker="o", label="train loss")
        if "valid_loss" in history and history["valid_loss"].notna().any():
            axes[0].plot(history["total_step"], history["valid_loss"], marker="o", label="valid loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("total step")
        axes[0].legend()
        if "valid_perplexity" in history and history["valid_perplexity"].notna().any():
            axes[1].plot(history["total_step"], history["valid_perplexity"], marker="o", color="tab:orange")
        axes[1].set_title("Validation perplexity")
        axes[1].set_xlabel("total step")
        plt.tight_layout()
        plt.show()
    else:
        print("training_history.csv exists but has no rows yet.")
else:
    print("No training_history.csv found yet. Run training with eval_interval > 0 to create it.")
            """
        ),
        markdown_cell("## 6. Candidate Generation"),
        code_cell(
            r"""
def best_checkpoint_from_summary(config):
    summary_path = PROJECT_ROOT / config["metrics_dir"] / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = summary["transformer"].get("best_checkpoint")
        if checkpoint:
            return checkpoint
    return config.get("resume_checkpoint")


def generate_command(config):
    checkpoint = best_checkpoint_from_summary(config)
    if not checkpoint:
        raise RuntimeError("No checkpoint found. Set CONFIG['resume_checkpoint'] or train first.")
    return [
        sys.executable,
        "scripts/train_main.py",
        "--mode", "generate",
        "--dataset-name", config["dataset_name"],
        "--input-dir", config["input_dir"],
        "--manifest-csv", config["manifest_csv"],
        "--output-dir", config["output_dir"],
        "--metrics-dir", config["metrics_dir"],
        "--max-files", str(config["max_files"]),
        "--block-size", str(config["block_size"]),
        "--stride", str(config["stride"]),
        "--valid-fraction", str(config["valid_fraction"]),
        "--batch-size", str(config["batch_size"]),
        "--resume-checkpoint", checkpoint,
        "--generate-tokens", str(config["generate_tokens"]),
        "--candidate-count", str(config["candidate_count"]),
        "--temperatures", list_arg(config["temperatures"]),
        "--top-ks", list_arg(config["top_ks"]),
        "--seed", str(config["seed"]),
    ]


RUN_GENERATION = False  # Change to True to generate/refresh candidates from the best checkpoint.
cmd = generate_command(CONFIG)
print("Generation command:")
print(" ".join(cmd))

if RUN_GENERATION:
    run_command(cmd)
else:
    print("Generation not started. Set RUN_GENERATION = True in this cell to run it.")
            """
        ),
        markdown_cell("## 7. Candidate Evaluation / Ranking"),
        code_cell(
            r"""
RUN_EVALUATION = False  # Change to True after candidates are generated.

eval_cmd = [
    sys.executable,
    "scripts/evaluate_maestro_full.py",
    "--metrics-dir", CONFIG["metrics_dir"],
    "--indexed-dir", "outputs/candidates/final/maestro",
    "--output-dir", "outputs/evaluation/maestro_full",
    "--nottingham-summary", "outputs/metrics/nottingham_final/summary.json",
    "--nottingham-selected-dir", "outputs/candidates/selected/nottingham_final",
]
print("Evaluation command:")
print(" ".join(eval_cmd))

if RUN_EVALUATION:
    run_command(eval_cmd)

tables_dir = PROJECT_ROOT / "outputs/evaluation/maestro_full/tables"
ranking_path = tables_dir / "candidate_ranking.csv"
selected_path = tables_dir / "selected_candidates.csv"
model_metrics_path = tables_dir / "model_metrics.csv"
dataset_summary_path = tables_dir / "dataset_summary.csv"

for path in [dataset_summary_path, model_metrics_path, ranking_path, selected_path]:
    print(path.relative_to(PROJECT_ROOT), "exists=", path.exists())

if ranking_path.exists():
    ranking_df = pd.read_csv(ranking_path)
    display(ranking_df.sort_values("score", ascending=False).head(20))
else:
    print("No candidate_ranking.csv yet.")

if selected_path.exists():
    selected_df = pd.read_csv(selected_path)
    display(selected_df[["run", "task_type", "selected_path", "note_count", "duration_seconds", "score"]])
else:
    print("No selected_candidates.csv yet.")
            """
        ),
        markdown_cell("## 8. Indexed Run Saving"),
        code_cell(
            r"""
import shutil
from dataclasses import asdict

from src.evaluate import analyze_candidate


RUN_ID = "run_004"  # Change this before saving a new indexed run.
RUN_NOTES = "Listening notes placeholder. Add observations after auditioning."


def save_single_indexed_run(config, run_id, notes):
    source_dir = PROJECT_ROOT / config["output_dir"]
    run_dir = PROJECT_ROOT / "outputs/candidates/final/maestro" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_paths = sorted(source_dir.glob("transformer_*.mid*"))
    rows = []
    for path in candidate_paths:
        row = asdict(analyze_candidate(path))
        row["dataset"] = config["dataset_name"]
        row["source_path"] = str(path.relative_to(PROJECT_ROOT))
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No transformer candidates found under {source_dir}")
    ranking_df = pd.DataFrame(rows).sort_values(["task_type", "score"], ascending=[True, False])
    ranking_df.to_csv(run_dir / "candidate_ranking.csv", index=False)

    selected_rows = []
    for task, filename in {
        "unconditioned": "symbolic_unconditioned.mid",
        "conditioned": "symbolic_conditioned.mid",
    }.items():
        task_df = ranking_df[(ranking_df["task_type"] == task) & (ranking_df["valid"] == True)].sort_values("score", ascending=False)
        if task_df.empty:
            print(f"No valid {task} candidate found.")
            continue
        selected = task_df.iloc[0].to_dict()
        source = Path(selected["path"])
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        target = run_dir / filename
        shutil.copy2(source, target)
        selected_rows.append({
            "run": run_id,
            "dataset": config["dataset_name"],
            "task_type": task,
            "selected_path": str(target.relative_to(PROJECT_ROOT)),
            "source_path": str(source.relative_to(PROJECT_ROOT)),
            "score": selected["score"],
            "note_count": selected["note_count"],
            "duration_seconds": selected["duration_seconds"],
            "notes_per_second": selected["notes_per_second"],
            "pitch_range": selected["pitch_range"],
            "max_simultaneous_notes": selected["max_simultaneous_notes"],
            "repeated_pitch_bigram_rate": selected["repeated_pitch_bigram_rate"],
        })

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(run_dir / "selected_candidates.csv", index=False)
    generation_config = {
        "run": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": config["dataset_name"],
        "source_dir": config["output_dir"],
        "checkpoint": best_checkpoint_from_summary(config),
        "generate_tokens": config["generate_tokens"],
        "candidate_count": config["candidate_count"],
        "temperatures": config["temperatures"],
        "top_ks": config["top_ks"],
        "selection_rule": "highest valid heuristic score per task_type",
    }
    (run_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2), encoding="utf-8")
    (run_dir / "notes.txt").write_text(notes.strip() + "\n", encoding="utf-8")
    return run_dir, selected_df


SAVE_INDEXED_RUN = False  # Change to True after reviewing candidates and setting RUN_ID.
if SAVE_INDEXED_RUN:
    run_dir, selected_df = save_single_indexed_run(CONFIG, RUN_ID, RUN_NOTES)
    print("saved:", run_dir)
    display(selected_df)
else:
    print("Indexed run not saved. Set SAVE_INDEXED_RUN = True after setting RUN_ID.")
            """
        ),
        markdown_cell("## 9. Quick Listening Helper"),
        code_cell(
            r"""
selected_path = PROJECT_ROOT / "outputs/evaluation/maestro_full/tables/selected_candidates.csv"
if selected_path.exists():
    selected_df = pd.read_csv(selected_path)
    display(selected_df[["run", "task_type", "selected_path", "note_count", "duration_seconds", "notes_per_second", "pitch_range"]])
    print("Open a MIDI from PowerShell with:")
    print("Invoke-Item 'outputs\\candidates\\final\\maestro\\run_001\\symbolic_conditioned.mid'")
else:
    print("No selected candidates table yet.")
            """
        ),
        markdown_cell(
            """
## 10. Experiment Notes

Use this cell after listening.

- Run ID:
- Sampling settings:
- Unconditioned listening notes:
- Conditioned listening notes:
- Best run so far:
- Should continue training?
- Next parameter change:
            """
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}")
    print(f"cells {len(nb.cells)}")


if __name__ == "__main__":
    main()
