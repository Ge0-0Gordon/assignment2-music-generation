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
BASE_CONFIG = {
    "dataset_name": "maestro_full",
    "input_dir": "data/raw/maestro_full/midi",
    "manifest_csv": "data/raw/maestro_full/manifest.csv",
    "output_dir": "outputs/candidates/maestro_full",
    "metrics_dir": "outputs/metrics/maestro_full",
    "checkpoint_dir": "outputs/checkpoints/maestro_full",
    "final_indexed_dir": "outputs/candidates/final/maestro",
    "evaluation_dir": "outputs/evaluation/maestro_full",
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
    "generate_tokens": 1024,
    "candidate_count": 20,
    "decode_retry_attempts": 1,
    "temperatures": [0.8, 0.9, 1.0],
    "top_ks": [50, 100],
    "generation_mode": "structural_seeded",  # "pure_bos", "structural_seeded", or "piece_start_seeded"
    "unconditioned_prefix_tokens": 32,
    "primer_source": "valid",
    "primer_index": None,
    "run_id": "run_050k_seeded",
    "seed": 253,
    "resume_checkpoint": "outputs/checkpoints/maestro_full/best_transformer.pt",
    "min_notes": None,
    "max_notes": None,
    "min_notes_per_second": None,
    "max_notes_per_second": None,
    "max_token_length": None,
    "max_polyphony": None,
}

EXPERIMENTS = {
    "maestro_full": {
        "dataset_name": "maestro_full",
        "input_dir": "data/raw/maestro_full/midi",
        "manifest_csv": "data/raw/maestro_full/manifest.csv",
        "output_dir": "outputs/candidates/maestro_full",
        "metrics_dir": "outputs/metrics/maestro_full",
        "checkpoint_dir": "outputs/checkpoints/maestro_full",
        "final_indexed_dir": "outputs/candidates/final/maestro",
        "evaluation_dir": "outputs/evaluation/maestro_full",
        "resume_checkpoint": "outputs/checkpoints/maestro_full/best_transformer.pt",
        "run_id": "run_050k_seeded",
        "lr": 0.0002,
        "max_steps": 5000,
        "min_notes": None,
        "max_notes": None,
        "min_notes_per_second": None,
        "max_notes_per_second": None,
        "max_token_length": None,
        "max_polyphony": None,
    },
    "maestro_clean": {
        "dataset_name": "maestro_clean",
        "input_dir": "data/raw/maestro_clean/midi",
        "manifest_csv": "data/raw/maestro_clean/manifest.csv",
        "output_dir": "outputs/candidates/maestro_clean",
        "metrics_dir": "outputs/metrics/maestro_clean",
        "checkpoint_dir": "outputs/checkpoints/maestro_clean",
        "final_indexed_dir": "outputs/candidates/final/maestro_clean",
        "evaluation_dir": "outputs/evaluation/maestro_clean",
        "resume_checkpoint": "",
        "run_id": "run_clean_10k_seeded",
        "lr": 0.0003,
        "max_steps": 10000,
        "block_size": 256,
        "stride": 256,
        "n_embd": 256,
        "n_layer": 4,
        "n_head": 4,
        "batch_size": 16,
        "generation_mode": "structural_seeded",
        "unconditioned_prefix_tokens": 32,
        "primer_source": "valid",
        "primer_index": None,
        "candidate_count": 20,
        "generate_tokens": 1024,
        "min_notes": 50,
        "min_notes_per_second": 0.8,
        "max_notes_per_second": 10.0,
        "max_token_length": 32768,
        "max_polyphony": 32,
    },
    "nottingham_final_ctx512": {
        "dataset_name": "nottingham_final_ctx512",
        "input_dir": "data/nottingham-dataset-master/MIDI",
        "manifest_csv": "",
        "output_dir": "outputs/candidates/nottingham_final_ctx512",
        "metrics_dir": "outputs/metrics/nottingham_final_ctx512",
        "checkpoint_dir": "outputs/checkpoints/nottingham_final_ctx512",
        "final_indexed_dir": "outputs/candidates/final/nottingham_final_ctx512",
        "evaluation_dir": "outputs/evaluation/nottingham_final_ctx512",
        "resume_checkpoint": "",
        "run_id": "run_nottingham_10k_seeded",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "max_steps": 10000,
        "eval_interval": 250,
        "block_size": 512,
        "stride": 256,
        "valid_fraction": 0.1,
        "n_embd": 256,
        "n_layer": 4,
        "n_head": 4,
        "dropout": 0.1,
        "batch_size": 16,
        "generate_tokens": 768,
        "candidate_count": 50,
        "temperatures": [0.7, 0.8, 0.9],
        "top_ks": [20, 50],
        "generation_mode": "structural_seeded",
        "unconditioned_prefix_tokens": 32,
        "primer_source": "valid",
        "primer_index": None,
        "min_notes": None,
        "max_notes": None,
        "min_notes_per_second": None,
        "max_notes_per_second": None,
        "max_token_length": None,
        "max_polyphony": None,
    },
    "nottingham_final_ctx1024": {
        "dataset_name": "nottingham_final_ctx1024",
        "input_dir": "data/nottingham-dataset-master/MIDI",
        "manifest_csv": "",
        "output_dir": "outputs/candidates/nottingham_final_ctx1024",
        "metrics_dir": "outputs/metrics/nottingham_final_ctx1024",
        "checkpoint_dir": "outputs/checkpoints/nottingham_final_ctx1024",
        "final_indexed_dir": "outputs/candidates/final/nottingham_final_ctx1024",
        "evaluation_dir": "outputs/evaluation/nottingham_final_ctx1024",
        "resume_checkpoint": "",
        "run_id": "run_nottingham_10k_seeded",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "max_steps": 10000,
        "eval_interval": 250,
        "block_size": 1024,
        "stride": 512,
        "valid_fraction": 0.1,
        "n_embd": 256,
        "n_layer": 4,
        "n_head": 4,
        "dropout": 0.1,
        "batch_size": 8,
        "generate_tokens": 768,
        "candidate_count": 50,
        "temperatures": [0.7, 0.8, 0.9],
        "top_ks": [20, 50],
        "generation_mode": "structural_seeded",
        "unconditioned_prefix_tokens": 32,
        "primer_source": "valid",
        "primer_index": None,
        "min_notes": None,
        "max_notes": None,
        "min_notes_per_second": None,
        "max_notes_per_second": None,
        "max_token_length": None,
        "max_polyphony": None,
    },
    "nottingham_final_retrain": {
        "dataset_name": "nottingham_final_retrain",
        "input_dir": "data/nottingham-dataset-master/MIDI",
        "manifest_csv": "",
        "output_dir": "outputs/candidates/nottingham_final_retrain",
        "metrics_dir": "outputs/metrics/nottingham_final_retrain",
        "checkpoint_dir": "outputs/checkpoints/nottingham_final_retrain",
        "final_indexed_dir": "outputs/candidates/final/nottingham_final_retrain",
        "evaluation_dir": "outputs/evaluation/nottingham_final_retrain",
        "resume_checkpoint": "",
        "run_id": "run_conditioned_bigpool",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "max_steps": 30000,
        "eval_interval": 250,
        "block_size": 512,
        "stride": 256,
        "valid_fraction": 0.1,
        "n_embd": 256,
        "n_layer": 4,
        "n_head": 4,
        "dropout": 0.1,
        "batch_size": 16,
        "generate_tokens": 64,
        "candidate_count": 1,
        "temperatures": [0.8],
        "top_ks": [20],
        "generation_mode": "structural_seeded",
        "unconditioned_prefix_tokens": 32,
        "primer_source": "valid",
        "primer_index": None,
        "final_generate_tokens": 768,
        "task1_candidate_count": 100,
        "conditioned_prefix_count": 20,
        "conditioned_prefix_tokens": 128,
        "conditioned_candidates_per_prefix": 10,
        "final_temperatures": [0.7, 0.8, 0.9],
        "final_top_ks": [20, 50],
        "min_notes": None,
        "max_notes": None,
        "min_notes_per_second": None,
        "max_notes_per_second": None,
        "max_token_length": None,
        "max_polyphony": None,
    },
}


CONFIG = BASE_CONFIG.copy()
EXPERIMENT = "nottingham_final_retrain"  # Options include "nottingham_final_ctx512", "nottingham_final_ctx1024", "maestro_full", "maestro_clean".


def select_experiment(name):
    CONFIG.clear()
    CONFIG.update(BASE_CONFIG)
    CONFIG.update(EXPERIMENTS[name])
    CONFIG["experiment"] = name
    return CONFIG


# Recommended Nottingham final default: nottingham_final_retrain.
# It starts from scratch when resume_checkpoint="" and uses no pretrained weights.
select_experiment(EXPERIMENT)
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


manifest_path = PROJECT_ROOT / CONFIG["manifest_csv"] if CONFIG.get("manifest_csv") else None
input_dir = PROJECT_ROOT / CONFIG["input_dir"]

if manifest_path is None:
    print("No manifest configured; train_main.py will discover MIDI files directly from:", input_dir)
elif not manifest_path.exists():
    print("manifest missing:", manifest_path)
    print("Use the clean manifest preparation cell below if EXPERIMENT == 'maestro_clean'.")
    print("For maestro_full, run scripts/prepare_maestro_full.py manually after checking local data paths.")
else:
    print("manifest exists:", manifest_path)

midi_files = sorted(input_dir.rglob("*.mid")) + sorted(input_dir.rglob("*.midi"))
print("midi_file_count:", len(midi_files))

if manifest_path is not None and manifest_path.exists():
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
        markdown_cell("## 3a. Clean Manifest Preparation"),
        code_cell(
            r"""
def prepare_clean_command(config):
    output_dir = str(Path(config["manifest_csv"]).parent)
    cmd = [
        sys.executable,
        "scripts/prepare_maestro_full.py",
        "--zip-path", "data/maestro-v3.0.0-midi.zip",
        "--output-dir", output_dir,
    ]
    for flag, key in [
        ("--min-notes", "min_notes"),
        ("--max-notes", "max_notes"),
        ("--min-notes-per-second", "min_notes_per_second"),
        ("--max-notes-per-second", "max_notes_per_second"),
        ("--max-token-length", "max_token_length"),
        ("--max-polyphony", "max_polyphony"),
    ]:
        if config.get(key) is not None:
            cmd.extend([flag, str(config[key])])
    return cmd


RUN_PREPARE_CLEAN = False  # Set True only when EXPERIMENT == "maestro_clean" and you want to build the clean manifest.

if EXPERIMENT != "maestro_clean":
    print("Current EXPERIMENT is not maestro_clean; this cell only prints the clean preparation command when switched.")
else:
    clean_manifest = PROJECT_ROOT / CONFIG["manifest_csv"]
    clean_cmd = prepare_clean_command(CONFIG)
    print("clean manifest:", clean_manifest)
    print("exists:", clean_manifest.exists())
    print("Prepare command:")
    print(" ".join(clean_cmd))
    if RUN_PREPARE_CLEAN:
        run_command(clean_cmd)
    else:
        print("Not running prepare. Set RUN_PREPARE_CLEAN = True to create data/raw/maestro_clean/.")
            """
        ),
        markdown_cell("## 4. Training / Resume"),
        code_cell(
            r"""
def list_arg(values):
    return ",".join(str(value) for value in values)


def add_optional_quality_args(cmd, config):
    for flag, key in [
        ("--min-notes", "min_notes"),
        ("--max-notes", "max_notes"),
        ("--min-notes-per-second", "min_notes_per_second"),
        ("--max-notes-per-second", "max_notes_per_second"),
        ("--max-token-length", "max_token_length"),
        ("--max-polyphony", "max_polyphony"),
    ]:
        if config.get(key) is not None:
            cmd.extend([flag, str(config[key])])
    return cmd


def train_command(config):
    cmd = [
        sys.executable,
        "scripts/train_main.py",
        "--dataset-name", config["dataset_name"],
        "--input-dir", config["input_dir"],
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
        "--decode-retry-attempts", str(config["decode_retry_attempts"]),
        "--temperatures", list_arg(config["temperatures"]),
        "--top-ks", list_arg(config["top_ks"]),
        "--unconditioned-mode", config["generation_mode"],
        "--unconditioned-prefix-tokens", str(config["unconditioned_prefix_tokens"]),
        "--primer-source", config["primer_source"],
        "--seed", str(config["seed"]),
    ]
    if config.get("manifest_csv"):
        cmd.extend(["--manifest-csv", config["manifest_csv"]])
    if config.get("primer_index") is not None:
        cmd.extend(["--primer-index", str(config["primer_index"])])
    if config.get("resume_checkpoint"):
        cmd.extend(["--resume-checkpoint", config["resume_checkpoint"]])
    return add_optional_quality_args(cmd, config)


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
    cmd = [
        sys.executable,
        "scripts/train_main.py",
        "--mode", "generate",
        "--dataset-name", config["dataset_name"],
        "--input-dir", config["input_dir"],
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
        "--decode-retry-attempts", str(config["decode_retry_attempts"]),
        "--temperatures", list_arg(config["temperatures"]),
        "--top-ks", list_arg(config["top_ks"]),
        "--unconditioned-mode", config["generation_mode"],
        "--unconditioned-prefix-tokens", str(config["unconditioned_prefix_tokens"]),
        "--primer-source", config["primer_source"],
        "--seed", str(config["seed"]),
    ]
    if config.get("manifest_csv"):
        cmd.extend(["--manifest-csv", config["manifest_csv"]])
    if config.get("primer_index") is not None:
        cmd.extend(["--primer-index", str(config["primer_index"])])
    return add_optional_quality_args(cmd, config)


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
        markdown_cell("## 6a. Nottingham Final Retrain Runs"),
        code_cell(
            r"""
def nottingham_final_retrain_command(config):
    checkpoint = best_checkpoint_from_summary(config)
    if not checkpoint:
        raise RuntimeError("No checkpoint found. Train nottingham_final_retrain first.")
    return [
        sys.executable,
        "scripts/generate_nottingham_final_retrain.py",
        "--metrics-dir", config["metrics_dir"],
        "--checkpoint", checkpoint,
        "--indexed-dir", config["final_indexed_dir"],
        "--evaluation-dir", config["evaluation_dir"],
        "--generate-tokens", str(config.get("final_generate_tokens", 768)),
        "--task1-candidate-count", str(config.get("task1_candidate_count", 100)),
        "--conditioned-prefix-count", str(config.get("conditioned_prefix_count", 20)),
        "--conditioned-prefix-tokens", str(config.get("conditioned_prefix_tokens", 128)),
        "--conditioned-candidates-per-prefix", str(config.get("conditioned_candidates_per_prefix", 10)),
        "--temperatures", list_arg(config.get("final_temperatures", config["temperatures"])),
        "--top-ks", list_arg(config.get("final_top_ks", config["top_ks"])),
        "--seed", str(config["seed"]),
    ]


RUN_FINAL_RETRAIN_GENERATION = False  # Set True after nottingham_final_retrain checkpoint exists.
if CONFIG["dataset_name"] != "nottingham_final_retrain":
    print("Switch EXPERIMENT to nottingham_final_retrain to use this final generation cell.")
else:
    final_cmd = nottingham_final_retrain_command(CONFIG)
    print("Final retrain generation command:")
    print(" ".join(final_cmd))
    if RUN_FINAL_RETRAIN_GENERATION:
        run_command(final_cmd)
    else:
        print("Final retrain generation not started. Set RUN_FINAL_RETRAIN_GENERATION = True to run it.")
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
    "--indexed-dir", CONFIG["final_indexed_dir"],
    "--output-dir", CONFIG["evaluation_dir"],
    "--nottingham-summary", "outputs/metrics/nottingham_final/summary.json",
    "--nottingham-selected-dir", "outputs/candidates/selected/nottingham_final",
]
print("Evaluation command:")
print(" ".join(eval_cmd))

if RUN_EVALUATION:
    run_command(eval_cmd)

tables_dir = PROJECT_ROOT / CONFIG["evaluation_dir"] / "tables"
ranking_path = tables_dir / "candidate_ranking.csv"
selected_path = tables_dir / "selected_candidates.csv"
model_metrics_path = tables_dir / "model_metrics.csv"
dataset_summary_path = tables_dir / "dataset_summary.csv"

for path in [dataset_summary_path, model_metrics_path, ranking_path, selected_path]:
    print(path.relative_to(PROJECT_ROOT), "exists=", path.exists())

if ranking_path.exists():
    ranking_df = pd.read_csv(ranking_path)
    display(ranking_df.sort_values("score", ascending=False).head(20))
    if "usable" in ranking_df and "reject_reason" in ranking_df:
        summary_cols = ["run", "task_type"] if "run" in ranking_df else ["task_type"]
        display(
            ranking_df.groupby(summary_cols + ["usable"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        display(
            ranking_df.loc[ranking_df["usable"] == False, "reject_reason"]
            .value_counts()
            .head(15)
            .rename_axis("reject_reason")
            .reset_index(name="count")
        )
else:
    print("No candidate_ranking.csv yet.")

if selected_path.exists():
    selected_df = pd.read_csv(selected_path)
    display(selected_df[["run", "task_type", "selected_path", "note_count", "duration_seconds", "score"]])
    print("Files to audition:")
    for row in selected_df.itertuples(index=False):
        print(getattr(row, "task_type"), PROJECT_ROOT / getattr(row, "selected_path"))
else:
    print("No selected_candidates.csv yet.")
            """
        ),
        markdown_cell("## 7a. Prefix / Continuation Diagnostics"),
        code_cell(
            r"""
RUN_PREFIX_SPLIT = False  # Set True after indexed run folders exist.

piece_start_dirs = [
    PROJECT_ROOT / CONFIG["final_indexed_dir"] / "run_nottingham_10k_piece_start_128",
    PROJECT_ROOT / CONFIG["final_indexed_dir"] / "run_nottingham_10k_piece_start_256",
]
conditioned_dirs = [
    PROJECT_ROOT / CONFIG["final_indexed_dir"] / "run_nottingham_10k_bos",
    PROJECT_ROOT / CONFIG["final_indexed_dir"] / "run_nottingham_10k_seeded",
    *piece_start_dirs,
]

piece_cmd = [sys.executable, "scripts/split_piece_start_outputs.py", *[str(path.relative_to(PROJECT_ROOT)) for path in piece_start_dirs if path.exists()]]
conditioned_cmd = [
    sys.executable,
    "scripts/split_conditioned_outputs.py",
    *[str(path.relative_to(PROJECT_ROOT)) for path in conditioned_dirs if path.exists()],
    "--evaluation-dir", CONFIG["evaluation_dir"],
]

print("Piece-start split command:")
print(" ".join(piece_cmd) if len(piece_cmd) > 2 else "No piece-start run dirs found yet.")
print("Conditioned split command:")
print(" ".join(conditioned_cmd) if len(conditioned_cmd) > 4 else "No indexed run dirs found yet.")

if RUN_PREFIX_SPLIT:
    if len(piece_cmd) > 2:
        run_command(piece_cmd)
    if len(conditioned_cmd) > 4:
        run_command(conditioned_cmd)
else:
    print("Prefix split not started. Set RUN_PREFIX_SPLIT = True after indexed runs exist.")
            """
        ),
        markdown_cell("## 8. Indexed Run Saving"),
        code_cell(
            r"""
import shutil
from dataclasses import asdict

from src.evaluate import analyze_candidate


RUN_ID = "run_004"  # Change this before saving a new indexed run.
RUN_ID = CONFIG.get("run_id", RUN_ID)
RUN_NOTES = "Listening notes placeholder. Add observations after auditioning."


def save_single_indexed_run(config, run_id, notes):
    source_dir = PROJECT_ROOT / config["output_dir"]
    run_dir = PROJECT_ROOT / config["final_indexed_dir"] / run_id
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
        task_df = ranking_df[
            (ranking_df["task_type"] == task)
            & (ranking_df["valid"] == True)
            & (ranking_df["usable"] == True)
        ].sort_values("score", ascending=False)
        if task_df.empty:
            print(f"No usable Transformer {task} candidate found; need regeneration.")
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
        "decode_retry_attempts": config["decode_retry_attempts"],
        "temperatures": config["temperatures"],
        "top_ks": config["top_ks"],
        "generation_mode": config["generation_mode"],
        "unconditioned_prefix_tokens": config["unconditioned_prefix_tokens"],
        "primer_source": config["primer_source"],
        "primer_index": config["primer_index"],
        "selection_rule": "highest score among usable candidates after hard reject filters",
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
selected_path = PROJECT_ROOT / CONFIG["evaluation_dir"] / "tables" / "selected_candidates.csv"
if selected_path.exists():
    selected_df = pd.read_csv(selected_path)
    display(selected_df[["run", "task_type", "selected_path", "note_count", "duration_seconds", "notes_per_second", "pitch_range"]])
    print("Files to audition:")
    for row in selected_df.itertuples(index=False):
        midi_path = PROJECT_ROOT / getattr(row, "selected_path")
        print(f"{getattr(row, 'run')} {getattr(row, 'task_type')}: {midi_path}")
    run_dir = PROJECT_ROOT / CONFIG["final_indexed_dir"] / CONFIG["run_id"]
    print("Current run helper commands:")
    print(f"Invoke-Item '{run_dir}\\symbolic_unconditioned.mid'")
    print(f"Invoke-Item '{run_dir}\\symbolic_conditioned.mid'")
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
