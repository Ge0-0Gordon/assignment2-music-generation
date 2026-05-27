"""Create indexed final-candidate run folders from flat Transformer outputs."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import analyze_candidate  # noqa: E402


RUN_SETTINGS = [
    ("run_001", 0.7, 20),
    ("run_002", 0.8, 50),
    ("run_003", 0.9, 50),
]


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


def format_temperature(value: float) -> str:
    return str(value).replace(".", "p")


def collect_candidates(source_dir: Path, temperature: float, top_k: int) -> list[Path]:
    temp_tag = f"temp{format_temperature(temperature)}"
    topk_tag = f"topk{top_k}"
    return sorted(
        path
        for path in source_dir.glob("transformer_*.mid*")
        if temp_tag in path.stem and topk_tag in path.stem
    )


def best_by_task(rows: list[dict[str, object]], task_type: str) -> dict[str, object] | None:
    task_rows = [row for row in rows if row["task_type"] == task_type and row["valid"]]
    task_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return task_rows[0] if task_rows else None


def main() -> None:
    parser = ArgumentParser(description="Create outputs/candidates/final/maestro/run_NNN folders.")
    parser.add_argument("--source-dir", default="outputs/candidates/maestro_full")
    parser.add_argument("--output-dir", default="outputs/candidates/final/maestro")
    parser.add_argument("--metrics-summary", default="outputs/metrics/maestro_full/summary.json")
    parser.add_argument("--dataset-name", default="maestro_full")
    args = parser.parse_args()

    source_dir = resolve(args.source_dir)
    output_dir = resolve(args.output_dir)
    metrics_summary = resolve(args.metrics_summary)
    summary = json.loads(metrics_summary.read_text(encoding="utf-8")) if metrics_summary.exists() else {}

    all_selected: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for run_name, temperature, top_k in RUN_SETTINGS:
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        candidates = collect_candidates(source_dir, temperature, top_k)
        ranking_rows: list[dict[str, object]] = []
        for candidate in candidates:
            row = asdict(analyze_candidate(candidate))
            row["dataset"] = args.dataset_name
            row["source_path"] = str(candidate.relative_to(ROOT))
            ranking_rows.append(row)
        ranking_rows.sort(key=lambda row: (row["task_type"], -float(row["score"])))

        fieldnames = list(ranking_rows[0].keys()) if ranking_rows else [
            "path",
            "dataset",
            "task_type",
            "model_type",
            "temperature",
            "top_k",
            "candidate_index",
            "valid",
            "note_count",
            "duration_seconds",
            "notes_per_second",
            "pitch_min",
            "pitch_max",
            "pitch_range",
            "unique_pitch_count",
            "max_simultaneous_notes",
            "repeated_pitch_bigram_rate",
            "score",
            "source_path",
        ]
        write_csv(run_dir / "candidate_ranking.csv", ranking_rows, fieldnames)

        selected_rows: list[dict[str, object]] = []
        for task_type, target_name in {
            "unconditioned": "symbolic_unconditioned.mid",
            "conditioned": "symbolic_conditioned.mid",
        }.items():
            best = best_by_task(ranking_rows, task_type)
            if best is None:
                continue
            source = Path(str(best["path"]))
            if not source.is_absolute():
                source = ROOT / source
            target = run_dir / target_name
            shutil.copy2(source, target)
            selected = {
                "run": run_name,
                "dataset": args.dataset_name,
                "task_type": task_type,
                "selected_path": str(target.relative_to(ROOT)),
                "source_path": str(source.relative_to(ROOT)),
                "score": best["score"],
                "note_count": best["note_count"],
                "duration_seconds": best["duration_seconds"],
                "notes_per_second": best["notes_per_second"],
                "pitch_range": best["pitch_range"],
                "max_simultaneous_notes": best["max_simultaneous_notes"],
                "repeated_pitch_bigram_rate": best["repeated_pitch_bigram_rate"],
            }
            selected_rows.append(selected)
            all_selected.append(selected)

        write_csv(
            run_dir / "selected_candidates.csv",
            selected_rows,
            [
                "run",
                "dataset",
                "task_type",
                "selected_path",
                "source_path",
                "score",
                "note_count",
                "duration_seconds",
                "notes_per_second",
                "pitch_range",
                "max_simultaneous_notes",
                "repeated_pitch_bigram_rate",
            ],
        )
        config = {
            "run": run_name,
            "dataset": args.dataset_name,
            "source_dir": str(source_dir.relative_to(ROOT)),
            "temperature": temperature,
            "top_k": top_k,
            "candidate_files_considered": len(candidates),
            "selection_rule": "highest valid heuristic score per task_type",
            "metrics_summary": str(metrics_summary.relative_to(ROOT)) if metrics_summary.exists() else None,
            "checkpoint": summary.get("transformer", {}).get("best_checkpoint"),
            "generate_tokens": summary.get("generation", {}).get("generate_tokens"),
        }
        (run_dir / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        notes = [
            f"{run_name}: temperature={temperature}, top_k={top_k}",
            "Candidates were generated by the existing GPT2-style Transformer pipeline.",
            "Selected files are copied from the highest valid heuristic score for each task.",
            "Listening notes placeholder: add qualitative observations after auditioning.",
        ]
        if len(selected_rows) < 2:
            notes.append("Warning: one or more task types did not have a valid selected MIDI.")
        (run_dir / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
        run_summaries.append(config | {"selected": selected_rows})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "indexed_runs_summary.json").write_text(
        json.dumps({"runs": run_summaries, "selected": all_selected}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir.relative_to(ROOT)), "selected": all_selected}, indent=2))


if __name__ == "__main__":
    main()
