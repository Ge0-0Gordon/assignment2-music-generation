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

MODE_RUN_SETTINGS = [
    ("run_bos", "pure_bos"),
    ("run_seeded", "structural_seeded"),
]

MAESTRO_GENERATION_PRESET = {
    "candidate_count": 10,
    "generate_tokens": 768,
    "temperatures": [0.8, 0.9, 1.0],
    "top_ks": [50, 100],
    "selection_rule": "highest score among usable candidates after hard reject filters",
}


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


def collect_mode_candidates(source_dir: Path, generation_mode: str) -> list[Path]:
    return sorted(
        path
        for path in source_dir.glob("transformer_*.mid*")
        if generation_mode in path.stem or path.stem.startswith("transformer_conditioned_")
    )


def collect_all_candidates(source_dir: Path) -> list[Path]:
    return sorted(source_dir.glob("transformer_*.mid*"))


def best_by_task(rows: list[dict[str, object]], task_type: str) -> dict[str, object] | None:
    task_rows = [
        row
        for row in rows
        if row["task_type"] == task_type and row["valid"] and row.get("usable", False)
    ]
    task_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return task_rows[0] if task_rows else None


def main() -> None:
    parser = ArgumentParser(description="Create outputs/candidates/final/maestro/run_NNN folders.")
    parser.add_argument("--source-dir", default="outputs/candidates/maestro_full")
    parser.add_argument("--output-dir", default="outputs/candidates/final/maestro")
    parser.add_argument("--metrics-summary", default="outputs/metrics/maestro_full/summary.json")
    parser.add_argument("--dataset-name", default="maestro_full")
    parser.add_argument("--include-mode-runs", action="store_true")
    parser.add_argument("--single-run-name", default=None)
    parser.add_argument("--generation-mode-filter", default=None)
    args = parser.parse_args()

    source_dir = resolve(args.source_dir)
    output_dir = resolve(args.output_dir)
    metrics_summary = resolve(args.metrics_summary)
    summary = json.loads(metrics_summary.read_text(encoding="utf-8")) if metrics_summary.exists() else {}

    all_selected: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    if args.single_run_name:
        run_specs: list[dict[str, object]] = [
            {
                "run_name": args.single_run_name,
                "temperature": None,
                "top_k": None,
                "generation_mode": args.generation_mode_filter,
            }
        ]
    else:
        run_specs = [
            {"run_name": run_name, "temperature": temperature, "top_k": top_k, "generation_mode": None}
            for run_name, temperature, top_k in RUN_SETTINGS
        ]
        if args.include_mode_runs:
            run_specs.extend(
                {"run_name": run_name, "temperature": None, "top_k": None, "generation_mode": generation_mode}
                for run_name, generation_mode in MODE_RUN_SETTINGS
            )

    for spec in run_specs:
        run_name = str(spec["run_name"])
        temperature = spec["temperature"]
        top_k = spec["top_k"]
        generation_mode = spec["generation_mode"]
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if generation_mode:
            candidates = collect_mode_candidates(source_dir, str(generation_mode))
        elif args.single_run_name:
            candidates = collect_all_candidates(source_dir)
        else:
            candidates = collect_candidates(source_dir, float(temperature), int(top_k))
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
            "usable",
            "reject_reason",
            "score",
            "source_path",
        ]
        write_csv(run_dir / "candidate_ranking.csv", ranking_rows, fieldnames)

        selected_rows: list[dict[str, object]] = []
        usable_by_task = {
            task_type: sum(
                1
                for row in ranking_rows
                if row["task_type"] == task_type and row["valid"] and row.get("usable", False)
            )
            for task_type in ("unconditioned", "conditioned")
        }
        for task_type, target_name in {
            "unconditioned": "symbolic_unconditioned.mid",
            "conditioned": "symbolic_conditioned.mid",
        }.items():
            best = best_by_task(ranking_rows, task_type)
            if best is None:
                selected_rows.append(
                    {
                        "run": run_name,
                        "dataset": args.dataset_name,
                        "task_type": task_type,
                        "selected_path": "",
                        "source_path": "",
                        "score": "",
                        "note_count": "",
                        "duration_seconds": "",
                        "notes_per_second": "",
                        "pitch_range": "",
                        "max_simultaneous_notes": "",
                        "repeated_pitch_bigram_rate": "",
                        "reject_reason": "No usable Transformer candidate found; need regeneration.",
                    }
                )
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
                "reject_reason": "",
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
                "reject_reason",
            ],
        )
        generation = summary.get("generation", {})
        config = {
            "run": run_name,
            "dataset": args.dataset_name,
            "source_dir": str(source_dir.relative_to(ROOT)),
            "temperature": temperature,
            "top_k": top_k,
            "generation_mode_filter": generation_mode,
            "candidate_files_considered": len(candidates),
            "usable_candidates_by_task": usable_by_task,
            "generation_preset_recommendation": MAESTRO_GENERATION_PRESET,
            "selection_rule": MAESTRO_GENERATION_PRESET["selection_rule"],
            "metrics_summary": str(metrics_summary.relative_to(ROOT)) if metrics_summary.exists() else None,
            "checkpoint": summary.get("transformer", {}).get("best_checkpoint"),
            "generate_tokens": summary.get("generation", {}).get("generate_tokens"),
            "generation": {
                "generate_tokens": generation.get("generate_tokens"),
                "temperatures": generation.get("temperatures"),
                "top_ks": generation.get("top_ks"),
                "candidate_count_per_setting": generation.get("candidate_count_per_setting"),
                "decode_retry_attempts": generation.get("decode_retry_attempts"),
                "unconditioned_mode": generation.get("unconditioned_mode"),
                "unconditioned_prefix_tokens": generation.get("unconditioned_prefix_tokens"),
                "requested_unconditioned_prefix_tokens": generation.get("requested_unconditioned_prefix_tokens"),
                "unconditioned_continuation_tokens": generation.get("unconditioned_continuation_tokens"),
                "primer_source": generation.get("primer_source"),
                "primer_index": generation.get("primer_index"),
                "unconditioned_primer": generation.get("unconditioned_primer"),
            },
        }
        (run_dir / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        notes = [
            f"{run_name}: temperature={temperature}, top_k={top_k}, generation_mode={generation_mode or 'mixed_legacy'}",
            "Candidates were generated by the existing GPT2-style Transformer pipeline.",
            "Selected files are copied from usable candidates only; sparse, too short, too dense, or invalid MIDI is rejected.",
            f"Usable candidates: unconditioned={usable_by_task['unconditioned']}, conditioned={usable_by_task['conditioned']}.",
            "Listening notes placeholder: add qualitative observations after auditioning.",
        ]
        if len(selected_rows) < 2:
            notes.append("Warning: one or more task types did not have a selected row.")
        if any(row.get("reject_reason") for row in selected_rows):
            notes.append("No usable Transformer candidate found for one or more task types; need regeneration.")
        (run_dir / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
        run_summaries.append(config | {"selected": selected_rows})

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "indexed_runs_summary.json"
    if args.single_run_name and summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_runs = [
            row for row in existing_summary.get("runs", []) if row.get("run") != args.single_run_name
        ]
        run_summaries = existing_runs + run_summaries
        all_selected = [
            selected
            for run_summary in run_summaries
            for selected in run_summary.get("selected", [])
            if selected.get("selected_path")
        ]
    (output_dir / "indexed_runs_summary.json").write_text(
        json.dumps({"runs": run_summaries, "selected": all_selected}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir.relative_to(ROOT)), "selected": all_selected}, indent=2))


if __name__ == "__main__":
    main()
