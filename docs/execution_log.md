# Execution Log

This log records environment checks, commands, outcomes, and fallback decisions for the CSE253 Assignment 2 symbolic MIDI generation project.

## 2026-05-26: MAESTRO MIDI-Only Full Official-Split Run

### Scope

- Reused the local `data/maestro-v3.0.0-midi.zip`; no audio was downloaded or used.
- Prepared a MAESTRO full MIDI-only manifest from official metadata.
- Trained the existing GPT2-style Transformer pipeline on official train/validation MIDI files with bounded `max_steps`.
- Saved best checkpoint, metrics, flat candidates, indexed candidate runs, and MAESTRO full evaluation outputs.
- Did not create final `submission/` files, did not export `workbook.html`, and did not commit or push.

### Local Data

```text
MAESTRO local MIDI zip: data/maestro-v3.0.0-midi.zip
Prepared MIDI path: data/raw/maestro_full/midi
Prepared manifest: data/raw/maestro_full/manifest.csv
Official metadata rows / MIDI files: 1276
Official split: train 962, validation 137, test 177
Used for training/eval: 1099 files (train + validation)
Skipped: 177 files, reason split_excluded=test
```

### Commands Run

```powershell
pwd
git status --short
conda info --envs
where.exe python
python --version
conda run -n cse253 python --version
conda run -n cse253 where.exe python
conda run -n cse253 python -c "import torch, json; print(json.dumps({'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))"
conda run -n cse253 python -c "import zipfile, csv, json, collections; z=zipfile.ZipFile('data/maestro-v3.0.0-midi.zip'); rows=list(csv.DictReader(__import__('io').TextIOWrapper(z.open('maestro-v3.0.0/maestro-v3.0.0.csv'), encoding='utf-8'))); print(json.dumps({'rows':len(rows),'splits':dict(collections.Counter(r['split'] for r in rows))}, indent=2))"
conda run -n cse253 python -c "import ast, pathlib; files=[pathlib.Path(p) for p in ['scripts/train_main.py','scripts/prepare_maestro_full.py','scripts/build_indexed_candidates.py','scripts/evaluate_maestro_full.py','scripts/build_workbook.py']]; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax_ok', len(files))"
conda run -n cse253 python scripts\prepare_maestro_full.py --zip-path data\maestro-v3.0.0-midi.zip --output-dir data\raw\maestro_full
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_full_verify --input-dir data\raw\maestro_full\midi --manifest-csv data\raw\maestro_full\manifest.csv --output-dir outputs\tmp\maestro_full_verify\candidates --metrics-dir outputs\tmp\maestro_full_verify\metrics --max-files 24 --block-size 64 --stride 64 --valid-fraction 0.2 --epochs 1 --max-steps 8 --batch-size 4 --lr 0.0003 --weight-decay 0.01 --n-embd 64 --n-layer 2 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\tmp\maestro_full_verify\checkpoints --eval-interval 4 --generate-tokens 96 --candidate-count 1 --temperatures 0.8 --top-ks 20 --seed 253
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_full --input-dir data\raw\maestro_full\midi --manifest-csv data\raw\maestro_full\manifest.csv --output-dir outputs\candidates\maestro_full --metrics-dir outputs\metrics\maestro_full --max-files 0 --block-size 256 --stride 256 --valid-fraction 0.2 --epochs 20 --max-steps 512 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --n-embd 256 --n-layer 4 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\maestro_full --eval-interval 64 --generate-tokens 384 --candidate-count 3 --temperatures '0.7,0.8,0.9' --top-ks '20,50' --seed 253
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_full --input-dir data\raw\maestro_full\midi --manifest-csv data\raw\maestro_full\manifest.csv --output-dir outputs\candidates\maestro_full --metrics-dir outputs\metrics\maestro_full --max-files 0 --block-size 256 --stride 256 --valid-fraction 0.2 --epochs 100 --max-steps 3000 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --n-embd 256 --n-layer 4 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\maestro_full --resume-checkpoint outputs\checkpoints\maestro_full\best_transformer.pt --eval-interval 250 --generate-tokens 512 --candidate-count 5 --temperatures '0.7,0.8,0.9,1.0' --top-ks '20,50' --seed 253
conda run -n cse253 python scripts\build_indexed_candidates.py --source-dir outputs\candidates\maestro_full --output-dir outputs\candidates\final\maestro --metrics-summary outputs\metrics\maestro_full\summary.json --dataset-name maestro_full
conda run -n cse253 python scripts\evaluate_maestro_full.py --metrics-dir outputs\metrics\maestro_full --indexed-dir outputs\candidates\final\maestro --output-dir outputs\evaluation\maestro_full --nottingham-summary outputs\metrics\nottingham_final\summary.json --nottingham-selected-dir outputs\candidates\selected\nottingham_final
conda run -n cse253 python scripts\build_workbook.py
```

### MAESTRO Full Metrics

```text
token length min/max/mean: 416 / 53651 / 12621.8007
train/validation windows: 48133 / 5497
vocab_size: 512
model: GPT2LMHeadModel from scratch
params: 3,356,160
block_size: 256
batch_size: 16
n_embd/n_layer/n_head/dropout: 256 / 4 / 4 / 0.1
steps_completed in resume run: 3000
total_steps_including_resume: 3512
train_loss_last: 3.7378041744232178
valid_loss: 3.9950605436813
valid_perplexity: 54.32912980976267
best checkpoint: outputs/checkpoints/maestro_full/best_transformer.pt
```

### Indexed Candidate Runs

```text
run_001: temperature 0.7, top_k 20
  selected unconditioned: outputs/candidates/final/maestro/run_001/symbolic_unconditioned.mid
  selected conditioned: outputs/candidates/final/maestro/run_001/symbolic_conditioned.mid
run_002: temperature 0.8, top_k 50
  selected unconditioned: outputs/candidates/final/maestro/run_002/symbolic_unconditioned.mid
  selected conditioned: outputs/candidates/final/maestro/run_002/symbolic_conditioned.mid
run_003: temperature 0.9, top_k 50
  selected unconditioned: outputs/candidates/final/maestro/run_003/symbolic_unconditioned.mid
  selected conditioned: outputs/candidates/final/maestro/run_003/symbolic_conditioned.mid
```

Each run contains `symbolic_unconditioned.mid`, `symbolic_conditioned.mid`, `candidate_ranking.csv`, `selected_candidates.csv`, `generation_config.json`, and `notes.txt`.

### Evaluation Outputs

```text
outputs/evaluation/maestro_full/tables/dataset_summary.csv
outputs/evaluation/maestro_full/tables/model_metrics.csv
outputs/evaluation/maestro_full/tables/candidate_ranking.csv
outputs/evaluation/maestro_full/tables/selected_candidates.csv
outputs/evaluation/maestro_full/tables/nottingham_vs_maestro_selected.csv
outputs/evaluation/maestro_full/figures/maestro_full_token_lengths.png
outputs/evaluation/maestro_full/figures/maestro_full_pitch_class_histogram.png
```

### Current Comparison

- Nottingham final remains the safer final fallback right now: validation perplexity is much lower (`3.381` vs MAESTRO full `54.329`) and selected unconditioned outputs are less sparse.
- MAESTRO full is now a real official-split experiment with 3,512 total training steps. Conditioned samples improved and are valid, but unconditioned samples remain inconsistent and often sparse/repetitive.
- Recommended next step: listen to the indexed MAESTRO candidates, then optionally resume MAESTRO full from `outputs/checkpoints/maestro_full/best_transformer.pt` for 5000+ additional steps before replacing Nottingham.

## 2026-05-26: Planning Session

### Scope

- Created planning documents only.
- No code files were created or modified.
- No datasets were downloaded.
- No git commits were created.
- No subagents, parallel agents, or git worktrees were used.

### Environment Checks

Commands run from:

```text
C:\Users\GD\OneDrive\Desktop\CSE253\assignment2-music-generation
```

## 2026-05-26: Training Workflow Hardening and Short Verification

### Scope

- Continued from the recovered Nottingham final-scale run.
- Did not create final submission files.
- Did not run a long formal training job.
- Did not download datasets.
- Reused local Nottingham and MAESTRO MIDI data under `data/`.
- Hardened the training script for user-run long jobs.

### Local Data Confirmed

```text
Nottingham full MIDI path: data/nottingham-dataset-master/MIDI
Nottingham top-level MIDI files: 1034
MAESTRO local MIDI zip: data/maestro-v3.0.0-midi.zip
MAESTRO bounded extracted path: data/raw/maestro_final/midi
MAESTRO bounded extracted MIDI files: 120
Zip files present: data/maestro-v3.0.0-midi.zip, data/nottingham-dataset-master.zip
```

### Existing Nottingham Final-Scale Results Confirmed

```text
best checkpoint: outputs/checkpoints/nottingham_final/best_transformer.pt
summary: outputs/metrics/nottingham_final/summary.json
selected unconditioned: outputs/candidates/selected/nottingham_final/unconditioned_transformer.mid
selected conditioned: outputs/candidates/selected/nottingham_final/conditioned_transformer.mid
```

Metrics from `outputs/metrics/nottingham_final/summary.json`:

```text
file_count: 500
train/valid split: 450/50
block_size: 256
n_embd: 256
n_layer: 4
n_head: 4
dropout: 0.1
steps_completed: 3000
params: 3,356,160
valid_loss: 2.9285693168640137
valid_perplexity: 18.70085634877218
train_loss_last: 0.7876027822494507
```

Selected MIDI validation:

```text
outputs/candidates/selected/nottingham_final/unconditioned_transformer.mid: valid, 176 notes, 93.25 seconds
outputs/candidates/selected/nottingham_final/conditioned_transformer.mid: valid, 164 notes, 93.75 seconds
```

### Code and Documentation Changes

- Updated `.gitignore` to ignore local datasets, zip archives, outputs, checkpoints, candidate MIDI, evaluation outputs, Python caches, and model checkpoints.
- Updated `scripts/prepare_dataset.py` so MAESTRO preparation prefers local zip files and uses metadata to select shorter MIDI-only files.
- Updated `scripts/train_main.py` with:
  - `--mode train_generate|generate`
  - `--resume-checkpoint`
  - `--seed`
  - `--weight-decay`
  - `--grad-clip`
  - `--max-files 0` full-dataset mode
  - checkpoint-only candidate generation
  - best checkpoint resume support
- Updated `src/evaluate.py` so selected files named `unconditioned_transformer.mid` are labeled as Transformer outputs.
- Added `docs/training_commands.md` with copy-ready Nottingham and MAESTRO train/resume/generate/evaluate commands.
- Updated `scripts/build_workbook.py` and rebuilt `notebooks/workbook.ipynb` as a draft report.
- Updated `docs/project_plan.md` to reflect the completed training-command workflow.

### Commands Run

```powershell
git status --short
Get-ChildItem -Force -Path data
Get-ChildItem -Path outputs\checkpoints -Recurse -File
Get-ChildItem -Path outputs\evaluation -Recurse -File
Get-ChildItem -Path data\nottingham-dataset-master\MIDI -File -Filter *.mid | Measure-Object
Get-ChildItem -Path data\raw\maestro_final\midi -File -Filter *.midi | Measure-Object
conda run -n cse253 python -c "import ast, pathlib; files=[pathlib.Path('scripts/train_main.py'), pathlib.Path('scripts/prepare_dataset.py'), pathlib.Path('scripts/build_workbook.py'), pathlib.Path('src/evaluate.py')]; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax_ok', len(files))"
conda run -n cse253 python scripts\train_main.py --help
conda run -n cse253 python scripts\train_main.py --dataset-name nottingham_verify --input-dir data\nottingham-dataset-master\MIDI --output-dir outputs\candidates\verify_nottingham --metrics-dir outputs\metrics\verify_nottingham --max-files 24 --block-size 64 --stride 64 --valid-fraction 0.2 --epochs 1 --max-steps 8 --batch-size 4 --lr 0.0003 --weight-decay 0.01 --n-embd 64 --n-layer 2 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\verify_nottingham --eval-interval 4 --generate-tokens 96 --candidate-count 1 --temperatures '0.8' --top-ks '20' --seed 253
conda run -n cse253 python scripts\train_main.py --mode generate --dataset-name nottingham_verify_generate --input-dir data\nottingham-dataset-master\MIDI --output-dir outputs\candidates\verify_nottingham_generate --metrics-dir outputs\metrics\verify_nottingham_generate --max-files 24 --block-size 64 --stride 64 --valid-fraction 0.2 --batch-size 4 --resume-checkpoint outputs\checkpoints\verify_nottingham\best_transformer.pt --generate-tokens 96 --candidate-count 1 --temperatures '0.8' --top-ks '20' --seed 253
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_verify --input-dir data\raw\maestro_final\midi --output-dir outputs\candidates\verify_maestro --metrics-dir outputs\metrics\verify_maestro --max-files 24 --block-size 64 --stride 128 --valid-fraction 0.2 --epochs 1 --max-steps 8 --batch-size 4 --lr 0.0003 --weight-decay 0.01 --n-embd 64 --n-layer 2 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\verify_maestro --eval-interval 4 --generate-tokens 96 --candidate-count 1 --temperatures '0.8' --top-ks '20' --seed 253
conda run -n cse253 python scripts\train_main.py --dataset-name nottingham_verify_resume --input-dir data\nottingham-dataset-master\MIDI --output-dir outputs\candidates\verify_nottingham_resume --metrics-dir outputs\metrics\verify_nottingham_resume --max-files 24 --block-size 64 --stride 64 --valid-fraction 0.2 --epochs 1 --max-steps 2 --batch-size 4 --lr 0.0003 --weight-decay 0.01 --n-embd 64 --n-layer 2 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\verify_nottingham_resume --resume-checkpoint outputs\checkpoints\verify_nottingham\best_transformer.pt --eval-interval 1 --generate-tokens 64 --candidate-count 1 --temperatures '0.8' --top-ks '20' --seed 253
```

### Short Verification Results

Nottingham short training:

```text
files: 24
train/valid: 19/5
steps_completed: 8
train_loss_first: 6.243983745574951
train_loss_last: 6.168376445770264
valid_loss: 6.150714874267578
valid_perplexity: 469.05258057742844
best_checkpoint: outputs/checkpoints/verify_nottingham/best_transformer.pt
transformer candidates: valid unconditioned and conditioned MIDI
```

Nottingham checkpoint-only generation:

```text
checkpoint: outputs/checkpoints/verify_nottingham/best_transformer.pt
valid_loss: 6.150714874267578
valid_perplexity: 469.05258057742844
generated transformer candidates: valid unconditioned and conditioned MIDI
```

Nottingham resume:

```text
resume checkpoint: outputs/checkpoints/verify_nottingham/best_transformer.pt
additional steps: 2
total_steps_including_resume: 10
valid_loss: 6.119503974914551
valid_perplexity: 454.6391261489731
generated transformer candidates: valid unconditioned and conditioned MIDI
```

MAESTRO short training:

```text
files: 24
train/valid: 19/5
steps_completed: 8
train_loss_first: 6.232507228851318
train_loss_last: 6.182765007019043
valid_loss: 6.178953548957562
valid_perplexity: 482.4867932768834
best_checkpoint: outputs/checkpoints/verify_maestro/best_transformer.pt
transformer candidates: valid unconditioned and conditioned MIDI
```

### Constraint Checks

```text
No from_pretrained("gpt2") call.
No GPT-2 text tokenizer use.
No pretrained music-generation checkpoint use.
Training commands use conda run -n cse253 python.
Data, zip archives, model checkpoints, and outputs are ignored by git.
```

Observed results:

- `pwd`: confirmed repository root.
- `git status --short`: failed because `git` is not recognized in the current PowerShell PATH.
- `conda info --envs`: `base` is the active shell environment; `cse253` exists at `C:\Users\GD\.conda\envs\cse253`.
- `where python`: returned no visible path in the command output.
- `python --version`: `Python 3.13.12`, which is the base environment and should not be used for project execution.
- `conda run -n cse253 python --version`: `Python 3.11.15`.

### Current Decisions

- Use an MVP-first route:
  1. Tiny local MIDI or Nottingham-style small dataset smoke test.
  2. Upgrade to MAESTRO small MIDI subset if dataset acquisition and preprocessing are manageable.
  3. Keep Nottingham/tiny symbolic MIDI as final fallback if MAESTRO becomes too heavy.
- Use MidiTok REMI as the primary tokenizer.
- Use a simple custom MIDI event tokenizer if MidiTok REMI fails.
- Use Markov/n-gram as the required baseline.
- Use HuggingFace `GPT2Config` / `GPT2LMHeadModel` only as a scratch-initialized GPT-2-style causal Transformer if available and stable.
- Do not call `from_pretrained("gpt2")`.
- Do not load pretrained GPT-2 weights.
- Do not load pretrained music generation checkpoints.
- Use custom PyTorch Transformer if HuggingFace is unavailable or unstable.
- Use LSTM if Transformer training or generation fails.

### Known Risks

- `git` is currently unavailable from PowerShell PATH.
- The active shell environment is `base`; all project Python commands must use `conda run -n cse253`.
- MAESTRO may be too heavy for the project timeline unless kept to a small MIDI-only subset.
- Generated token streams may decode to invalid MIDI unless sampling and validation are constrained.

## Future Execution Entries

When implementation begins, append entries in this format:

```text
Date:
Phase:
Commands:
Results:
Files changed:
Verification:
Fallback decisions:
Next step:
```

## 2026-05-26: Phase 0 - Project Skeleton and Environment Check

### Scope

- Executed only Phase 0 from `docs/project_plan.md`.
- Confirmed repository root and Git availability.
- Confirmed project Python checks work through `conda run -n cse253 python ...`.
- Checked dependency availability and GPU status.
- Created `.gitignore`.
- Created `scripts/check_env.py` for repeatable environment checks.
- No datasets were downloaded.
- No packages were installed.
- No git commits were created.
- No data loading, tokenization, baseline, model training, generation, evaluation, or notebook writing was implemented.

### Commands Run

```powershell
pwd
git --version
Get-Command git
git status --short
conda info --envs
where python
python --version
conda run -n cse253 python --version
conda run -n cse253 python -c "import torch; print('torch_version', torch.__version__); print('torch_cuda_version', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
conda run -n cse253 python -c "import importlib.util; mods=['torch','miditok','symusic','mido','midiutil','music21','transformers','nbformat','pandas','numpy','matplotlib','tqdm']; [print(m, importlib.util.find_spec(m) is not None) for m in mods]"
Test-Path .gitignore; if (Test-Path .gitignore) { Get-Content .gitignore }
conda run -n cse253 python scripts\check_env.py
```

### Results

- Current directory:

```text
C:\Users\GD\OneDrive\Desktop\CSE253\assignment2-music-generation
```

- Git:

```text
git version 2.54.0.windows.1
git source: C:\Program Files\Git\cmd\git.exe
git status --short: clean output before Phase 0 file edits
```

- Conda:

```text
base is the active shell environment.
cse253 exists at C:\Users\GD\.conda\envs\cse253.
```

- Python:

```text
python --version: Python 3.13.12
conda run -n cse253 python --version: Python 3.11.15
scripts/check_env.py python_executable: C:\Users\GD\.conda\envs\cse253\python.exe
```

- Torch and GPU:

```text
torch_version 2.11.0+cu128
torch_cuda_version 12.8
cuda_available True
cuda_device_count 1
gpu_name NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

### Dependency Status

```text
torch: True
miditok: True
symusic: True
mido: True
midiutil: True
music21: True
transformers: False
nbformat: True
pandas: False
numpy: True
matplotlib: True
tqdm: True
```

### Files Changed

- Created `.gitignore`.
- Created `scripts/check_env.py`.
- Updated `docs/execution_log.md`.

### Environment Risks

- The interactive shell is still using `base`; use only `conda run -n cse253 python ...` for project Python commands.
- `transformers` is not installed in `cse253`; HuggingFace `GPT2Config` / `GPT2LMHeadModel` is unavailable unless the user approves installation later.
- `pandas` is not installed in `cse253`; notebook tables can use plain Python, NumPy, Markdown tables, or the user can later approve installing pandas if needed.

### Verification

- `scripts/check_env.py` ran successfully through:

```powershell
conda run -n cse253 python scripts\check_env.py
```

- The repeatable environment report confirmed `cse253` Python, dependency availability, Torch version, CUDA version, CUDA availability, device count, and GPU name.

### Fallback Decisions

- Because `transformers` is missing, keep the custom PyTorch Transformer fallback active unless the user later approves installing HuggingFace `transformers`.
- Because `pandas` is missing, avoid depending on pandas for Phase 1 unless installation is approved.

### Next Step

- Proceed to Phase 1 only after user approval: tiny dataset smoke test.

## 2026-05-26: Part A - Environment Refresh After User Install

### Scope

- Verified that `transformers` and `pandas` are now importable in `cse253`.
- Verified that a tiny GPT-2-style causal Transformer can be initialized from scratch with `GPT2Config` and `GPT2LMHeadModel`.
- Did not call `from_pretrained("gpt2")`.
- Did not download or load pretrained weights.

### Commands Run

```powershell
conda run -n cse253 python -c "import importlib.util; mods=['transformers','pandas']; [print(m, importlib.util.find_spec(m) is not None) for m in mods]"
conda run -n cse253 python -c "import torch; from transformers import GPT2Config, GPT2LMHeadModel; config=GPT2Config(vocab_size=32,n_positions=16,n_ctx=16,n_embd=16,n_layer=1,n_head=2,bos_token_id=0,eos_token_id=1); model=GPT2LMHeadModel(config); x=torch.randint(0,32,(2,8)); y=model(input_ids=x, labels=x); print('model_class', model.__class__.__name__); print('params', sum(p.numel() for p in model.parameters())); print('loss_finite', torch.isfinite(y.loss).item()); print('logits_shape', tuple(y.logits.shape)); print('pretrained_loaded', False)"
```

### Results

```text
transformers True
pandas True
model_class GPT2LMHeadModel
params 4080
loss_finite True
logits_shape (2, 8, 32)
pretrained_loaded False
```

### Notes

- HuggingFace emitted a harmless loss-type warning and used its default causal language modeling loss.
- The model was created with `GPT2LMHeadModel(config)`, so weights were randomly initialized from the provided config.

## 2026-05-26: Phase 1 - Tiny Dataset Smoke Test

### Scope

- Checked for local MIDI files under `data/` and `docs/`.
- No local MIDI files were found.
- Created three tiny synthetic smoke-test MIDI files under `data/samples/smoke_test/`.
- Implemented the smallest end-to-end smoke-test path:
  - MIDI discovery/creation.
  - MidiTok REMI tokenization.
  - train/validation token windows.
  - tiny trigram Markov baseline.
  - tiny GPT-2-style random model one-step training smoke test.
  - token sampling.
  - MIDI decoding.
  - output MIDI parse and nonzero-note validation.
- Did not download MAESTRO or any external dataset.
- Did not implement full data loading, full training, final generation, evaluation, or notebook construction.

### Commands Run

```powershell
Get-ChildItem -Path data,docs -Recurse -File -Include *.mid,*.midi | Select-Object FullName,Length
conda run -n cse253 python scripts\run_smoke_test.py
```

### Files Created

- `src/data.py`
- `src/tokenizers.py`
- `src/markov.py`
- `scripts/run_smoke_test.py`
- `data/samples/smoke_test/smoke_ascending.mid`
- `data/samples/smoke_test/smoke_arpeggio.mid`
- `data/samples/smoke_test/smoke_minor.mid`
- `outputs/smoke_test/remi_roundtrip_v4.mid`
- `outputs/smoke_test/smoke_unconditioned.mid`
- `outputs/smoke_test/smoke_conditioned.mid`
- `outputs/smoke_test/summary.json`

### Results

```text
midi_source: synthetic_smoke_test
tokenizer_mode: remi
tokenizer_detail: remi_num_velocities_4
vocab_size: 256
sequence_lengths: [33, 33, 33]
window_count: 9
train_window_count: 8
valid_window_count: 1
markov_order: 3
markov_valid_perplexity: 7.5516225432474835
gpt_model_class: GPT2LMHeadModel
gpt_parameter_count: 21472
gpt_block_size: 16
gpt_loss_before: 5.594419956207275
gpt_loss_after: 5.466349124908447
gpt_loss_is_finite: True
gpt_pretrained_loaded: False
unconditioned_midi: outputs/smoke_test/smoke_unconditioned.mid
conditioned_midi: outputs/smoke_test/smoke_conditioned.mid
unconditioned_note_count: 15
conditioned_note_count: 15
```

### Notes

- MidiTok emitted a warning that `vocab_size` equaled the number of base tokens, so tokenizer training was skipped. This did not block REMI tokenization, decoding, or the smoke test.
- HuggingFace emitted the same harmless loss-type warning and used its default causal language modeling loss.
- REMI succeeded, so the fallback tokenizer was implemented but not used in this smoke-test run.

### Verification

- `conda run -n cse253 python scripts\run_smoke_test.py` exited successfully.
- Generated smoke-test MIDI outputs parse successfully and contain nonzero notes.
- Final verification reran the smoke test successfully after files existed locally; the rerun reported `midi_source: local_existing` because the synthetic smoke-test files had already been created.
- `conda run -n cse253 python -m compileall src scripts` hit Windows/OneDrive permission errors while writing `__pycache__` files, so syntax was verified without writing bytecode via:

```powershell
conda run -n cse253 python -c "import ast, pathlib; files=list(pathlib.Path('src').glob('*.py'))+list(pathlib.Path('scripts').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax_ok', len(files))"
```

Result:

```text
syntax_ok 5
```
- Final environment check through `scripts/check_env.py` now reports `transformers: True` and `pandas: True`.

### Current Risks

- The synthetic MIDI files are only smoke-test data and must not be presented as the final dataset.
- REMI training was skipped for the tiny synthetic set because the requested vocab size was not larger than the base vocabulary. This is acceptable for Phase 1 but should be revisited for a real dataset.
- The GPT-2-style model only completed a tiny random-initialized smoke test, not real training.
- `compileall` may fail in this OneDrive-backed workspace when trying to write `__pycache__`; use `ast.parse` or set a separate cache location if bytecode compilation checks are needed.

### Next Step

- Proceed to Phase 2/3 planning for real tokenizer and baseline implementation, or begin a slightly larger local/Nottingham smoke dataset before MAESTRO.

## 2026-05-26: Small Real MIDI Dataset Smoke Run

### Scope

- Started from a clean git working tree.
- Checked for usable non-synthetic local MIDI files under `data/` and `docs/`.
- No usable real local MIDI files were found.
- Researched a small MIDI-only real dataset option.
- Selected a tiny subset from the cleaned Nottingham dataset because its GitHub repository includes a `MIDI/` directory and is released under GPL-3.0.
- Did not download MAESTRO.
- Did not download audio data.
- Downloaded only 12 small Nottingham MIDI files into ignored path `data/raw/nottingham_smoke/`.
- Updated `scripts/run_smoke_test.py` only enough to support explicit real input/output directories and prevent accidental synthetic fallback.
- Ran the REMI tokenizer path, Markov baseline, tiny GPT-2-style random-initialized smoke train step, generation, and MIDI validity checks.

### Dataset Source

```text
Name: cleaned Nottingham dataset
Repository: https://github.com/jukedeck/nottingham-dataset
Format used: MIDI-only files from the repository MIDI directory
License noted by repository: GPL-3.0
Subset used: 12 files named ashover1.mid through ashover12.mid
Local path: data/raw/nottingham_smoke/
```

### Commands Run

```powershell
git status --short
Get-ChildItem -Path data,docs -Recurse -File -Include *.mid,*.midi | Where-Object { $_.FullName -notlike '*data\samples\smoke_test*' } | Select-Object FullName,Length
$out = 'data\raw\nottingham_smoke'; New-Item -ItemType Directory -Force -Path $out | Out-Null; 1..12 | ForEach-Object { $name = "ashover$_.mid"; $url = "https://raw.githubusercontent.com/jukedeck/nottingham-dataset/master/MIDI/$name"; Invoke-WebRequest -Uri $url -OutFile (Join-Path $out $name) }; Get-ChildItem $out | Select-Object Name,Length
conda run -n cse253 python scripts\run_smoke_test.py --input-dir data\raw\nottingham_smoke --output-dir outputs\candidates\nottingham_smoke --max-files 12
```

### Downloaded Files

```text
ashover1.mid   1574 bytes
ashover2.mid   3385 bytes
ashover3.mid   4743 bytes
ashover4.mid   4575 bytes
ashover5.mid   4649 bytes
ashover6.mid   3649 bytes
ashover7.mid   2515 bytes
ashover8.mid   1843 bytes
ashover9.mid   3482 bytes
ashover10.mid  5715 bytes
ashover11.mid  3105 bytes
ashover12.mid  3285 bytes
```

### Smoke-Test Results

```text
midi_source: local_existing
midi_files_used: 12
tokenizer_mode: remi
tokenizer_detail: remi_num_velocities_4
remi_error: empty
vocab_size: 256
sequence_lengths: [296, 1757, 701, 989, 889, 1188, 1445, 1189, 781, 892, 540, 1056]
window_count: 1446
train_window_count: 1445
valid_window_count: 1
markov_order: 3
markov_valid_perplexity: 8.94304813069377
gpt_model_class: GPT2LMHeadModel
gpt_parameter_count: 21472
gpt_block_size: 16
gpt_loss_before: 5.573023319244385
gpt_loss_after: 5.465811252593994
gpt_loss_is_finite: True
gpt_pretrained_loaded: False
unconditioned_midi: outputs/candidates/nottingham_smoke/smoke_unconditioned.mid
conditioned_midi: outputs/candidates/nottingham_smoke/smoke_conditioned.mid
unconditioned_note_count: 15
conditioned_note_count: 15
```

### Verification

- `conda run -n cse253 python scripts\run_smoke_test.py --input-dir data\raw\nottingham_smoke --output-dir outputs\candidates\nottingham_smoke --max-files 12` exited successfully.
- REMI tokenization worked on the real MIDI subset.
- Generated candidate MIDI files parse successfully and contain nonzero notes.
- The tiny GPT-2-style model was initialized from scratch with config and did not load pretrained weights.

### Notes

- The first `Invoke-WebRequest` attempt failed with a connection receive error; rerunning with approved network escalation succeeded.
- MidiTok again warned that `vocab_size` equaled the number of base tokens, so tokenizer training was skipped. This is acceptable for smoke testing but should be revisited for a real tokenizer/data phase.
- The model smoke test is intentionally tiny and does not represent meaningful music training quality.

### Current Risks

- Nottingham is suitable for a small real symbolic smoke test, but the final dataset choice is still open.
- Downloaded raw MIDI files are ignored and should not be committed.
- Candidate MIDI outputs are ignored and should not be submitted as final artifacts.
- Validation currently uses only one window because this smoke script keeps the split simple; real evaluation should use a larger validation split.

### Next Step

- Review this real-data smoke run, then decide whether to expand the Nottingham subset for a stronger baseline/tokenizer phase or proceed toward a small MAESTRO MIDI-only subset.

## 2026-05-26: Expanded Nottingham and Small MAESTRO Pipeline Runs

### Scope

- Expanded the real symbolic MIDI pipeline beyond the 12-file smoke test.
- Kept runs bounded and short.
- Did not download MAESTRO audio.
- Did not run full-scale training.
- Did not create final notebook or submission files.
- Did not create commits or push.
- Did not install packages.

### Code Changes

- Added `src/config.py` for bounded experiment defaults.
- Extended `src/data.py` with deterministic file splitting and pitch-class histogram helpers.
- Updated `src/tokenizers.py` so REMI training requests a vocabulary larger than the base token count.
- Added `scripts/prepare_dataset.py` for bounded MIDI-only dataset preparation.
- Added `scripts/train_main.py` for manifest, tokenization, Markov baseline, short scratch GPT-2-style Transformer training, candidate generation, MIDI validation, and metric summary outputs.

### Commands Run

```powershell
git status --short
conda run -n cse253 python -c "import ast, pathlib; files=list(pathlib.Path('src').glob('*.py'))+list(pathlib.Path('scripts').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax_ok', len(files))"
conda run -n cse253 python scripts\prepare_dataset.py --dataset nottingham --output-dir data\raw\nottingham_subset --max-files 150
conda run -n cse253 python scripts\train_main.py --dataset-name nottingham_subset --input-dir data\raw\nottingham_subset --output-dir outputs\candidates\nottingham_subset --metrics-dir outputs\metrics\nottingham_subset --max-files 150 --block-size 64 --stride 32 --valid-fraction 0.2 --epochs 2 --max-steps 80 --batch-size 16 --lr 0.0003 --generate-tokens 192
conda run -n cse253 python -c "import json; s=json.load(open('outputs/metrics/nottingham_subset/summary.json')); print('remi_ok', s['tokenizer_mode']=='remi'); print('train_windows', s['train_window_count']); print('valid_windows', s['valid_window_count']); print('markov_finite', s['markov']['valid_perplexity'] < float('inf')); print('transformer_losses_finite', s['transformer']['losses_finite']); print('all_outputs_valid', all(v['valid'] and v['note_count'] > 0 for v in s['outputs'].values()))"
Select-String -Path src\*.py,scripts\*.py -Pattern 'from_pretrained|GPT2Tokenizer|AutoTokenizer|pretrained'
git status --short --ignored
conda run -n cse253 python scripts\prepare_dataset.py --dataset maestro-midi --output-dir data\raw\maestro_subset --max-files 40
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_subset --input-dir data\raw\maestro_subset\midi --output-dir outputs\candidates\maestro_subset --metrics-dir outputs\metrics\maestro_subset --max-files 40 --block-size 64 --stride 64 --valid-fraction 0.2 --epochs 1 --max-steps 80 --batch-size 16 --lr 0.0003 --generate-tokens 192
```

### Expanded Nottingham Dataset

```text
Source: https://github.com/jukedeck/nottingham-dataset/tree/master/MIDI
Format: MIDI-only
License noted by repository: GPL-3.0
Local path: data/raw/nottingham_subset/
File count: 150
Total size: 439,141 bytes
Selection: first 150 MIDI files returned by the repository MIDI directory listing, sorted by filename
```

Initial non-escalated download failed due network refusal; approved escalated rerun succeeded.

### Expanded Nottingham Results

```text
train_file_count: 120
valid_file_count: 30
tokenizer_mode: remi
tokenizer_detail: remi_num_velocities_8
vocab_size: 512
token_length_min: 57
token_length_max: 3243
token_length_mean: 277.8666666666667
train_window_count: 761
valid_window_count: 314
markov_order: 3
markov_valid_perplexity: 72.89547291034337
transformer_model_class: GPT2LMHeadModel
transformer_pretrained_loaded: False
transformer_device: cuda
transformer_parameter_count: 136960
transformer_block_size: 64
transformer_batch_size: 16
transformer_steps_completed: 80
transformer_train_loss_first: 6.219630718231201
transformer_train_loss_last: 5.149081707000732
transformer_valid_loss: 5.251197934150696
transformer_valid_perplexity: 190.79469108990622
transformer_losses_finite: True
```

Generated Nottingham candidates:

```text
outputs/candidates/nottingham_subset/markov_unconditioned.mid: valid, 130 notes
outputs/candidates/nottingham_subset/markov_conditioned.mid: valid, 130 notes
outputs/candidates/nottingham_subset/transformer_unconditioned.mid: valid, 48 notes
outputs/candidates/nottingham_subset/transformer_conditioned.mid: valid, 74 notes
```

Evaluation outputs:

```text
outputs/metrics/nottingham_subset/manifest.csv
outputs/metrics/nottingham_subset/dataset_summary.csv
outputs/metrics/nottingham_subset/token_length_distribution.csv
outputs/metrics/nottingham_subset/pitch_class_histogram.csv
outputs/metrics/nottingham_subset/summary.json
```

### Nottingham Self-Review Gate

```text
REMI worked reliably: yes
Train/validation windows sufficient: yes, 761 train and 314 validation windows
Markov metrics finite: yes
Transformer losses finite: yes
Generated MIDI valid and nonempty: yes
Code still reusable: yes, moved bounded run into prepare/train scripts
Downloaded data and outputs ignored by git: yes, data/raw and outputs are ignored
Pretrained GPT-2/music checkpoint use: no from_pretrained calls; no GPT-2 tokenizer use
```

### MAESTRO Small MIDI-Only Dataset

```text
Source: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
Official dataset page: https://magenta.tensorflow.org/datasets/maestro
Format: MIDI-only zip archive
License: CC BY-NC-SA 4.0
Local path: data/raw/maestro_subset/
Archive size: 58,416,533 bytes
Extracted MIDI file count: 40
Extracted MIDI total size: 2,325,603 bytes
Selection/filtering: first 40 MIDI members in sorted archive order, extracted to data/raw/maestro_subset/midi/
```

Initial non-escalated download failed due network refusal; approved escalated rerun succeeded.

### MAESTRO Results

```text
train_file_count: 32
valid_file_count: 8
tokenizer_mode: remi
tokenizer_detail: remi_num_velocities_8
vocab_size: 512
token_length_min: 3087
token_length_max: 37808
token_length_mean: 11963.2
train_window_count: 6033
valid_window_count: 1422
markov_order: 3
markov_valid_perplexity: 296.8909029097737
transformer_model_class: GPT2LMHeadModel
transformer_pretrained_loaded: False
transformer_device: cuda
transformer_parameter_count: 136960
transformer_block_size: 64
transformer_batch_size: 16
transformer_steps_completed: 80
transformer_train_loss_first: 6.244447708129883
transformer_train_loss_last: 5.70381498336792
transformer_valid_loss: 5.679322842801555
transformer_valid_perplexity: 292.75112425693015
transformer_losses_finite: True
```

Generated MAESTRO candidates:

```text
outputs/candidates/maestro_subset/markov_unconditioned.mid: valid, 92 notes
outputs/candidates/maestro_subset/markov_conditioned.mid: valid, 87 notes
outputs/candidates/maestro_subset/transformer_unconditioned.mid: valid, 92 notes
outputs/candidates/maestro_subset/transformer_conditioned.mid: valid, 10 notes
```

Evaluation outputs:

```text
outputs/metrics/maestro_subset/manifest.csv
outputs/metrics/maestro_subset/dataset_summary.csv
outputs/metrics/maestro_subset/token_length_distribution.csv
outputs/metrics/maestro_subset/pitch_class_histogram.csv
outputs/metrics/maestro_subset/summary.json
```

### Notes and Risks

- The Transformer runs are short bounded smoke/training runs, not final-quality training.
- MAESTRO token sequences are much longer than Nottingham, so future real training should tune context/windowing carefully.
- Markov and Transformer perplexities are finite but not directly comparable as musical-quality scores.
- Generated candidates are valid MIDI, but listening quality still needs human review.
- The MAESTRO MIDI-only archive is about 58 MB; it is ignored and should not be committed.
- All downloaded data and generated outputs are under ignored paths.

### Next Step

- Review candidates and metrics, then decide whether to continue with MAESTRO as the final dataset or keep Nottingham as the fallback while improving model/training quality.

### Final Self-Review

Commands:

```powershell
conda run -n cse253 python -c "import ast, pathlib; files=list(pathlib.Path('src').glob('*.py'))+list(pathlib.Path('scripts').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax_ok', len(files))"
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_final_check --input-dir data\raw\maestro_subset\midi --output-dir outputs\tmp\final_check\candidates --metrics-dir outputs\tmp\final_check\metrics --max-files 8 --block-size 32 --stride 128 --valid-fraction 0.25 --epochs 1 --max-steps 5 --batch-size 4 --lr 0.0003 --generate-tokens 64
Select-String -Path src\*.py,scripts\*.py -Pattern 'from_pretrained|GPT2Tokenizer|AutoTokenizer|pretrained|music-generation checkpoint'
git check-ignore data\raw\nottingham_subset\ashover1.mid data\raw\maestro_subset\maestro-v3.0.0-midi.zip outputs\metrics\nottingham_subset\summary.json outputs\candidates\maestro_subset\transformer_unconditioned.mid outputs\tmp\final_check\metrics\summary.json
conda run -n cse253 python -c "from pathlib import Path; from src.data import count_midi_notes; paths=['outputs/candidates/nottingham_subset/markov_unconditioned.mid','outputs/candidates/nottingham_subset/markov_conditioned.mid','outputs/candidates/nottingham_subset/transformer_unconditioned.mid','outputs/candidates/nottingham_subset/transformer_conditioned.mid','outputs/candidates/maestro_subset/markov_unconditioned.mid','outputs/candidates/maestro_subset/markov_conditioned.mid','outputs/candidates/maestro_subset/transformer_unconditioned.mid','outputs/candidates/maestro_subset/transformer_conditioned.mid']; [print(p, Path(p).exists(), count_midi_notes(Path(p))) for p in paths]"
git status --short
```

Results:

```text
syntax_ok: 8 files
final_check: 8 MAESTRO MIDI files, REMI, 682 train windows, 222 valid windows, finite Markov perplexity, 5 finite Transformer steps, all four final-check outputs valid
from_pretrained scan: no from_pretrained calls; only pretrained_loaded=False metadata fields found
ignored path check: data/raw, outputs/metrics, outputs/candidates, and outputs/tmp are ignored
candidate MIDI note checks: all Nottingham and MAESTRO candidate MIDI files exist and contain nonzero notes
```

Self-review findings:

```text
No GPT-2 pretrained weights are loaded.
No pretrained music-generation checkpoints are loaded.
No GPT-2 text tokenizer is used.
The current code has some deliberate overlap between smoke and bounded training scripts, but the reusable path for future work is scripts/train_main.py.
Paths are CLI-configurable for dataset/input/output/metrics directories.
Downloaded datasets and generated outputs are ignored by git.
The current training runs are bounded and proof-of-pipeline, not final model-quality runs.
```

## Candidate Analysis, Evaluation Artifacts, and Draft Notebook

Date: 2026-05-26

Purpose: finish the interrupted draft-report pass without rerunning long training jobs or creating final submission files.

Commands:

```powershell
conda run -n cse253 python -c "import ast, pathlib; files=list(pathlib.Path('src').glob('*.py'))+list(pathlib.Path('scripts').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax_ok', len(files))"
conda run -n cse253 python scripts\analyze_candidates.py --datasets nottingham maestro --output-dir outputs\evaluation
conda run -n cse253 python scripts\build_workbook.py
conda run -n cse253 python -c "import nbformat; nb=nbformat.read('notebooks/workbook.ipynb', as_version=4); print('notebook_cells', len(nb.cells)); print('markdown_cells', sum(1 for c in nb.cells if c.cell_type=='markdown'))"
Select-String -Path src\*.py,scripts\*.py,notebooks\workbook.ipynb -Pattern 'from_pretrained|GPT2Tokenizer|AutoTokenizer'
conda run -n cse253 python -c "from pathlib import Path; from src.evaluate import analyze_candidate; paths=sorted(Path('outputs/candidates/selected').rglob('*.mid')); [print(p.as_posix(), analyze_candidate(p).valid, analyze_candidate(p).note_count) for p in paths]"
```

Initial notebook build issue:

```text
pandas.to_markdown required optional package tabulate, which is not installed in cse253.
No package was installed. scripts/build_workbook.py was changed to render simple Markdown tables directly.
```

Evaluation outputs:

```text
outputs/evaluation/tables/dataset_summary.csv
outputs/evaluation/tables/model_metrics.csv
outputs/evaluation/tables/candidate_ranking.csv
outputs/evaluation/tables/selected_candidates.csv
outputs/evaluation/tables/figures.csv
outputs/evaluation/analysis_summary.json
outputs/evaluation/figures/nottingham_token_lengths.png
outputs/evaluation/figures/nottingham_pitch_class_histogram.png
outputs/evaluation/figures/maestro_token_lengths.png
outputs/evaluation/figures/maestro_pitch_class_histogram.png
```

Selected draft MIDI candidates:

```text
outputs/candidates/selected/nottingham/unconditioned_markov.mid
outputs/candidates/selected/nottingham/conditioned_markov.mid
outputs/candidates/selected/maestro/unconditioned_transformer.mid
outputs/candidates/selected/maestro/conditioned_markov.mid
```

Candidate ranking summary:

```text
Nottingham conditioned best: markov_conditioned.mid, valid, 130 notes, 40.5 seconds, score 0.9137
Nottingham unconditioned best: markov_unconditioned.mid, valid, 130 notes, 50.75 seconds, score 0.9297
MAESTRO unconditioned selected: transformer_unconditioned.mid, valid, 92 notes, 6.875 seconds, score -0.3498
MAESTRO conditioned best: markov_conditioned.mid, valid, 87 notes, 11.875 seconds, score 0.7140
```

Notebook:

```text
notebooks/workbook.ipynb created as a 15-cell draft report.
The notebook includes task definitions, dataset/preprocessing, tokenization, Markov baseline, scratch GPT-2-style Transformer, both generation tasks, evaluation tables, figures, related work notes, limitations, and next steps.
It is not exported to HTML and no submission files were created.
```

Verification:

```text
syntax_ok: 11 Python files
candidate analysis rerun: succeeded and regenerated the same selected-candidate records
notebook_cells: 15
markdown_cells: 15
from_pretrained/GPT2Tokenizer/AutoTokenizer scan: no matches
selected MIDI validation:
  outputs/candidates/selected/maestro/conditioned_markov.mid valid=True notes=87
  outputs/candidates/selected/maestro/unconditioned_transformer.mid valid=True notes=92
  outputs/candidates/selected/nottingham/conditioned_markov.mid valid=True notes=130
  outputs/candidates/selected/nottingham/unconditioned_markov.mid valid=True notes=130
```
