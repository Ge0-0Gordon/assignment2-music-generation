# Execution Log

This log records environment checks, commands, outcomes, and fallback decisions for the CSE253 Assignment 2 symbolic MIDI generation project.

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
