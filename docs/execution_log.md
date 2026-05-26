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
