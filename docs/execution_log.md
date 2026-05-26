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
