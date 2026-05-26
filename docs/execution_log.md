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
