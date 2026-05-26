# Symbolic MIDI Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline plan execution with review checkpoints. Do not use subagents, parallel agents, git worktrees, dataset downloads, or git commits unless the user later gives explicit approval. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Assignment 2 pipeline for symbolic unconditioned MIDI generation and symbolic prefix-conditioned MIDI continuation.

**Architecture:** Use one shared symbolic next-token pipeline: MIDI files are tokenized into REMI or fallback event tokens, split into train/validation windows, modeled with a Markov/n-gram baseline and a scratch-trained GPT-2-style causal Transformer, then sampled into the two required MIDI deliverables. The project proceeds MVP first, Transformer second, polish third, with Nottingham/tiny MIDI as the smoke-test/fallback dataset and MAESTRO small MIDI subset as the preferred final dataset if feasible.

**Tech Stack:** Conda environment `cse253`, Python 3.11, PyTorch, MidiTok REMI, symusic/mido/midiutil as available, HuggingFace `transformers` if available for `GPT2Config` and `GPT2LMHeadModel`, custom PyTorch Transformer and LSTM fallbacks.

---

## Non-Negotiable Constraints

- [ ] Work only inside this repository.
- [ ] Run Python through `conda run -n cse253 python ...`.
- [ ] Do not use the `base` environment for project execution.
- [ ] Do not load pretrained GPT-2 weights.
- [ ] Do not call `from_pretrained("gpt2")`.
- [ ] Do not load pretrained music-generation checkpoints.
- [ ] Train final model weights on the selected MIDI token dataset.
- [ ] Use a MIDI/REMI tokenizer, not the GPT-2 text tokenizer.
- [ ] Produce these final files:
  - `submission/workbook.html`
  - `submission/video_url.txt`
  - `submission/symbolic_unconditioned.mid`
  - `submission/symbolic_conditioned.mid`

## Proposed File Structure

- `docs/project_plan.md`: this implementation plan.
- `docs/execution_log.md`: chronological execution notes, environment checks, commands run, results, and fallback decisions.
- `src/config.py`: shared paths, seeds, dataset limits, tokenizer/model hyperparameters.
- `src/data.py`: MIDI discovery, dataset split, token window creation.
- `src/tokenizers.py`: MidiTok REMI wrapper and custom fallback event tokenizer.
- `src/markov.py`: n-gram baseline training, perplexity, sampling.
- `src/models/transformer_hf.py`: scratch-initialized HuggingFace GPT-2-style causal LM wrapper.
- `src/models/transformer_torch.py`: custom PyTorch causal Transformer fallback.
- `src/models/lstm.py`: LSTM language-model fallback.
- `src/train.py`: training loop, validation loss/perplexity, checkpoint writing.
- `src/generate.py`: unconditioned and prefix-conditioned sampling plus MIDI decoding.
- `src/evaluate.py`: metrics, tables, and figure generation.
- `scripts/check_env.py`: environment and dependency sanity report.
- `scripts/run_smoke_test.py`: tiny end-to-end pipeline runner.
- `scripts/train_main.py`: main training entry point.
- `scripts/generate_submission.py`: final MIDI generation entry point.
- `scripts/export_notebook.py`: notebook-to-HTML export helper if needed.
- `notebooks/workbook.ipynb`: final documented notebook for peer grading.
- `outputs/`: generated intermediate metrics, plots, logs, model checkpoints, and candidate MIDI files.
- `submission/`: final files copied or exported for Gradescope.

## Phase 0: Project Skeleton and Environment Check

**Purpose:** Confirm the repository and `cse253` environment are usable before implementation.

- [ ] Run:

```powershell
pwd
git status --short
conda info --envs
where python
python --version
conda run -n cse253 python --version
conda run -n cse253 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
conda run -n cse253 python -c "import importlib.util; mods=['torch','miditok','symusic','mido','midiutil','music21','transformers','nbformat']; [print(m, importlib.util.find_spec(m) is not None) for m in mods]"
```

- [ ] Record results in `docs/execution_log.md`.
- [ ] If `git` is not found, record the issue and continue without git operations until PATH is fixed by the user.
- [ ] If `transformers` is missing, keep HuggingFace as optional and plan to use the custom PyTorch Transformer unless the user approves a minimal `conda run -n cse253 pip install transformers`.
- [ ] If MidiTok or symusic is missing, plan to use the custom tokenizer or ask the user before installing packages.

**Verification:** The log clearly states current directory, Python executable situation, `cse253` Python version, CUDA availability, and dependency availability.

## Phase 1: Tiny Dataset Smoke Test

**Purpose:** Prove the complete pipeline before using MAESTRO.

- [ ] Select a tiny local MIDI set if files are already present under `data/`.
- [ ] If no MIDI files are present, use a tiny Nottingham-style MIDI subset only after the user approves dataset acquisition.
- [ ] Limit smoke-test data to enough files to exercise parsing, splitting, training, generation, and decoding.
- [ ] Use deterministic seeds for dataset split and sampling.
- [ ] Create a train/validation split with at least one validation sequence.
- [ ] Keep sequence windows short enough for fast CPU/GPU execution.

**Verification:** A smoke-test run can load MIDI files, create token windows, train a minimal baseline/model briefly, generate tokens, decode a MIDI file, and parse the decoded MIDI without error.

## Phase 2: Tokenization

**Primary: MidiTok REMI**

- [ ] Configure REMI for symbolic piano-style MIDI.
- [ ] Keep the first implementation simple: no chord tokens, no complex multi-program conditioning, and a small vocabulary suitable for the dataset.
- [ ] Save tokenizer metadata under `outputs/tokenizer/` during execution.
- [ ] Log token counts per file, vocabulary size, and example tokens in the notebook.

**Fallback: Custom MIDI Event Tokenizer**

- [ ] Quantize note onsets and durations to a fixed grid.
- [ ] Represent music with simple events such as `BAR`, `POSITION`, `PITCH`, `DURATION`, and optional `VELOCITY_BIN`.
- [ ] Provide deterministic encode/decode functions.
- [ ] Reject or repair impossible decoded notes by clipping pitch, duration, and velocity into valid MIDI ranges.

**Fallback trigger:** Use the custom tokenizer if MidiTok cannot tokenize/decode the smoke-test files reliably within one focused debugging pass.

**Verification:** Tokenizer round-trip succeeds for at least one MIDI file: MIDI -> tokens -> MIDI -> parseable MIDI with nonzero notes.

## Phase 3: Markov / N-Gram Baseline

**Purpose:** Provide a reliable baseline for generation and evaluation.

- [ ] Implement unigram, bigram, and trigram next-token models.
- [ ] Add smoothing or explicit unknown/fallback behavior for unseen contexts.
- [ ] Compute validation negative log likelihood and perplexity.
- [ ] Sample unconditioned token sequences.
- [ ] Sample continuation token sequences from a prefix.
- [ ] Decode both sample types to MIDI.

**Verification:** Markov baseline produces at least one valid `symbolic_unconditioned.mid` candidate and one valid `symbolic_conditioned.mid` candidate.

## Phase 4: HuggingFace GPT-2-Style Causal Transformer From Scratch

**Purpose:** Train the preferred main model using a GPT-2-style causal Transformer architecture without pretrained weights.

- [ ] Use `GPT2Config` with custom values such as small vocabulary size, short context length, small embedding dimension, few layers, and few attention heads.
- [ ] Instantiate `GPT2LMHeadModel(config)` only.
- [ ] Do not call any `from_pretrained(...)` method.
- [ ] Train with next-token cross entropy on MIDI token windows.
- [ ] Track train loss, validation loss, validation perplexity, and runtime.
- [ ] Save checkpoints under `outputs/checkpoints/`.
- [ ] Generate both unconditioned and prefix-conditioned samples with temperature and top-k controls.

**Fallback trigger:** Move to Phase 5 if `transformers` is unavailable, HuggingFace APIs are incompatible, or scratch model training/generation cannot run after one focused compatibility pass.

**Verification:** The model trains for a small number of batches, validation perplexity is finite, generation returns token IDs in vocabulary, and at least one generated sequence decodes to valid MIDI.

## Phase 5: Custom PyTorch Causal Transformer Fallback

**Purpose:** Preserve the Transformer project route if HuggingFace is unstable.

- [ ] Implement a compact decoder-only Transformer using PyTorch modules.
- [ ] Use token embeddings, position embeddings, causal self-attention through PyTorch Transformer blocks or an equivalent causal mask, layer norm, and a linear vocabulary head.
- [ ] Keep hyperparameters small enough for fast training.
- [ ] Reuse the same dataset windows, training loop, and generation code shape where possible.
- [ ] Report this as the main Transformer if HuggingFace is not used.

**Fallback trigger:** Move to Phase 6 if Transformer loss diverges repeatedly, training is too slow for the available hardware, or generated MIDI remains invalid after sampling constraints.

**Verification:** Same as Phase 4: finite validation perplexity and at least one valid decoded unconditioned and conditioned sample.

## Phase 6: LSTM Fallback

**Purpose:** Guarantee a trainable neural next-token model if Transformer training fails.

- [ ] Implement a small embedding + LSTM + linear head language model.
- [ ] Train with the same token windows and next-token objective.
- [ ] Generate unconditioned and prefix-conditioned samples.
- [ ] Compare LSTM perplexity and generated-token statistics against the Markov baseline.
- [ ] Explain in the notebook why LSTM became the final neural model if the Transformer route failed.

**Verification:** LSTM produces valid decoded MIDI for both required tasks or Markov baseline remains the guaranteed final artifact generator.

## Phase 7: Upgrade Dataset to MAESTRO Small MIDI Subset

**Purpose:** Move from proof-of-pipeline to the preferred final dataset.

- [ ] Use only MAESTRO MIDI files, not audio.
- [ ] Start with a small number of MIDI files.
- [ ] Filter files that fail parsing, have extreme token lengths, or create unusable decode results.
- [ ] Truncate or window long token sequences rather than training on full pieces.
- [ ] Re-run tokenizer statistics, train/validation split, baseline evaluation, and neural training.

**Fallback trigger:** Stay with Nottingham/tiny symbolic MIDI if MAESTRO download, parsing, token length, or runtime blocks progress.

**Verification:** The selected final dataset has documented source, filtering criteria, token statistics, and successful valid MIDI generation.

## Phase 8: Generation for Required Tasks

**Task 1: symbolic unconditioned generation**

- [ ] Generate from a beginning-of-sequence token, bar token, or short seed.
- [ ] Use controlled sampling: temperature, top-k, maximum token count, and retry limit.
- [ ] Decode candidates until one valid MIDI is found.
- [ ] Save final file as `submission/symbolic_unconditioned.mid`.

**Task 2: symbolic prefix-conditioned continuation**

- [ ] Select a real validation MIDI excerpt as the prefix.
- [ ] Encode the prefix with the same tokenizer.
- [ ] Generate a continuation using the trained next-token model.
- [ ] Decode prefix plus continuation or continuation-only according to whichever makes the final MIDI most understandable.
- [ ] Save final file as `submission/symbolic_conditioned.mid`.

**Verification:** Both files exist, parse as MIDI, contain nonzero notes, and are small enough for submission.

## Phase 9: Evaluation Tables and Figures

**Purpose:** Satisfy the rubric with evidence beyond “it generated something.”

- [ ] Dataset table: number of MIDI files, train/validation split, token count, window count, vocabulary size.
- [ ] Model table: Markov order, Transformer/LSTM hyperparameters, train loss, validation loss, validation perplexity.
- [ ] Generation validity table: attempts, valid decodes, selected output duration, note count.
- [ ] Pitch-class histogram: train vs generated.
- [ ] Duration/rhythm histogram: train vs generated.
- [ ] Token diversity metrics: unique tokens, unique n-grams, repetition rate.
- [ ] Conditioned continuation analysis: prefix length, generated length, boundary behavior, qualitative listening notes.
- [ ] Discussion of why perplexity does not fully measure musical quality.

**Verification:** Notebook contains at least one table or figure for data, modeling, evaluation, and generated-output analysis.

## Phase 10: Notebook Construction

**Purpose:** Produce a clean peer-gradable workbook that does not require execution.

- [ ] Build `notebooks/workbook.ipynb` as a narrative report.
- [ ] Include the two task definitions near the top.
- [ ] Include dataset source and preprocessing decisions.
- [ ] Include tokenization examples.
- [ ] Include Markov baseline implementation summary and results.
- [ ] Include Transformer architecture summary and training results.
- [ ] Include fallback decisions if any fallback is used.
- [ ] Include Task 1 generated MIDI discussion.
- [ ] Include Task 2 prefix and continuation discussion.
- [ ] Include related work: Module 3, REMI/Pop Music Transformer, Music Transformer, Performance RNN, Markov/n-gram models, MAESTRO/Nottingham.
- [ ] Include final submission file list.

**Verification:** A reader can understand the full pipeline and results without running the notebook.

## Phase 11: HTML Export and Final Submission Check

**Purpose:** Produce files matching the assignment autograder expectations.

- [ ] Export notebook to `submission/workbook.html`.
- [ ] Confirm `submission/workbook.html` starts with an HTML doctype or valid notebook HTML header accepted by the assignment checker.
- [ ] Create `submission/video_url.txt` with exactly one public Drive or YouTube URL line after the video exists.
- [ ] Confirm these files exist:
  - `submission/workbook.html`
  - `submission/video_url.txt`
  - `submission/symbolic_unconditioned.mid`
  - `submission/symbolic_conditioned.mid`
- [ ] Confirm the two MIDI files are parseable and nonempty.
- [ ] Confirm no large checkpoints, datasets, caches, or secrets are included in the final submission bundle.

**Verification:** Final submission folder contains exactly the required files plus any explicitly approved extras.

## Phase 12: Risk Register and Fallback Triggers

| Risk | Trigger | Action |
| --- | --- | --- |
| `git` unavailable | `git status` fails because command is missing | Record in log; do not perform git operations until user fixes PATH |
| Wrong Python environment | `python --version` points to base or Python 3.13 | Use only `conda run -n cse253 python ...` |
| Missing HuggingFace `transformers` | Import check returns false | Use custom PyTorch Transformer or ask user before installing |
| MidiTok REMI API incompatibility | Tokenization/decode fails on smoke-test MIDI after one focused pass | Switch to custom MIDI event tokenizer |
| MAESTRO too heavy | Download, parsing, token lengths, or runtime prevents progress | Use Nottingham/tiny symbolic MIDI as final dataset |
| Token sequences too long | Context windows exceed memory or training slows sharply | Shorten context length, window sequences, filter extreme files |
| Transformer training unstable | Loss diverges, NaNs appear, or validation perplexity is unusable | Lower learning rate/model size; then use custom Transformer or LSTM |
| Poor generation quality | Valid MIDI is repetitive/noisy despite finite loss | Adjust temperature/top-k, generate multiple candidates, select best, explain limitations |
| Invalid MIDI decoding | Generated tokens cannot decode or parse | Retry with constrained sampling; if needed use Markov/LSTM valid output |
| Notebook too thin for rubric | Missing tables/plots/discussion sections | Add evaluation tables, figures, related work, and qualitative listening notes |

## Definition of Done

- [ ] The final route is documented in the notebook.
- [ ] At least one baseline and one neural model attempt are documented.
- [ ] The final dataset choice and fallback decisions are explained.
- [ ] Both required symbolic MIDI files exist and parse successfully.
- [ ] Evaluation includes quantitative metrics, visual summaries, and qualitative discussion.
- [ ] `submission/workbook.html` and `submission/video_url.txt` exist.
- [ ] The execution log records commands, outcomes, and any fallback decisions.
