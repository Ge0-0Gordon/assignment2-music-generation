# Training Commands

Use PowerShell from the repository root:

```powershell
cd C:\Users\GD\OneDrive\Desktop\CSE253\assignment2-music-generation
```

All project Python commands must run through the `cse253` Conda environment:

```powershell
conda run -n cse253 python ...
```

Do not run project training in `base`. Do not use pretrained GPT-2 weights, GPT-2 text tokenizers, or pretrained music checkpoints.

## Nottingham Main Route

Local data path:

```text
data\nottingham-dataset-master\MIDI
```

This is the recommended final route. Use `--max-files 0` for all discovered MIDI files, or set an integer such as `500` for a bounded run.

### Full Training

5000-step starter run:

```powershell
conda run -n cse253 python scripts\train_main.py --dataset-name nottingham_final --input-dir data\nottingham-dataset-master\MIDI --output-dir outputs\candidates\nottingham_final --metrics-dir outputs\metrics\nottingham_final --max-files 0 --block-size 256 --stride 128 --valid-fraction 0.1 --epochs 100 --max-steps 5000 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --n-embd 256 --n-layer 4 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\nottingham_final --eval-interval 250 --generate-tokens 384 --candidate-count 3 --temperatures '0.7,0.8,0.9,1.0' --top-ks '20,50' --seed 253
```

Longer options: change `--max-steps 5000` to `10000` or `20000`. Keep the same checkpoint/output paths so the best checkpoint and generated candidates stay in the final Nottingham directories.

Batch size guidance: start with `--batch-size 16` on the available GPU. If CUDA memory runs out, rerun with `--batch-size 8`.

### Resume Training

Resume from the current best checkpoint:

```powershell
conda run -n cse253 python scripts\train_main.py --dataset-name nottingham_final --input-dir data\nottingham-dataset-master\MIDI --output-dir outputs\candidates\nottingham_final --metrics-dir outputs\metrics\nottingham_final --max-files 0 --block-size 256 --stride 128 --valid-fraction 0.1 --epochs 100 --max-steps 5000 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --n-embd 256 --n-layer 4 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\nottingham_final --resume-checkpoint outputs\checkpoints\nottingham_final\best_transformer.pt --eval-interval 250 --generate-tokens 384 --candidate-count 3 --temperatures '0.7,0.8,0.9,1.0' --top-ks '20,50' --seed 253
```

### Generate Candidates From Checkpoint

```powershell
conda run -n cse253 python scripts\train_main.py --mode generate --dataset-name nottingham_final --input-dir data\nottingham-dataset-master\MIDI --output-dir outputs\candidates\nottingham_final --metrics-dir outputs\metrics\nottingham_final --max-files 0 --block-size 256 --stride 128 --valid-fraction 0.1 --batch-size 16 --resume-checkpoint outputs\checkpoints\nottingham_final\best_transformer.pt --generate-tokens 384 --candidate-count 5 --temperatures '0.7,0.8,0.9,1.0' --top-ks '20,50' --seed 253
```

### Evaluate And Rank

```powershell
conda run -n cse253 python scripts\analyze_candidates.py --datasets nottingham_final --output-dir outputs\evaluation\nottingham_final --transformer-only-selected
```

To rebuild the combined final evaluation tables for Nottingham and MAESTRO:

```powershell
conda run -n cse253 python scripts\analyze_candidates.py --datasets nottingham_final maestro_final --output-dir outputs\evaluation --transformer-only-selected
```

## MAESTRO MIDI-Only Optional Route

Local sources currently available:

```text
data\maestro-v3.0.0-midi.zip
data\raw\maestro_final\midi
```

Use only MIDI files. Do not use MAESTRO audio. The dataset preparation script reuses the local zip when present and selects short MIDI files using metadata from the archive.

### Prepare Or Refresh A Bounded MIDI Subset

100-file subset:

```powershell
conda run -n cse253 python scripts\prepare_dataset.py --dataset maestro-midi --output-dir data\raw\maestro_final --max-files 100
```

200-file subset:

```powershell
conda run -n cse253 python scripts\prepare_dataset.py --dataset maestro-midi --output-dir data\raw\maestro_final --max-files 200
```

500 files may be feasible, but expect longer tokenization and training:

```powershell
conda run -n cse253 python scripts\prepare_dataset.py --dataset maestro-midi --output-dir data\raw\maestro_final --max-files 500
```

### Subset Training

3000-step starter run:

```powershell
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_final --input-dir data\raw\maestro_final\midi --output-dir outputs\candidates\maestro_final --metrics-dir outputs\metrics\maestro_final --max-files 200 --block-size 128 --stride 128 --valid-fraction 0.2 --epochs 100 --max-steps 3000 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --n-embd 128 --n-layer 4 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\maestro_final --eval-interval 250 --generate-tokens 384 --candidate-count 3 --temperatures '0.7,0.8,0.9,1.0' --top-ks '20,50' --seed 253
```

Longer options: change `--max-steps 3000` to `5000` or `10000`. If memory allows and generation quality is weak, try `--block-size 256 --n-embd 256`.

### Resume Training

```powershell
conda run -n cse253 python scripts\train_main.py --dataset-name maestro_final --input-dir data\raw\maestro_final\midi --output-dir outputs\candidates\maestro_final --metrics-dir outputs\metrics\maestro_final --max-files 200 --block-size 128 --stride 128 --valid-fraction 0.2 --epochs 100 --max-steps 3000 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --n-embd 128 --n-layer 4 --n-head 4 --dropout 0.1 --grad-clip 1.0 --checkpoint-dir outputs\checkpoints\maestro_final --resume-checkpoint outputs\checkpoints\maestro_final\best_transformer.pt --eval-interval 250 --generate-tokens 384 --candidate-count 3 --temperatures '0.7,0.8,0.9,1.0' --top-ks '20,50' --seed 253
```

### Generate Candidates From Checkpoint

```powershell
conda run -n cse253 python scripts\train_main.py --mode generate --dataset-name maestro_final --input-dir data\raw\maestro_final\midi --output-dir outputs\candidates\maestro_final --metrics-dir outputs\metrics\maestro_final --max-files 200 --block-size 128 --stride 128 --valid-fraction 0.2 --batch-size 16 --resume-checkpoint outputs\checkpoints\maestro_final\best_transformer.pt --generate-tokens 384 --candidate-count 5 --temperatures '0.7,0.8,0.9,1.0' --top-ks '20,50' --seed 253
```

### Evaluate And Rank

```powershell
conda run -n cse253 python scripts\analyze_candidates.py --datasets maestro_final --output-dir outputs\evaluation\maestro_final --transformer-only-selected
```

## Healthy Training Checks

- Train loss should generally trend downward over hundreds or thousands of steps.
- Validation loss and perplexity should stay finite. A rising validation loss after many steps can mean overfitting.
- `outputs\checkpoints\<dataset>\best_transformer.pt` should exist after the first validation improvement.
- Candidate MIDI files should be valid, parseable, and nonempty.
- Selected candidates should have a reasonable note count and duration, not just a few notes or an extremely dense burst.
- Perplexity is useful for next-token prediction, but listening quality still needs qualitative review.

## After Long Training

1. Run candidate generation from the best checkpoint if the training command did not generate enough candidates.
2. Run `scripts\analyze_candidates.py` to rebuild ranking tables and selected outputs.
3. Run `conda run -n cse253 python scripts\build_workbook.py` to refresh `notebooks\workbook.ipynb`.
4. Listen to selected candidates and record qualitative notes.
5. Only after review, create final `submission\` files with the required names.
