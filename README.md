# Multimodal Deep Learning Approaches for Human Activity Recognition Using Wearable Sensor Data

Code and precomputed results for the master's thesis of the same name by
Ahmed Mohamady, supervised by Prof. Dr. Kristof Van Laerhoven and Robin
Burchard, M.Sc., at the
[Ubiquitous Computing group](https://ubicomp.eti.uni-siegen.de/),
University of Siegen.

Seven multimodal fusion strategies are compared for
human activity recognition on the
[HARMES dataset](https://zenodo.org/records/19425719) — dual-wrist IMU,
ambient audio and humidity from 20 participants over 16 activity classes
including a Null class — under a leakage-free, participant-grouped protocol.

This repository extends the accompanying
[conference paper repository](https://github.com/AhmedMohamady98/A-Comparison-of-Fusion-Techniques-for-Multi-Modal-Human-Activity-Recognition-on-the-HARMES-Dataset)
with the analyses that appear only in the thesis: gate-activation
interpretability, the modality ablation, calibration, top-k accuracy,
per-participant consistency, hardest-window analysis and the model ensemble.

## Results

Three-fold group cross-validation, macro F1. Every participant appears in the
test set of exactly one fold.

| Model | Modality | Macro F1 |
|---|---|---|
| TinyHAR | IMU | 0.696 |
| AST | Audio | 0.734 |
| TSMixer | Humidity | 0.088 |
| **GMF** | **fusion** | **0.827** |
| Late Fusion | fusion | 0.817 |
| CMA | fusion | 0.795 |
| CLS Transformer | fusion | 0.793 |
| MBT | fusion | 0.787 |
| LMF | fusion | 0.747 |
| Decision Fusion | fusion | 0.747 |

GMF under leave-one-participant-out: **0.819** macro F1, against the 0.760
multi-modal baseline reported with the dataset.

Pairwise testing over the 20 per-participant scores (Wilcoxon signed-rank,
Holm-corrected) separates the methods into three tiers rather than a strict
order. All 16 cross-tier comparisons are significant; none of the 5
within-tier comparisons are. Tier 1 is GMF and Late Fusion, tier 2 is CMA,
CLS Transformer and MBT, tier 3 is LMF and Decision Fusion.

## Does humidity contribute?

No, in aggregate. Retraining the three strongest fusion methods on IMU and
audio alone changes mean macro F1 by **+0.0002**. Reading GMF's gates directly
from the trained checkpoints shows why the null result is not simply the model
ignoring the stream: the humidity gate opens **wider on activities involving
liquid water in all 20 participants**, so the encoder has learned a physically
correct association, but those are precisely the activities where removing
humidity costs nothing, because running water is acoustically distinctive and
already separated by audio. The signal is real and redundant.

## Layout

```
src/        encoders (TinyHAR, AST, TSMixer), the seven fusion modules,
            dataset and participant-grouped splits, training loop, plot style
jobs/       SLURM scripts for training and evaluation on a GPU cluster
notebooks/  analysis and figure generation, paired .py via jupytext
results/    precomputed metrics, per-window predictions and gate activations
tests/      shape and overfitting sanity checks
```

## Reproducing

```bash
pip install -r requirements.txt
export HARMES_DATA=/path/to/HARMES-preprocessed
```

Every figure and table can be regenerated from `results/` without touching a
GPU:

```bash
PYTHONPATH=. python notebooks/02_fusion_overview.py
```

Retraining needs the dataset and a GPU. Each `jobs/train_*.sh` is a SLURM
script; the paired `.py` runs standalone if you prefer. Checkpoints are not
included here — they are roughly 330 MB each — but every script writes them to
`results/models/`.

The prediction files under `results/predictions/` are gzipped. Load them with
`pandas.read_csv("...csv.gz")`, which handles decompression directly.

## Protocol

Windows are 500 samples at 50 Hz (10 s), with 50% overlap for cross-validation
and no overlap for leave-one-participant-out, matching the dataset paper.

Each fold holds out one further participant from the training pool as a
validation set, so training, validation and test sets are disjoint at the
participant level and model selection never sees the test fold. This matters:
an earlier version of these experiments selected checkpoints on the test set,
which inflated every score by roughly 4 to 7 points. All numbers here come from
the corrected protocol.

## Citing

See `CITATION.cff`. If you use the dataset itself, please cite Burchard et al.
(2026), [10.5281/zenodo.19425719](https://doi.org/10.5281/zenodo.19425719).

## License

MIT, see `LICENSE`. The HARMES dataset is distributed separately under CC BY
4.0.
