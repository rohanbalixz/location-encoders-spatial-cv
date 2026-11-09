# Hypotheses and decision rules (written before any spatial-block result was computed)

Written: 2026-11-07, before E1–E3 were run. Only the encoders had been loaded at that point, plus a check of site displacement.

## Setup (fixed in advance)
- Encoders: TTE (MVRL/TTE, L2-normalized, as in eval/eval_range.py), SatCLIP ViT16-L10 and ViT16-L40 (microsoft), and a fixed spherical-harmonic basis (L=10, 100 features).
- Simple baselines: raw xyz on the unit sphere, and a kNN regressor/classifier on coordinates (haversine).
- Probe, the same as TTE's eval_range.py: MinMaxScaler, then RidgeClassifierCV(alphas=(0.1,1,10), cv=10) or RidgeCV(alphas=(0.1,1,10), cv=3).
- Random split: 80/20 with seeds 42–46, as in TTE's eval_range.py (`--n_seeds 5`).
- Block split: 5-fold grouped CV over square blocks on a Lambert cylindrical equal-area grid, with block sides of 100, 250, 500, 1000 and 2000 km and 5 random grid offsets per size.
- Primary x-axis: the median great-circle distance from each test point to its nearest training point.

## Hypotheses
- H1. All encoders score lower as blocks grow. TTE's advantage over SatCLIP on a random split shrinks as the distance to the nearest label grows. Tasks: Temperature, Elevation, Population, Biome.
- H2. kNN on coordinates is competitive on the random split and degrades fastest under blocks. In other words, the random split partly measures lookup.
- H3. For classification under blocks, part of the accuracy drop comes from classes missing in the training folds (positivity in label space). Accuracy on test points whose class was seen in training drops less.
- H4. Under block CV, dissimilarity in embedding space (D1: distance to the nearest training embedding, normalized as in Meyer & Pebesma's dissimilarity index) ranks per-point error better than geographic distance to the nearest label (D0). Measured by Spearman correlation and the area under the risk–coverage curve.
- H5. TTE's per-block advantage over SatCLIP is positively associated with the local density of learned sites.

## Decision rules
- A claim is reported as a finding only if it holds for all 5 grid offsets and under both the ridge probe and a kNN-on-embeddings probe (E3).
- Uncertainty comes from a block bootstrap over test blocks (1,000 resamples), never from point-level intervals.
- Labels are rebuilt from public sources, not the RANGE CSVs, which are not public. The random-split pass criterion for E0 is that TTE vs SatCLIP shows the same ordering and a gap of similar size to the paper's Table 1, not identical numbers.
- If a hypothesis fails, it is reported as failed.
