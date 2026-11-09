#!/bin/bash
# Re-run every experiment in this repo, in the order the results in outputs/ were produced.
# Assumes the setup in README.md is done: data/raw/ filled, external/TTE cloned at 2bce5e6.
#
#   PY="python" N_JOBS=8 ./scripts/reproduce.sh
#
# Rough wall-clock times on an Apple M3 Pro (11 cores, N_JOBS=10): E0 and E1 under an hour,
# E3 kNN probe about 1.7 h, tuned MLP about 3.3 h, every other step under 20 min; about 6.5 h in total.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N_JOBS="${N_JOBS:-8}"
CV="$PY src/spatial_cv.py --n_jobs $N_JOBS"
step() { echo "[$(date +%T)] $*"; }

step "labels: rebuild the seven tasks from public sources"
$PY src/build_data.py

step "E0: random 80/20 splits, seeds 42-46"
# Population was rebuilt (5' area averages) after the first random-split run, so it has its own
# results file; analyze.py takes Population from e0_pop.
$CV --exp e0_random --splits random --knn_coords --tasks biome ecoregion country temperature elevation housing
$CV --exp e0_pop    --splits random --knn_coords --tasks population

step "E1: spatial blocks, 5 sizes x 5 grid offsets"
$CV --exp e1_block --splits block --knn_coords

step "E1b: random split with thinned training labels"
$CV --exp e1b_sparse --splits sparse --knn_coords

step "E3c: H3 hexagons instead of square blocks"
$CV --exp e3_h3 --splits h3 --knn_coords

step "E2: label-free trust scores and site density (uses the saved E1 predictions)"
$PY src/diagnostics.py

step "E3d: wide alpha grid, chosen by grouped inner CV"
$CV --exp e3_ridgewide --probe ridge_wide_spatialcv --tasks biome temperature elevation population housing \
    --encoders tte satclip_l40 --splits block --block_sizes 250 1000 25 100

step "E3a: kNN on embeddings"
$CV --exp e3_knnemb --probe knn_emb --encoders tte satclip_l40 --splits random block --block_sizes 250 1000 25 100

step "E3b: MLP on embeddings, sklearn defaults, 3 offsets"
$CV --exp e3_mlp --probe mlp --tasks biome temperature elevation population housing --encoders tte satclip_l40 \
    --splits random block --block_sizes 1000 100 --offsets 3

step "E3b: MLP on embeddings, size and alpha chosen by grouped inner CV, 1000 km blocks"
$CV --exp e3_mlp_tuned_biome --probe mlp_tuned --tasks biome --encoders tte satclip_l40 --splits block --block_sizes 1000
$CV --exp e3_mlp_tuned_reg   --probe mlp_tuned --tasks temperature elevation population --encoders tte satclip_l40 \
    --splits block --block_sizes 1000

step "E4: ridge probe on TTE's intermediate stages"
$PY src/tte_layers.py
$CV --exp e4_tte_layers --tasks biome ecoregion country temperature elevation population --encoders tte_site tte_reg \
    --splits random block --block_sizes 1000
$CV --exp e4_tte_layers_housing --tasks housing --encoders tte_site tte_reg --splits random block --block_sizes 100

step "leakage checks on the exact splits and saved predictions"
$PY src/check_leakage.py

step "tables and figures"
$PY src/analyze.py
$PY src/figures_extra.py
step "done: outputs/RESULTS_TABLES.md, outputs/figures/"
