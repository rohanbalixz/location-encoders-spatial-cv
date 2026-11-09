# Outputs

Every file here is written by the scripts in `../src` (run order: `../scripts/reproduce.sh`).
`RESULTS_TABLES.md` is the readable version: every table, with 95% block-bootstrap CIs.
The per-point predictions behind the CIs (`points/`, 318 MB) are not committed. The runs regenerate them.

## Results files: one row per task × encoder × probe × split

| File | Experiment | Written by |
|---|---|---|
| `results_e0_random.csv` | E0: random 80/20 split, seeds 42–46. Population rows are superseded by `results_e0_pop.csv` | `spatial_cv.py` |
| `results_e0_pop.csv` | E0 for the rebuilt Population target | `spatial_cv.py` |
| `results_e1_block.csv` | E1: square spatial blocks, 5 sizes × 5 grid offsets | `spatial_cv.py` |
| `results_e1b_sparse.csv` | E1b: random split with training labels thinned to 30 / 10 / 3 / 1% | `spatial_cv.py` |
| `results_e3_h3.csv` | E3c: H3 hexagon blocks | `spatial_cv.py` |
| `results_e3_ridgewide.csv` | E3d: ridge with α in 10⁻⁴…10⁴, chosen by grouped inner CV | `spatial_cv.py` |
| `results_e3_knnemb.csv` | E3a: kNN on embeddings | `spatial_cv.py` |
| `results_e3_mlp.csv` | E3b: MLP with sklearn defaults, 3 offsets | `spatial_cv.py` |
| `results_e3_mlp_tuned_{biome,reg}.csv` | E3b: MLP tuned by grouped inner CV, 1000 km blocks | `spatial_cv.py` |
| `results_e4_tte_layers{,_housing}.csv` | E4: ridge on TTE's intermediate stages | `spatial_cv.py` |

| Column | Meaning |
|---|---|
| `task` | `biome`, `ecoregion`, `country` (classification); `temperature`, `elevation`, `population`, `housing` (regression) |
| `encoder` | `tte`, `satclip_l40`, `satclip_l10`, `sh_l10` (fixed spherical harmonics), `xyz`, `coords` (raw lon/lat, used by `knn_coords`); E4 adds `tte_site` and `tte_reg` |
| `probe` | `ridge` (the paper's protocol), `ridge_wide_spatialcv`, `knn_emb`, `mlp`, `mlp_tuned`, `knn_coords` |
| `split_kind` | `random`, `sparse`, `block` or `h3` |
| `split_seed`, `split_frac` | random and thinned splits: seed, and the share of training labels kept |
| `split_size_km`, `split_offset` | block splits: block side in km, and grid offset (0–4) |
| `split_res` | H3 splits: hexagon resolution |
| `score` | accuracy (classification) or R² (regression), computed once over all out-of-fold predictions |
| `median_nn_km` | median great-circle distance from a test point to its nearest training label |
| `n_test`, `n_folds`, `n_groups` | points scored, folds, and blocks or cells (0 for random splits) |
| `probe_info` | JSON, one entry per fold: the chosen α, k, or MLP size and α (with inner-CV scores) |
| `ceiling` | classification only: share of test points whose class appears in their training fold |
| `acc_supported` | classification only: accuracy on those test points |
| `secs` | wall-clock seconds for the job |

## Derived files

| File | Contents |
|---|---|
| `gap_tte_vs_satclip.csv` | TTE − SatCLIP L40 under ridge per task and block size (`size` 0 = random), grid offset 0, with 95% block-bootstrap CI (`lo`, `hi`) and median label distance (`nn`). Plotted in Figure 2 |
| `decomposition_distance_region.csv` | Score on the random split, on thinned labels at the same label distance, and under blocks. `distance_part` = random − thinned, `region_part` = thinned − blocks |
| `e2_diagnostics.csv` | E2: for each label-free score (`D0_geo_km`, `D1_emb_DI`, `D2_emb_sens`), its Spearman correlation with per-point error and its `aurc_gain`: the share of the possible risk–coverage improvement achieved, where 0 is random abstention and 1 is the oracle. `gainD1_minus_D0` with CI compares D1 to D0. `subset`: all points, or classification points whose class is in training |
| `e2_risk_coverage.csv` | Risk–coverage curves behind Figure 4 |
| `e2_h5_site_density.csv` | H5: Spearman correlation between learned-site density within 500 km and TTE's per-block advantage, with CI and the mean advantage by density tercile |
| `e4_tte_assignment_stats.csv` | E4: effective number of sites per location (median, p10, p90), median top site weight, and a check that the re-run stages reproduce the cached TTE embeddings |

## Figures

| File | Shows |
|---|---|
| `figures/fig1_score_vs_label_distance.png` | Score against median distance to the nearest training label, all encoders, ridge probe |
| `figures/fig2_gap_tte_minus_satclip.png` | TTE − SatCLIP L40 under ridge by block size, offset 0, with CIs |
| `figures/fig3_gap_by_probe.png` | The same gap under ridge, kNN and MLP readouts |
| `figures/fig4_risk_coverage_tte.png` | Risk–coverage of the label-free trust scores for TTE at 1000 km blocks |
