"""The probes in src/spatial_cv.py are fit on training data only.

fit_predict never receives test labels. These tests check the remaining ways test data could leak in:
scaling or tuning on the test set would make one test point's prediction depend on the others.
"""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from spatial_cv import fit_predict

EMBEDDING_PROBES = ["ridge", "ridge_wide_spatialcv", "knn_emb", "mlp", "mlp_tuned"]


def toy(kind, n=240, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    signal = X[:, 0] + 0.5 * X[:, 1]
    if kind == "cls":
        return X, (signal > 0).astype(int) + (signal > 1).astype(int)  # three classes
    return X, signal + 0.1 * rng.normal(size=n)


@pytest.mark.parametrize("kind", ["cls", "reg"])
@pytest.mark.parametrize("probe", EMBEDDING_PROBES)
def test_a_test_points_prediction_does_not_depend_on_other_test_points(kind, probe):
    X, y = toy(kind)
    train, test = np.arange(160), np.arange(160, 240)
    groups = np.repeat(np.arange(16), 10)  # 16 spatial blocks of training points
    all_test, _ = fit_predict(X[train], y[train], X[test], kind, probe, groups_tr=groups)
    some_test, _ = fit_predict(X[train], y[train], X[test[:20]], kind, probe, groups_tr=groups)
    np.testing.assert_allclose(some_test, all_test[:20])


@pytest.mark.parametrize("probe", ["ridge", "knn_emb"])
def test_an_extreme_test_point_does_not_rescale_the_others(probe):
    X, y = toy("reg")
    X_test = X[160:].copy()
    X_test_outlier = X_test.copy()
    X_test_outlier[0] *= 1000  # would dominate a MinMaxScaler fit on train + test
    p, _ = fit_predict(X[:160], y[:160], X_test, "reg", probe)
    q, _ = fit_predict(X[:160], y[:160], X_test_outlier, "reg", probe)
    np.testing.assert_allclose(p[1:], q[1:])


@pytest.mark.parametrize("probe", ["ridge", "knn_emb", "mlp"])
def test_probes_recover_a_simple_signal(probe):
    X, y = toy("reg", n=500, d=3)
    p, _ = fit_predict(X[:400], y[:400], X[400:], "reg", probe)
    assert r2_score(y[400:], p) > 0.6


def test_tuned_mlp_picks_from_its_grid_by_grouped_inner_cv():
    X, y = toy("reg", n=160)
    groups = np.repeat(np.arange(12), 10)
    _, info = fit_predict(X[:120], y[:120], X[120:], "reg", "mlp_tuned", groups_tr=groups)
    assert tuple(info["hidden"]) in {(256,), (512, 512)}
    assert info["alpha"] in {1e-4, 1e-1}
    assert len(info["inner_scores"]) == 4


def test_coordinate_knn_looks_up_an_identical_training_point():
    """Why the random split rewards lookup: a test point that duplicates a training point gets its label."""
    rng = np.random.default_rng(0)
    lonlat = np.column_stack([rng.uniform(-180, 180, 200), rng.uniform(-60, 80, 200)])
    y = rng.integers(0, 5, 200)
    p, _ = fit_predict(lonlat, y, lonlat[:30], "cls", "knn_coords")
    np.testing.assert_array_equal(p, y[:30])
