"""Properties of the splits in src/spatial_cv.py that the results depend on."""

import numpy as np
import pytest

from spatial_cv import R_EARTH, block_ids, equal_area_xy, make_split, nn_dist_km


@pytest.mark.parametrize("size_km", [250, 1000, 2000])
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_block_folds_are_disjoint_by_block_and_test_every_point_once(global_points, size_km, offset):
    lonlat = global_points()
    folds, groups = make_split(lonlat, {"kind": "block", "size_km": size_km, "offset": offset})
    assert len(folds) == 5
    tested = np.zeros(len(lonlat), dtype=int)
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        assert np.intersect1d(np.unique(groups[train]), np.unique(groups[test])).size == 0
        assert len(train) + len(test) == len(lonlat)
        tested[test] += 1
    assert (tested == 1).all()


@pytest.mark.parametrize("size_km", [100, 500, 1000])
def test_points_in_one_block_lie_within_one_block_side(global_points, size_km):
    lonlat = global_points()
    x, y = equal_area_xy(lonlat)
    groups = block_ids(lonlat, size_km, offset_seed=2)
    for g in np.unique(groups):
        member = groups == g
        assert np.ptp(x[member]) < size_km
        assert np.ptp(y[member]) < size_km


def test_equal_area_projection_spans_the_sphere():
    x, y = equal_area_xy(np.array([[-180.0, -90.0], [180.0, 90.0], [0.0, 30.0]]))
    np.testing.assert_allclose(x[:2], [-np.pi * R_EARTH, np.pi * R_EARTH])
    np.testing.assert_allclose(y, [-R_EARTH, R_EARTH, R_EARTH / 2])  # y = R sin(lat): equal areas, equal heights


def test_grid_offsets_move_the_block_boundaries(global_points):
    lonlat = global_points()
    a = block_ids(lonlat, 1000, offset_seed=0)
    b = block_ids(lonlat, 1000, offset_seed=1)
    # if the offset did nothing, every block of `a` would map to exactly one block of `b`
    assert len(set(zip(a, b))) > len(np.unique(a))


def test_random_split_is_80_20_disjoint_and_seeded(global_points):
    lonlat = global_points(1000)
    ((train, test),), groups = make_split(lonlat, {"kind": "random", "seed": 42})
    assert groups is None
    assert (len(train), len(test)) == (800, 200)
    assert np.union1d(train, test).size == 1000
    ((train_again, _),), _ = make_split(lonlat, {"kind": "random", "seed": 42})
    ((train_other, _),), _ = make_split(lonlat, {"kind": "random", "seed": 43})
    np.testing.assert_array_equal(train, train_again)
    assert not np.array_equal(np.sort(train), np.sort(train_other))


@pytest.mark.parametrize("frac", [0.3, 0.1, 0.03, 0.01])
def test_thinned_labels_keep_the_random_split_test_set(global_points, frac):
    lonlat = global_points(2000)
    ((train_full, test_random),), _ = make_split(lonlat, {"kind": "random", "seed": 44})
    ((train, test),), _ = make_split(lonlat, {"kind": "sparse", "seed": 44, "frac": frac})
    np.testing.assert_array_equal(test, test_random)  # only the training labels are thinned
    assert set(train) <= set(train_full)
    assert len(train) == max(50, round(frac * len(train_full)))


def test_h3_folds_are_disjoint_by_cell(global_points):
    pytest.importorskip("h3")
    lonlat = global_points(1500)
    folds, groups = make_split(lonlat, {"kind": "h3", "res": 1})
    tested = np.zeros(len(lonlat), dtype=int)
    for train, test in folds:
        assert np.intersect1d(np.unique(groups[train]), np.unique(groups[test])).size == 0
        tested[test] += 1
    assert (tested == 1).all()


def test_nearest_label_distance_is_great_circle_km():
    train = np.array([[0.0, 0.0]])
    test = np.array([[0.0, 1.0], [1.0, 0.0], [180.0, 0.0]])  # 1 degree north, 1 degree east, antipode
    one_degree = np.pi * R_EARTH / 180
    np.testing.assert_allclose(nn_dist_km(train, test), [one_degree, one_degree, np.pi * R_EARTH], rtol=1e-9)
