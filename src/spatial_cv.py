"""E0/E1/E3/E4: random-split vs spatial-block evaluation of frozen location encoders.

The ridge probe mirrors external/TTE/eval/eval_range.py exactly (MinMaxScaler +
RidgeClassifierCV(alphas=(0.1,1,10), cv=10) / RidgeCV(alphas=(0.1,1,10), cv=3)).
E3 reads the same embeddings out with kNN and MLP probes on the same folds.

    python src/spatial_cv.py --exp e1_block --splits block --knn_coords
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import BallTree, KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data/processed"
EMB = ROOT / "data/emb"
OUT = ROOT / "outputs"
R_EARTH = 6371.0088

TASKS = {
    "biome": "cls",
    "ecoregion": "cls",
    "country": "cls",
    "temperature": "reg",
    "elevation": "reg",
    "population": "reg",
    "housing": "reg",
}
BLOCK_KM = {"global": [100, 250, 500, 1000, 2000], "housing": [10, 25, 50, 100, 200]}
H3_RES = {"global": [2, 1, 0], "housing": [5, 4, 3]}
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
RANDOM_SEEDS = [42, 43, 44, 45, 46]
N_OFFSETS = 5
SPARSE_FRACS = [0.3, 0.1, 0.03, 0.01]


# ----------------------------------------------------------------------------- data
def load_task(task):
    d = pd.read_parquet(DATA / f"{task}.parquet")
    lonlat = d[["lon", "lat"]].to_numpy(np.float64)
    if TASKS[task] == "cls":
        y, _ = pd.factorize(d["y"])
    else:
        y = d["y"].to_numpy(np.float64)
    return lonlat, y


def get_embeddings(task, enc_name):
    EMB.mkdir(parents=True, exist_ok=True)
    f = EMB / f"{enc_name}__{task}.npy"
    if not f.exists():
        from encoders import get_encoder

        lonlat, _ = load_task(task)
        np.save(f, get_encoder(enc_name)(lonlat))
    return np.load(f)


# ----------------------------------------------------------------------------- splits
def equal_area_xy(lonlat):
    lam = np.radians(lonlat[:, 0])
    phi = np.radians(lonlat[:, 1])
    return R_EARTH * lam, R_EARTH * np.sin(phi)  # Lambert cylindrical equal-area, km


def block_ids(lonlat, size_km, offset_seed):
    x, y = equal_area_xy(lonlat)
    rng = np.random.default_rng(1000 + offset_seed)
    ox, oy = rng.uniform(0, size_km, 2)
    bx = np.floor((x + ox) / size_km).astype(np.int64)
    by = np.floor((y + oy) / size_km).astype(np.int64)
    return bx * 1_000_003 + by


def h3_ids(lonlat, res):
    import h3

    cells = [h3.latlng_to_cell(la, lo, res) for lo, la in lonlat]
    return pd.factorize(np.array(cells))[0]


def make_split(lonlat, split):
    """Return list of (train_idx, test_idx) and per-point group ids (or None)."""
    n = len(lonlat)
    kind = split["kind"]
    if kind == "random":
        rng = np.random.default_rng(split["seed"])
        perm = rng.permutation(n)
        n_tr = int(round(0.8 * n))
        return [(perm[:n_tr], perm[n_tr:])], None
    if kind == "sparse":  # random 80/20, then keep only a random fraction of the training labels
        rng = np.random.default_rng(split["seed"])
        perm = rng.permutation(n)
        n_tr = int(round(0.8 * n))
        tr = perm[:n_tr]
        tr = tr[: max(50, int(round(split["frac"] * n_tr)))]
        return [(tr, perm[n_tr:])], None
    if kind == "block":
        g = block_ids(lonlat, split["size_km"], split["offset"])
    elif kind == "h3":
        g = h3_ids(lonlat, split["res"])
    else:
        raise ValueError(kind)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=split.get("offset", 0))
    return list(gkf.split(np.zeros(n), groups=g)), g


# ----------------------------------------------------------------------------- probes
def fit_predict(Xtr, ytr, Xte, kind, probe, groups_tr=None):
    if probe == "knn_coords":  # Xtr/Xte are lonlat in degrees
        A = np.radians(Xtr[:, ::-1])
        B = np.radians(Xte[:, ::-1])
        est = (KNeighborsClassifier if kind == "cls" else KNeighborsRegressor)(
            metric="haversine", algorithm="ball_tree", weights="distance"
        )
        gs = GridSearchCV(est, {"n_neighbors": [1, 3, 10, 30]}, cv=3)
        gs.fit(A, ytr)
        return gs.predict(B), {"k": gs.best_params_["n_neighbors"]}
    sc = MinMaxScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    Xte = sc.transform(Xte)
    if probe == "ridge":
        m = RidgeClassifierCV(alphas=RIDGE_ALPHAS, cv=10) if kind == "cls" else RidgeCV(alphas=RIDGE_ALPHAS, cv=3)
        m.fit(Xtr, ytr)
        return m.predict(Xte), {"alpha": float(m.alpha_)}
    if probe == "ridge_wide_spatialcv":  # wider alpha grid, alpha picked by grouped (spatial) inner CV
        alphas = np.logspace(-4, 4, 17)
        inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=0).split(Xtr, groups=groups_tr))
        if kind == "cls":
            from sklearn.linear_model import RidgeClassifier as M
        else:
            from sklearn.linear_model import Ridge as M
        gs = GridSearchCV(M(), {"alpha": alphas}, cv=inner)
        gs.fit(Xtr, ytr)
        return gs.predict(Xte), {"alpha": float(gs.best_params_["alpha"])}
    if probe == "knn_emb":
        est = (KNeighborsClassifier if kind == "cls" else KNeighborsRegressor)(weights="distance", algorithm="brute")
        gs = GridSearchCV(est, {"n_neighbors": [3, 10, 30]}, cv=3)
        gs.fit(Xtr, ytr)
        return gs.predict(Xte), {"k": gs.best_params_["n_neighbors"]}
    if probe == "mlp":
        M = MLPClassifier if kind == "cls" else MLPRegressor
        yt = ytr
        if kind == "reg":
            mu, sd = ytr.mean(), ytr.std()
            yt = (ytr - mu) / sd
        m = M(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=0, alpha=1e-4)
        m.fit(Xtr, yt)
        p = m.predict(Xte)
        if kind == "reg":
            p = p * sd + mu
        return p, {}
    if probe == "mlp_tuned":  # small grid, picked by grouped (spatial) inner CV on the training folds only
        M = MLPClassifier if kind == "cls" else MLPRegressor
        yt = ytr
        if kind == "reg":
            mu, sd = ytr.mean(), ytr.std()
            yt = (ytr - mu) / sd
        grid = {"hidden_layer_sizes": [(256,), (512, 512)], "alpha": [1e-4, 1e-1]}
        inner = (
            list(GroupKFold(n_splits=3, shuffle=True, random_state=0).split(Xtr, groups=groups_tr))
            if groups_tr is not None
            else 3
        )
        gs = GridSearchCV(M(early_stopping=True, max_iter=200, random_state=0), grid, cv=inner)
        gs.fit(Xtr, yt)
        p = gs.predict(Xte)
        if kind == "reg":
            p = p * sd + mu
        bp = gs.best_params_
        return p, {
            "hidden": list(bp["hidden_layer_sizes"]),
            "alpha": bp["alpha"],
            "inner_scores": [round(float(v), 4) for v in gs.cv_results_["mean_test_score"]],
        }
    raise ValueError(probe)


def nn_dist_km(lonlat_tr, lonlat_te):
    t = BallTree(np.radians(lonlat_tr[:, ::-1]), metric="haversine")
    d, _ = t.query(np.radians(lonlat_te[:, ::-1]), k=1)
    return d[:, 0] * R_EARTH


# ----------------------------------------------------------------------------- one job
def run_job(task, enc, probe, split, save_points=False):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    kind = TASKS[task]
    lonlat, y = load_task(task)
    X = lonlat if probe == "knn_coords" else get_embeddings(task, enc)
    folds, groups = make_split(lonlat, split)
    n = len(y)
    pred = np.full(n, np.nan) if kind == "reg" else np.full(n, -1, dtype=np.int64)
    fold_id = np.full(n, -1, dtype=np.int16)
    nnd = np.full(n, np.nan)
    supported = np.ones(n, dtype=bool)
    infos = []
    for k, (tr, te) in enumerate(folds):
        g_tr = groups[tr] if groups is not None else None
        p, info = fit_predict(X[tr], y[tr], X[te], kind, probe, g_tr)
        pred[te] = p
        fold_id[te] = k
        nnd[te] = nn_dist_km(lonlat[tr], lonlat[te])
        if kind == "cls":
            supported[te] = np.isin(y[te], np.unique(y[tr]))
        infos.append(info)
    m = fold_id >= 0
    if kind == "cls":
        score = float(np.mean(pred[m] == y[m]))
        extra = {
            "ceiling": float(supported[m].mean()),
            "acc_supported": float(np.mean(pred[m & supported] == y[m & supported])),
        }
    else:
        score = float(r2_score(y[m], pred[m]))
        extra = {}
    row = {
        "task": task,
        "encoder": enc,
        "probe": probe,
        **{f"split_{k}": v for k, v in split.items()},
        "score": score,
        "median_nn_km": float(np.median(nnd[m])),
        "n_test": int(m.sum()),
        "n_folds": len(folds),
        "n_groups": int(len(np.unique(groups))) if groups is not None else 0,
        "probe_info": json.dumps(infos),
        "secs": round(time.time() - t0, 1),
        **extra,
    }
    if save_points:
        pdir = OUT / "points"
        pdir.mkdir(parents=True, exist_ok=True)
        tag = "_".join(f"{k}{v}" for k, v in split.items())
        np.savez_compressed(
            pdir / f"{task}__{enc}__{probe}__{tag}.npz",
            y=y,
            pred=pred,
            fold=fold_id,
            nn_km=nnd,
            supported=supported,
            groups=groups if groups is not None else np.zeros(n, dtype=np.int64),
        )
    return row


def splits_for(task, which):
    reg = "housing" if task == "housing" else "global"
    out = []
    if "random" in which:
        out += [{"kind": "random", "seed": s} for s in RANDOM_SEEDS]
    if "block" in which:
        out += [{"kind": "block", "size_km": b, "offset": o} for b in BLOCK_KM[reg] for o in range(N_OFFSETS)]
    if "h3" in which:
        out += [{"kind": "h3", "res": r} for r in H3_RES[reg]]
    if "sparse" in which:
        out += [{"kind": "sparse", "seed": s, "frac": f} for f in SPARSE_FRACS for s in RANDOM_SEEDS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--encoders", nargs="+", default=["tte", "satclip_l40", "satclip_l10", "sh_l10", "xyz"])
    ap.add_argument("--probe", default="ridge")
    ap.add_argument("--splits", nargs="+", default=["random", "block"])
    ap.add_argument("--block_sizes", nargs="+", type=float, default=None)
    ap.add_argument("--offsets", type=int, default=N_OFFSETS)
    ap.add_argument("--knn_coords", action="store_true", help="also run the kNN-on-coordinates baseline")
    ap.add_argument("--n_jobs", type=int, default=9)
    a = ap.parse_args()

    for t in a.tasks:  # precompute embeddings serially (cached)
        for e in a.encoders:
            get_embeddings(t, e)

    jobs = []
    for t in a.tasks:
        sp = splits_for(t, a.splits)
        if a.block_sizes:
            sp = [s for s in sp if s["kind"] != "block" or s["size_km"] in a.block_sizes]
        sp = [s for s in sp if s["kind"] != "block" or s["offset"] < a.offsets]
        pairs = [(e, a.probe) for e in a.encoders]
        if a.knn_coords:
            pairs.append(("coords", "knn_coords"))
        for e, pr in pairs:
            for s in sp:
                save = (
                    (s["kind"] == "random" and s["seed"] == 42)
                    or (s["kind"] == "block" and s["offset"] == 0)
                    or s["kind"] == "h3"
                )
                jobs.append((t, e, pr, s, save))
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    out = OUT / f"results_{a.exp}.csv"
    partial = OUT / f"results_{a.exp}.partial.csv"
    rows = []
    # rows are written as each job finishes, so a long run that is interrupted keeps what it finished
    for r in Parallel(n_jobs=a.n_jobs, verbose=5, return_as="generator_unordered")(delayed(run_job)(*j) for j in jobs):
        rows.append(r)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(
            f"[{time.strftime('%H:%M:%S')}] finished {len(rows)}/{len(jobs)}: {r['task']} {r['encoder']} "
            f"{r.get('split_size_km', r.get('split_seed', ''))} offset {r.get('split_offset', '')} score {r['score']:.4f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    partial.unlink(missing_ok=True)
    print(f"done in {time.time() - t0:.0f}s -> outputs/results_{a.exp}.csv")


if __name__ == "__main__":
    main()
