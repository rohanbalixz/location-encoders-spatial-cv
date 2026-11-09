"""E2: can a label-free score predict where a frozen location encoder will fail under
spatial-block CV?  Plus H5: is TTE's advantage tied to learned site density?

Scores (higher = less trustworthy), per test point:
  D0  geographic distance to the nearest training label (km)
  D1  embedding-space dissimilarity: Euclidean distance (MinMax-scaled, train-fit) to the
      nearest training embedding / mean pairwise training distance (Meyer & Pebesma 2021 DI,
      unweighted, computed in the encoder's embedding space)
  D2  local embedding sensitivity ||e(x+25km) - e(x)||, mean over N/E/S/W (exploratory)
Uses the saved out-of-fold predictions from spatial_cv.py (offset 0), so no refitting.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from spatial_cv import TASKS, R_EARTH, load_task, get_embeddings  # noqa: E402

OUT = ROOT / "outputs"
PTS = OUT / "points"
BLOCKS = {"global": [500, 1000], "housing": [50, 100]}
ENCS = ["tte", "satclip_l40", "satclip_l10"]
N_BOOT = 500
COV = np.linspace(0.05, 1.0, 20)


def per_point_error(task, y, pred):
    if TASKS[task] == "cls":
        return (pred != y).astype(float)
    return (y - pred) ** 2 / np.var(y)


def embedding_dissimilarity(E, fold):
    d1 = np.full(len(E), np.nan)
    rng = np.random.default_rng(0)
    for k in np.unique(fold[fold >= 0]):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        sc = MinMaxScaler().fit(E[tr])
        A, B = sc.transform(E[tr]), sc.transform(E[te])
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(A)
        dist, _ = nn.kneighbors(B)
        s = A[rng.choice(len(A), size=min(2000, len(A)), replace=False)]
        dbar = pairwise_distances(s)[np.triu_indices(len(s), 1)].mean()
        d1[te] = dist[:, 0] / dbar
    return d1


def local_sensitivity(task, enc_name, lonlat, delta_km=25.0):
    f = ROOT / "data/emb" / f"{enc_name}__{task}__sens.npy"
    if f.exists():
        return np.load(f)
    from encoders import get_encoder
    enc = get_encoder(enc_name)
    E0 = enc(lonlat)
    sc = MinMaxScaler().fit(E0)
    E0s = sc.transform(E0)
    dlat = np.degrees(delta_km / R_EARTH)
    dlon = dlat / np.maximum(np.cos(np.radians(lonlat[:, 1])), 0.05)
    tot = np.zeros(len(lonlat))
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        p = lonlat.copy()
        p[:, 0] = (p[:, 0] + dx * dlon + 180) % 360 - 180
        p[:, 1] = np.clip(p[:, 1] + dy * dlat, -89.9, 89.9)
        tot += np.linalg.norm(sc.transform(enc(p)) - E0s, axis=1)
    out = tot / 4
    np.save(f, out)
    return out


def risk_coverage(score, err, w=None):
    order = np.argsort(score, kind="stable")
    e = err[order]
    w = np.ones_like(e) if w is None else w[order]
    cw, ce = np.cumsum(w), np.cumsum(w * e)
    idx = np.clip(np.searchsorted(cw, COV * cw[-1]), 0, len(e) - 1)
    return ce[idx] / cw[idx]


def aurc_gain(score, err, w=None):
    """Share of the achievable risk-coverage improvement over random abstention: 0=random, 1=oracle."""
    rc = risk_coverage(score, err, w).mean()
    orc = risk_coverage(err, err, w).mean()
    base = np.average(err, weights=w)
    return (base - rc) / (base - orc + 1e-12)


def block_boot(groups, n_boot=N_BOOT, seed=0):
    ug, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        yield cnt[inv].astype(float)


def main():
    rows, curves, h5 = [], [], []
    for task in TASKS:
        reg = "housing" if task == "housing" else "global"
        lonlat, y = load_task(task)
        for size in BLOCKS[reg]:
            tag = f"kindblock_size_km{size}_offset0"
            errs = {}
            for enc in ENCS:
                f = PTS / f"{task}__{enc}__ridge__{tag}.npz"
                if not f.exists():
                    print("missing", f.name); continue
                z = np.load(f)
                pred, fold, nn_km, groups, sup = z["pred"], z["fold"], z["nn_km"], z["groups"], z["supported"]
                err = per_point_error(task, y, pred)
                errs[enc] = err
                D = {"D0_geo_km": nn_km,
                     "D1_emb_DI": embedding_dissimilarity(get_embeddings(task, enc), fold),
                     "D2_emb_sens": local_sensitivity(task, enc, lonlat)}
                subsets = {"all": np.ones(len(y), bool)}
                if TASKS[task] == "cls":
                    subsets["supported"] = sup
                for sub, msk in subsets.items():
                    for dn, dv in D.items():
                        rho = spearmanr(dv[msk], err[msk]).statistic
                        g = aurc_gain(dv[msk], err[msk])
                        row = dict(task=task, block_km=size, encoder=enc, subset=sub, score=dn,
                                   spearman=rho, aurc_gain=g, mean_err=err[msk].mean())
                        if dn == "D1_emb_DI":  # paired block-bootstrap CI for D1 - D0
                            diffs = [aurc_gain(dv[msk], err[msk], w[msk]) - aurc_gain(D["D0_geo_km"][msk], err[msk], w[msk])
                                     for w in block_boot(groups)]
                            row["gainD1_minus_D0"] = g - aurc_gain(D["D0_geo_km"][msk], err[msk])
                            row["ci_lo"], row["ci_hi"] = np.percentile(diffs, [2.5, 97.5])
                        rows.append(row)
                        if sub == "all":
                            rc = risk_coverage(dv, err)
                            curves += [dict(task=task, block_km=size, encoder=enc, score=dn, coverage=c, risk=r) for c, r in zip(COV, rc)]
                    if sub == "all":
                        rc = risk_coverage(err, err)
                        curves += [dict(task=task, block_km=size, encoder=enc, score="oracle", coverage=c, risk=r) for c, r in zip(COV, rc)]
                print(f"{task:11s} {size:5d}km {enc:12s} done", flush=True)

            # ---- H5: per-block TTE advantage vs learned site density
            if "tte" in errs and "satclip_l40" in errs:
                from encoders import TTEEncoder
                sites, _ = TTEEncoder().sites_lonlat()
                init = np.load(ROOT / "external/TTE/train/sites/sites_4096.npz")["sites"]
                init = init / np.linalg.norm(init, axis=1, keepdims=True)
                init_ll = np.stack([np.degrees(np.arctan2(init[:, 1], init[:, 0])), np.degrees(np.arcsin(init[:, 2]))], 1)
                q = np.radians(lonlat[:, ::-1])
                r = 500.0 / R_EARTH
                dens = BallTree(np.radians(sites[:, ::-1]), metric="haversine").query_radius(q, r, count_only=True)
                dens0 = BallTree(np.radians(init_ll[:, ::-1]), metric="haversine").query_radius(q, r, count_only=True)
                z = np.load(PTS / f"{task}__tte__ridge__{tag}.npz")
                adv = errs["satclip_l40"] - errs["tte"]  # >0: TTE better
                df = pd.DataFrame({"g": z["groups"], "adv": adv, "dens": dens, "ddens": dens - dens0})
                b = df.groupby("g").agg(adv=("adv", "mean"), dens=("dens", "mean"), ddens=("ddens", "mean"), n=("adv", "size"))
                b = b[b.n >= 20]
                rho = spearmanr(b.dens, b.adv).statistic
                rho_d = spearmanr(b.ddens, b.adv).statistic
                rng = np.random.default_rng(0)
                boots = [spearmanr(*b.iloc[i][["dens", "adv"]].to_numpy().T).statistic
                         for i in (rng.integers(0, len(b), len(b)) for _ in range(N_BOOT))]
                terc = pd.qcut(b.dens.rank(method="first"), 3, labels=["low", "mid", "high"])
                h5.append(dict(task=task, block_km=size, n_blocks=len(b), spearman_density_adv=rho,
                               ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                               spearman_density_change_adv=rho_d,
                               **{f"adv_{k}_density": v for k, v in b.groupby(terc, observed=True).adv.mean().items()}))
    pd.DataFrame(rows).to_csv(OUT / "e2_diagnostics.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "e2_risk_coverage.csv", index=False)
    pd.DataFrame(h5).to_csv(OUT / "e2_h5_site_density.csv", index=False)
    print("wrote e2_diagnostics.csv, e2_risk_coverage.csv, e2_h5_site_density.csv")


if __name__ == "__main__":
    main()
