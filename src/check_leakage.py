"""Empirical leakage checks on the exact spatial-block splits used by E1 and E3.

For every task, block size and grid offset:
  - no block appears in both the training and the test side of a fold,
  - no point index appears on both sides, and every point is tested exactly once,
  - at the finest block size (offset 0), no exact coordinate is shared across a fold.
Then the saved out-of-fold kNN and MLP block predictions are checked to cover every point.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from spatial_cv import BLOCK_KM, TASKS, load_task, make_split  # noqa: E402


def coords_shared(lonlat, tr, te):
    rounded = np.round(lonlat, 6)
    return len(set(map(tuple, rounded[te])) & set(map(tuple, rounded[tr])))


def main():
    bad = 0
    for task in TASKS:
        lonlat, y = load_task(task)
        sizes = BLOCK_KM["housing" if task == "housing" else "global"]
        for s in sizes:
            for o in range(5):
                folds, g = make_split(lonlat, {"kind": "block", "size_km": s, "offset": o})
                cover = np.zeros(len(y), int)
                for tr, te in folds:
                    cover[te] += 1
                    shared_blocks = np.intersect1d(np.unique(g[tr]), np.unique(g[te]))
                    shared_xy = coords_shared(lonlat, tr, te) if (s == sizes[0] and o == 0) else 0
                    if len(shared_blocks) or len(np.intersect1d(tr, te)) or shared_xy:
                        bad += 1
                        print("LEAK", task, s, o, len(shared_blocks), shared_xy)
                if not (cover == 1).all():
                    bad += 1
                    print("COVER", task, s, o, np.bincount(cover))
        print(task, "ok" if not bad else "PROBLEMS", flush=True)

    # saved out-of-fold predictions: every point was predicted by some fold
    files = sorted(glob.glob("outputs/points/*__knn_emb__kindblock*")) + sorted(glob.glob("outputs/points/*__mlp__kindblock*"))
    for f in files:
        z = np.load(f)
        assert (z["fold"] >= 0).all(), f
    print(f"saved kNN/MLP block predictions checked: {len(files)} files;", "no problems" if not bad else f"{bad} problems")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
