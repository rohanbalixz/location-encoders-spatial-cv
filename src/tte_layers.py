"""E4 (added after E3, not pre-registered): where in TTE does linear readability appear?

TTE's location encoder is  coords -> w = softmax(tau_k <g, s_k>) over 4096 learned sites -> site_emb = w @ E (384-d)
-> semantic-register attention + skip + norm (512-d) -> residual MLP -> L2 normalize (512-d, the released embedding).

This script caches the two intermediate stages as extra "encoders" so the same ridge harness can probe them:
  tte_site : site_emb, the soft-Voronoi-weighted site embedding (384-d)
  tte_reg  : after register attention, before the MLP (512-d)
and writes descriptive stats of the soft assignment w (how many sites a location effectively mixes).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "TTE"))
from spatial_cv import EMB, TASKS, load_task

torch.set_num_threads(8)


def main():
    from tte import TTE

    m = TTE.from_pretrained("MVRL/TTE").eval()
    loc = m.location
    tau = loc.tau.detach().numpy()
    stats = []
    for task in TASKS:
        lonlat, _ = load_task(task)
        latlon = torch.tensor(lonlat[:, ::-1].copy(), dtype=torch.float32)
        S, R, neff, wmax, final = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(latlon), 4096):
                _, w = loc(latlon[i : i + 4096], return_weights=True)
                se = w @ loc._embeddings
                att, _ = loc.semantic_registers.forward_location(se)
                S.append(se.numpy())
                R.append(att.numpy())
                neff.append((1.0 / (w**2).sum(1)).numpy())
                wmax.append(w.max(1).values.numpy())
                final.append(torch.nn.functional.normalize(loc.mlp(att), dim=-1).numpy())
        np.save(EMB / f"tte_site__{task}.npy", np.concatenate(S).astype(np.float32))
        np.save(EMB / f"tte_reg__{task}.npy", np.concatenate(R).astype(np.float32))
        final = np.concatenate(final)
        ref = np.load(EMB / f"tte__{task}.npy")
        neff, wmax = np.concatenate(neff), np.concatenate(wmax)
        stats.append(
            {
                "task": task,
                "n": len(lonlat),
                "max_abs_diff_vs_cached_tte": float(np.abs(final - ref).max()),
                "neff_median": float(np.median(neff)),
                "neff_p10": float(np.percentile(neff, 10)),
                "neff_p90": float(np.percentile(neff, 90)),
                "wmax_median": float(np.median(wmax)),
            }
        )
        print(stats[-1], flush=True)
    st = pd.DataFrame(stats)
    st.to_csv(ROOT / "outputs/e4_tte_assignment_stats.csv", index=False)
    print(f"tau: min {tau.min():.1f} median {np.median(tau):.1f} max {tau.max():.1f}")


if __name__ == "__main__":
    main()
