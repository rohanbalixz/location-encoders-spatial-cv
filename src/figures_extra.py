"""Fig 3: E2 risk-coverage of the label-free trust scores for TTE at 1000 km blocks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"biome": "Biome", "temperature": "Temperature", "elevation": "Elevation", "population": "Population",
         "country": "Country", "ecoregion": "EcoRegion", "housing": "California housing"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig3():
    rc = pd.read_csv(OUT / "e2_risk_coverage.csv")
    tasks = ["biome", "temperature", "elevation", "population"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, t in zip(axes, tasks):
        style(ax)
        r = rc[(rc.task == t) & (rc.encoder == "tte") & (rc.block_km == 1000)]
        for sc, col, lab, ls in [("D1_emb_DI", C1, "embedding dissimilarity (D1)", "-"),
                                 ("D0_geo_km", C2, "distance to nearest label (D0)", "-"),
                                 ("D2_emb_sens", C3, "local embedding sensitivity (D2)", "-"),
                                 ("oracle", INK2, "oracle (sorted by true error)", ":")]:
            s = r[r.score == sc].sort_values("coverage")
            ax.plot(s.coverage, s.risk, ls, color=col, lw=2 if sc != "oracle" else 1.2, label=lab)
        full = r[(r.score == "D0_geo_km") & (r.coverage == r.coverage.max())].risk.iloc[0]
        ax.axhline(full, color=INK2, lw=0.8, ls="--", label="no abstention")
        ax.set_title(f"{LABEL[t]}: TTE, 1000 km blocks", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("coverage (share of test points kept)", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("error on kept points\n(misclassification or MSE/var)", fontsize=8.5, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig3_risk_coverage_tte.png", dpi=200)
    print("wrote figures/fig3_risk_coverage_tte.png")


if __name__ == "__main__":
    fig3()
