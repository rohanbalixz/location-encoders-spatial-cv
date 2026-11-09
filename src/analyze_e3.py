"""E3 robustness (probe choice, alpha grid, block shape) and E2 (label-free trust scores, site density) sections
for RESULTS_TABLES.md, plus fig3 (TTE minus SatCLIP L40 gap under each probe).

Helpers (fmt, boot_gap, LABEL, ...) come from analyze.py, passed in as `h` to avoid a circular import."""

import json

import numpy as np
import pandas as pd

GLOBAL = ["biome", "ecoregion", "country", "temperature", "elevation", "population"]
BIG = {"housing": 100}  # largest block size shared by all probes; 1000 km for the global tasks


def _read(h, name):
    if name == "e3_mlp_tuned":  # run in two stages (biome, then regression); include rows of a stage still running
        fs = [
            f
            for stage in ("biome", "reg")
            for f in (h.OUT / f"results_{name}_{stage}.csv", h.OUT / f"results_{name}_{stage}.partial.csv")
            if f.exists()
        ]
        return pd.concat([pd.read_csv(f) for f in fs]) if fs else None
    f = h.OUT / f"results_{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def _setting(df):
    return np.where(df.split_kind == "random", "random", df.split_size_km.fillna(-1).astype(int).astype(str))


def _cell(h, task, df, enc, setting):
    r = df[(df.task == task) & (df.encoder == enc) & (df.setting == setting)]
    return (r.score.mean(), len(r)) if len(r) else (np.nan, 0)


def _ahead(df, task, setting, a="tte", b="satclip_l40"):
    r = df[(df.task == task) & (df.setting == setting)].copy()
    if r.empty:
        return ""
    key = "split_seed" if setting == "random" else "split_offset"
    p = r.pivot_table(index=key, columns="encoder", values="score")
    if a not in p or b not in p:
        return ""
    return f"{int((p[a] > p[b]).sum())}/{len(p)}"


def _ci(h, task, probe, setting):
    tag = "kindrandom_seed42" if setting == "random" else f"kindblock_size_km{setting}_offset0"
    if not (h.PTS / f"{task}__tte__{probe}__{tag}.npz").exists():
        return None
    return h.boot_gap(task, tag, probe=probe, by_block=setting != "random")


def e3_sections(e0, e1, h):
    md = []
    e0 = e0.assign(setting="random")
    e1 = e1.assign(setting=e1.split_size_km.astype(int).astype(str))
    ridge = pd.concat([e0, e1])
    probes = {"ridge (paper protocol)": ridge[ridge.probe == "ridge"]}
    for name, label in [
        ("e3_ridgewide", "ridge, wide α, spatial inner CV"),
        ("e3_knnemb", "kNN on embeddings"),
        ("e3_mlp", "MLP, untuned defaults"),
        ("e3_mlp_tuned", "MLP, tuned (grouped inner CV)"),
    ]:
        d = _read(h, name)
        if d is not None:
            probes[label] = d.assign(setting=_setting(d))
    coords = ridge[ridge.encoder == "coords"]

    # ---------------- probe comparison
    md.append("## E3a/b/d: does the TTE lead survive a change of probe?\n")
    md.append(
        "Same embeddings, same folds. TTE, SatCLIP L40 and the mean gap (TTE minus SatCLIP L40) are means over all grid "
        "offsets (blocks) or seeds (random); 'ahead' counts the offsets or seeds where TTE scores higher. The last gap "
        "column is offset 0 alone (seed 42 for random), next to its own 95% block-bootstrap CI (by point for random). "
        "kNN on raw coordinates is the lookup baseline.\n"
    )
    md.append(
        "| task | split | probe | TTE | SatCLIP L40 | gap, mean | ahead | gap, offset 0 [95% CI] | kNN on coords |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    verdict = []
    for k in h.ORDER:
        settings = ["random", "100", "250", "1000"] if k != "housing" else ["random", "25", "100"]
        for s in settings:
            for pname, df in probes.items():
                a, na = _cell(h, k, df, "tte", s)
                b, nb = _cell(h, k, df, "satclip_l40", s)
                if not na or not nb:
                    continue
                ci = _ci(h, k, df.probe.iloc[0], s)
                cis = f"{h.fmt(k, ci[0])} [{h.fmt(k, ci[1])}, {h.fmt(k, ci[2])}]" if ci else ""
                c, _ = _cell(h, k, coords, "coords", s)
                md.append(
                    f"| {h.LABEL[k]} | {s if s == 'random' else s + ' km'} | {pname} | {h.fmt(k, a)} | {h.fmt(k, b)} | "
                    f"{h.fmt(k, a - b)} | {_ahead(df, k, s)} | {cis} | {h.fmt(k, c) if not np.isnan(c) else ''} |"
                )
                big = str(BIG.get(k, 1000))
                if s == big:
                    verdict.append(
                        {
                            "task": k,
                            "probe": pname,
                            "gap": a - b,
                            "lo": ci[1] if ci else np.nan,
                            "hi": ci[2] if ci else np.nan,
                            "ahead": _ahead(df, k, s),
                        }
                    )
    md.append("")

    # ---------------- pre-registered decision rule at the largest shared block size
    v = pd.DataFrame(verdict)
    md.append("## Pre-registered rule: does 'TTE beats SatCLIP L40 under large blocks' hold under every probe?\n")
    md.append(
        "Rule (PREREGISTRATION.md): the sign must hold on every grid offset, under the ridge probe and the kNN probe. "
        "Added after the pre-registration: the offset-0 block-bootstrap CI must also exclude zero. Checked at 1000 km "
        "(global tasks) and 100 km (housing).\n"
    )
    md.append("| task | " + " | ".join(v.probe.unique()) + " | rule met? |")
    md.append("|---|" + "---|" * (v.probe.nunique() + 1))
    for k in h.ORDER:
        r = v[v.task == k].set_index("probe")
        cells, ok = [], True
        for pname in v.probe.unique():
            if pname not in r.index:
                cells.append("")
                continue
            x = r.loc[pname]
            n_ahead, n_tot = (int(t) for t in x.ahead.split("/"))
            sig = x.lo > 0 if x.lo == x.lo else None
            cells.append(
                f"{h.fmt(k, x.gap)} ({x.ahead}{', CI>0' if sig else (', CI spans 0' if sig is False else '')})"
            )
            if pname in ("ridge (paper protocol)", "kNN on embeddings"):
                ok &= (n_ahead == n_tot) and bool(sig)
        md.append(f"| {h.LABEL[k]} | " + " | ".join(cells) + f" | {'yes' if ok else 'no'} |")
    md.append("")

    # ---------------- probe headroom: how much does a non-linear probe add over ridge, per encoder?
    knn = probes.get("kNN on embeddings")
    if knn is not None:
        md.append("## How much a kNN readout adds over the linear readout, per encoder\n")
        md.append(
            "kNN-on-embeddings score minus ridge score (same folds). A large value means the information is in the "
            "embedding but a global linear map cannot read it out.\n"
        )
        md.append("| task | split | TTE: kNN − ridge | SatCLIP L40: kNN − ridge |")
        md.append("|---|---|---|---|")
        for k in h.ORDER:
            for s in ["random", "100", "250", "1000"] if k != "housing" else ["random", "25", "100"]:
                cells = []
                for enc in ("tte", "satclip_l40"):
                    a, na = _cell(h, k, knn, enc, s)
                    b, nb = _cell(h, k, probes["ridge (paper protocol)"], enc, s)
                    cells.append(h.fmt(k, a - b) if na and nb else "")
                md.append(f"| {h.LABEL[k]} | {s} | " + " | ".join(cells) + " |")
        md.append("")

    # ---------------- alphas chosen by the wide grid
    rw = _read(h, "e3_ridgewide")
    if rw is not None:
        rw["alphas"] = rw.probe_info.apply(lambda t: [d["alpha"] for d in json.loads(t)])
        md.append("## E3d: alphas picked by the wide grid (logspace −4..4, grouped inner CV)\n")
        md.append("| task | encoder | median α | range |")
        md.append("|---|---|---|---|")
        for (k, enc), r in rw.groupby(["task", "encoder"]):
            a = np.concatenate(r.alphas.values)
            md.append(f"| {h.LABEL[k]} | {h.NAME[enc]} | {np.median(a):.3g} | {a.min():.3g}–{a.max():.3g} |")
        md.append("")

    # ---------------- H3 hexagons
    hx = _read(h, "e3_h3")
    if hx is not None:
        md.append("## E3c: H3 hexagon blocks instead of square equal-area blocks (ridge; kNN for coords)\n")
        md.append(
            "| task | H3 res | median km to nearest label | TTE | SatCLIP L40 | SatCLIP L10 | kNN on coords | TTE − L40 | TTE − L10 |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|")
        for k in h.ORDER:
            for res in sorted(hx[hx.task == k].split_res.unique(), reverse=True):
                r = hx[(hx.task == k) & (hx.split_res == res)].set_index("encoder")
                sc = r.score
                md.append(
                    f"| {h.LABEL[k]} | {int(res)} | {r.median_nn_km.median():.0f} | {h.fmt(k, sc['tte'])} | {h.fmt(k, sc['satclip_l40'])} | "
                    f"{h.fmt(k, sc['satclip_l10'])} | {h.fmt(k, sc['coords'])} | {h.fmt(k, sc['tte'] - sc['satclip_l40'])} | "
                    f"{h.fmt(k, sc['tte'] - sc['satclip_l10'])} |"
                )
        md.append("")

    md += e4_section(h, ridge)
    fig3(h, probes)
    return md


def e4_section(h, ridge):
    """E4 (not pre-registered): linear probe on each stage of the TTE location encoder."""
    parts = [d for d in (_read(h, "e4_tte_layers"), _read(h, "e4_tte_layers_housing")) if d is not None]
    if not parts:
        return []
    d = pd.concat(parts)
    d = d.assign(setting=_setting(d))
    allr = pd.concat([d, ridge[ridge.probe == "ridge"]])
    md = [
        "## E4 (added after E3, not pre-registered): where in TTE does linear readability appear?\n",
        "Ridge probe (paper protocol) on each stage of the TTE location encoder, same folds. "
        "Stages: soft-Voronoi site embedding `w @ E` (384-d) → after semantic-register attention (512-d) → after the residual "
        "MLP, normalized (the released 512-d embedding). kNN on the released embedding is the readability ceiling.\n",
    ]
    st = h.OUT / "e4_tte_assignment_stats.csv"
    if st.exists():
        s = pd.read_csv(st)
        g = s[s.task != "housing"]
        md.append(
            f"Soft assignment: a location's effective number of sites (1/Σw²) has median {g.neff_median.median():.0f} "
            f"(10th–90th pct {g.neff_p10.median():.0f}–{g.neff_p90.median():.0f}) and the largest single weight has "
            f"median {g.wmax_median.median():.3f}, on the global tasks. Recomputed final embeddings match the cached ones "
            f"exactly (max |diff| {s.max_abs_diff_vs_cached_tte.max():.1g}).\n"
        )
    knn = _read(h, "e3_knnemb")
    knn = knn.assign(setting=_setting(knn)) if knn is not None else None
    md.append("| task | split | site embedding | + registers | + MLP (released) | SatCLIP L40 | kNN on released TTE |")
    md.append("|---|---|---|---|---|---|---|")
    for k in h.ORDER:
        for s in ["random", "1000"] if k != "housing" else ["random", "100"]:
            cells = [_cell(h, k, allr, e, s)[0] for e in ("tte_site", "tte_reg", "tte", "satclip_l40")]
            if np.isnan(cells[0]):
                continue
            kn = _cell(h, k, knn, "tte", s)[0] if knn is not None else np.nan
            md.append(
                f"| {h.LABEL[k]} | {s if s == 'random' else s + ' km'} | "
                + " | ".join(h.fmt(k, c) for c in cells)
                + f" | {h.fmt(k, kn) if not np.isnan(kn) else ''} |"
            )
    md.append("")
    return md


def fig3(h, probes):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tasks = [t for t in h.ORDER if t != "housing"]
    cols = {
        "ridge (paper protocol)": "#2a78d6",
        "kNN on embeddings": "#eb6834",
        "MLP, untuned defaults": "#1baf7a",
        "MLP, tuned (grouped inner CV)": "#eda100",
    }
    xs = ["random", "100", "250", "1000"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(14, 3.1))
    for ax, k in zip(axes, tasks):
        h.style(ax)
        ax.axhline(0, color=h.INK2, lw=0.8)
        scale = 100 if h.TASKS[k] == "cls" else 1
        for pname, col in cols.items():
            df = probes.get(pname)
            if df is None:
                continue
            # NaN where a probe was not run (MLP skips 250 km): the line breaks instead of implying a value there
            g = np.array([_cell(h, k, df, "tte", s)[0] - _cell(h, k, df, "satclip_l40", s)[0] for s in xs]) * scale
            if np.isfinite(g).any():
                ax.plot(range(len(xs)), g, "-o", color=col, lw=2, ms=5, mec="white", mew=1, label=pname)
        ax.set_xticks(range(len(xs)), ["rand.", "100", "250", "1000"])
        ax.set_title(
            h.LABEL[k].replace(" (acc)", " (pts)").replace(" (R²)", " (R²)"), fontsize=9.5, color=h.INK, loc="left"
        )
        ax.set_xlabel("block size (km)", fontsize=8.5, color=h.INK2)
    axes[0].set_ylabel("TTE minus SatCLIP L40", fontsize=8.5, color=h.INK2)
    handles = [Line2D([], [], color=c, lw=2, marker="o", mec="white", label=p) for p, c in cols.items() if p in probes]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower center", ncol=4, labelcolor=h.INK)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(h.FIG / "fig3_gap_by_probe.png", dpi=200)
    plt.close(fig)


def e2_sections(h):
    md = []
    f = h.OUT / "e2_diagnostics.csv"
    if not f.exists():
        return md
    e2 = pd.read_csv(f)
    a = e2[e2.subset == "all"]
    md.append("## E2: label-free trust scores (AURC gain over random abstention; higher is better)\n")
    md.append(
        "D0 = distance to nearest training label; D1 = dissimilarity index in embedding space (Meyer & Pebesma style); "
        "D2 = local embedding sensitivity (25 km). Offset 0. D1 − D0 CI is the 95% block bootstrap.\n"
    )
    md.append("| task | block | encoder | D0 | D1 | D2 | D1 − D0 | 95% CI |")
    md.append("|---|---|---|---|---|---|---|---|")
    wins = []
    for k in h.ORDER:
        for b in sorted(a[a.task == k].block_km.unique()):
            for enc in ("tte", "satclip_l40", "satclip_l10"):
                r = a[(a.task == k) & (a.block_km == b) & (a.encoder == enc)].set_index("score")
                if r.empty:
                    continue
                d1 = r.loc["D1_emb_DI"]
                wins.append({"task": k, "enc": enc, "gain": d1.gainD1_minus_D0, "lo": d1.ci_lo, "hi": d1.ci_hi})
                md.append(
                    f"| {h.LABEL[k]} | {int(b)} km | {h.NAME[enc]} | {r.loc['D0_geo_km'].aurc_gain:.3f} | {d1.aurc_gain:.3f} | "
                    f"{r.loc['D2_emb_sens'].aurc_gain:.3f} | {d1.gainD1_minus_D0:+.3f} | [{d1.ci_lo:+.3f}, {d1.ci_hi:+.3f}] |"
                )
    w = pd.DataFrame(wins)
    g = w[w.task != "housing"]
    md.append(
        f"\nGlobal tasks, {len(g)} task × block × encoder cases: D1 better than D0 with CI above 0 in {(g.lo > 0).sum()}, "
        f"CI spanning 0 in {((g.lo <= 0) & (g.hi >= 0)).sum()}, D1 worse with CI below 0 in {(g.hi < 0).sum()}. "
        "By encoder: " + "; ".join(f"{h.NAME[e]} {(r.lo > 0).sum()}/{len(r)}" for e, r in g.groupby("enc")) + ".\n"
    )

    f5 = h.OUT / "e2_h5_site_density.csv"
    if f5.exists():
        h5 = pd.read_csv(f5)
        md.append("## E2 / H5: is TTE's per-block advantage associated with learned-site density?\n")
        md.append(
            "Per held-out block: number of TTE learned sites within 500 km vs the TTE − SatCLIP L40 score on that block "
            "(ridge). Spearman with 95% bootstrap CI over blocks; mean advantage by site-density tercile.\n"
        )
        md.append("| task | block | blocks | Spearman | 95% CI | adv. low density | mid | high |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in h5.iterrows():
            md.append(
                f"| {h.LABEL[r.task]} | {int(r.block_km)} km | {int(r.n_blocks)} | {r.spearman_density_adv:+.3f} | "
                f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {h.fmt(r.task, r.adv_low_density)} | {h.fmt(r.task, r.adv_mid_density)} | "
                f"{h.fmt(r.task, r.adv_high_density)} |"
            )
        md.append("")
    return md
