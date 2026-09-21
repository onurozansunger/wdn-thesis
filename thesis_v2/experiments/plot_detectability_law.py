"""Plot the detectability collapse.

Left panel: the label-free |z| score. Right: the trained supervised head.
Points are (dataset, family) pairs from test_detectability_law.py, spanning
two networks, four regimes and five attack mechanisms. Lines are the
matched-magnitude injection sweeps from test_generalisation_matched.py,
where displacement was set by hand and only the attack's shape varied.

    python3 scripts/plot_detectability_law.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

FAM_C = {"random": "#4477aa", "replay": "#cc3311", "stealthy": "#228833",
         "noise": "#ccbb44", "targeted": "#aa3377"}
SHAPE_C = {"bias": "#333333", "sinusoidal": "#0077bb",
           "slowdrift": "#ee7733", "swap": "#009988"}


def main():
    rows = json.load(open(RUNS / "detectability_law.json"))
    sweeps = {}
    for net, f in (("Modena", "gen_matched_modena.json"),
                   ("L-Town", "gen_matched_ltown.json")):
        p = RUNS / f
        if p.exists():
            sweeps[net] = json.load(open(p))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, key, title in (
            (axes[0], "z", "Label-free $|z|$ on the residual"),
            (axes[1], "sup", "Trained supervised head")):
        for net, sw in sweeps.items():
            ls = "-" if net == "Modena" else "--"
            byshape = {}
            for k, v in sw.items():
                shape, mag = k.split("@")
                byshape.setdefault(shape, []).append(
                    (float(mag), float(np.mean(v[key]))))
            for shape, pts in byshape.items():
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls,
                        color=SHAPE_C.get(shape, "grey"), lw=1.1, alpha=.55,
                        zorder=1)

        for r in rows:
            mk = "o" if "ltown" in r["dataset"] else "s"
            ax.scatter(max(r["disp"], 0.05), r[key], s=46, marker=mk,
                       color=FAM_C.get(r["family"], "grey"),
                       edgecolor="black", linewidth=.5, zorder=3)

        ax.axhline(0.5, color="black", lw=.7, ls=":", zorder=0)
        ax.axvspan(0.05, 1.0, color="black", alpha=.05, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("displacement  $|p_{\\mathrm{reported}}-p_{\\mathrm{true}}|$"
                      "  /  clean residual sd")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.15, 1.03)
        ax.grid(alpha=.25, lw=.5)
    axes[0].set_ylabel("AUROC")

    fam = [plt.Line2D([], [], marker="s", ls="", color=c, mec="k", mew=.5,
                      label=f) for f, c in FAM_C.items()]
    shp = [plt.Line2D([], [], color=c, lw=1.3, label=f"injected: {s}")
           for s, c in SHAPE_C.items()]
    net = [plt.Line2D([], [], marker="s", ls="", color="w", mec="k", label="Modena"),
           plt.Line2D([], [], marker="o", ls="", color="w", mec="k", label="L-Town")]
    axes[1].legend(handles=fam + net + shp, fontsize=7.2, loc="lower right",
                   ncol=2, framealpha=.95)

    d = np.array([r["disp"] for r in rows])
    rho = {k: float(np.corrcoef(np.argsort(np.argsort(d)),
                                np.argsort(np.argsort([r[k] for r in rows])))[0, 1])
           for k in ("z", "sup")}
    axes[0].text(.03, .95, f"Spearman $\\rho$ = {rho['z']:+.3f}",
                 transform=axes[0].transAxes, va="top", fontsize=9.5)
    axes[1].text(.03, .95, f"Spearman $\\rho$ = {rho['sup']:+.3f}",
                 transform=axes[1].transAxes, va="top", fontsize=9.5)

    fig.suptitle("Detection is set by displacement, not by attack family "
                 f"({len(rows)} dataset$\\times$family points, 2 networks, "
                 "4 regimes)", fontsize=11.5, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .95))
    out = ROOT / "figures" / "detectability_law.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=190)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
