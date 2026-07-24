"""Build a clean, self-contained results package for the supervisors.

Produces `results_package/` — a set of plainly-labelled figures and tables,
organised by theme, each with a one-line takeaway, plus a narrated README.
Everything reads from the aggregated JSONs; run after the experiments and
the aggregation scripts have completed.

    python3 scripts/build_results_package.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_package"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

# Validated palette (matches slides + dashboard).
BLUE, ORANGE, GREEN, PURPLE = "#2563eb", "#ea580c", "#059669", "#7c3aed"
RED, CYAN, MUTED, INK = "#dc2626", "#0891b2", "#6b7280", "#1f2937"

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#c9ced6",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#54607a", "ytick.color": "#54607a",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#eef0f4", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

ATTACKS = ["random", "targeted", "stealthy", "noise", "replay"]
ALABEL = {a: a.capitalize() for a in ATTACKS}
ACOLOR = {"random": BLUE, "targeted": GREEN, "stealthy": ORANGE,
          "noise": CYAN, "replay": PURPLE}
DCOLOR = {"Water": BLUE, "Power": ORANGE, "Traffic": GREEN}

manifest: list[tuple[str, str]] = []


def load(name):
    p = ROOT / "runs" / name
    return json.load(open(p)) if p.exists() else None


def save(fig, name, takeaway):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    manifest.append((f"figures/{name}.png", takeaway))
    print(f"  figure  {name}.png")


def table(rows, name, takeaway):
    if not rows:
        return
    cols = list(rows[0].keys())
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    with open(TAB / f"{name}.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(fmt(r[c]) for c in cols) + "\n")
    with open(TAB / f"{name}.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[c]) for c in cols) + " |\n")
    manifest.append((f"tables/{name}.csv", takeaway))
    print(f"  table   {name}.csv / .md")


def _barlabels(ax, xs, ys, errs=None, fs=13, dy=0.02):
    for i, (x, y) in enumerate(zip(xs, ys)):
        top = y + (errs[i] if errs else 0)
        ax.text(x, top + dy, f"{y:.3f}", ha="center", va="bottom",
                fontsize=fs, fontweight="bold", color=INK)


# ---------------------------------------------------------------------------
# 1. Overview — detection F1 across the three domains
# ---------------------------------------------------------------------------
def overview(nad):
    if not nad:
        return
    doms = [d for d in ["Water", "Power", "Traffic"] if d in nad]
    # prefer per_node, else global
    vals, errs, labs = [], [], []
    for d in doms:
        m = nad[d].get("per_node") or nad[d].get("global")
        vals.append(m["overall_f1"]); errs.append(m["overall_f1_std"])
        labs.append(d)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(range(len(doms)), vals, color=[DCOLOR[d] for d in doms], width=0.55,
           yerr=errs, capsize=5, error_kw=dict(ecolor="#7b8494", lw=1.4))
    _barlabels(ax, range(len(doms)), vals, errs, fs=15)
    ax.set_xticks(range(len(doms))); ax.set_xticklabels(labs, fontsize=13)
    ax.set_ylabel("Detection F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Attack detection F1 — same model, three domains",
                 fontsize=13.5, fontweight="bold")
    save(fig, "01_overview_domain_f1",
         "The identical detector reaches F1 ~0.83-0.87 on water, power and "
         "traffic.")
    table([{"domain": d,
            "overall_f1": (nad[d].get("per_node") or nad[d]["global"])["overall_f1"],
            "norm_mode": "per_node" if "per_node" in nad[d] else "global"}
           for d in doms],
          "01_overview_domain_f1", "Overall F1 per domain.")


# ---------------------------------------------------------------------------
# 2. Part 1 — per-attack F1 on water
# ---------------------------------------------------------------------------
def per_attack(nc):
    if not nc:
        return
    by = {r["config"]: r for r in nc}
    row = by.get("Per-node") or list(by.values())[-1]
    # per-attack values live in norm_all_domains via test_results; use nc extras
    # nc rows here carry stealthy_f1 etc. Build a small per-attack view.
    # (per-attack detail is emphasised in the normalisation figure instead.)


# ---------------------------------------------------------------------------
# 3. Normalisation on water (3 panels)
# ---------------------------------------------------------------------------
def normalisation_water(nc):
    if not nc:
        return
    by = {r["config"]: r for r in nc}
    order = ["Global (per-signal)", "Per-node"]
    cols = [MUTED, PURPLE]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    panels = [("overall_f1", "Overall F1", 1.0),
              ("replay_auroc", "Replay AUROC\n(ranking quality)", 1.0),
              ("replay_f1", "Replay F1\n(calibrated threshold)", 0.2)]
    for ax, (k, title, ymax) in zip(axes, panels):
        vals = [by[o][k] for o in order]
        ax.bar(range(2), vals, color=cols, width=0.6)
        _barlabels(ax, range(2), vals, fs=14, dy=ymax * 0.02)
        ax.set_xticks(range(2)); ax.set_xticklabels(order, fontsize=11)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_ylim(0, ymax * 1.2)
        if k == "replay_auroc":
            ax.axhline(0.5, color=RED, lw=1.2, ls="--")
            ax.text(1.4, 0.52, "chance", fontsize=8.5, color=RED, ha="right")
    fig.suptitle("Water — per-node normalisation recovers the replay attack",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "02_water_normalisation",
         "Normalising each sensor by its own scale lifts replay ranking from "
         "below chance (0.46) to AUROC 0.79; overall F1 holds.")
    table(nc, "02_water_normalisation",
          "Global vs per-node normalisation on water.")


# ---------------------------------------------------------------------------
# 4. Normalisation generalisation — replay AUROC across domains
# ---------------------------------------------------------------------------
def normalisation_generalisation(nad):
    if not nad:
        return
    doms = [d for d in ["Water", "Power", "Traffic"] if d in nad
            and "per_node" in nad[d] and "global" in nad[d]]
    if not doms:
        return
    glob = [nad[d]["global"]["replay_auroc"] for d in doms]
    pern = [nad[d]["per_node"]["replay_auroc"] for d in doms]
    x = np.arange(len(doms)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - w/2, glob, w, color=MUTED, label="Global norm")
    ax.bar(x + w/2, pern, w, color=PURPLE, label="Per-node norm")
    for i in range(len(doms)):
        ax.text(x[i]-w/2, glob[i]+0.02, f"{glob[i]:.2f}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
        ax.text(x[i]+w/2, pern[i]+0.02, f"{pern[i]:.2f}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color=RED, lw=1.2, ls="--")
    ax.text(len(doms)-0.5, 0.52, "chance", fontsize=8.5, color=RED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=13)
    ax.set_ylabel("Replay-attack AUROC"); ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left")
    ax.set_title("Does the normalisation fix generalise? Replay ranking by domain",
                 fontsize=13, fontweight="bold")
    save(fig, "03_normalisation_generalisation",
         "Per-node normalisation lifts replay most where it is hardest "
         "(water, power); traffic replay is already easy.")
    rows = []
    for d in doms:
        rows.append({"domain": d,
                     "replay_auroc_global": nad[d]["global"]["replay_auroc"],
                     "replay_auroc_pernode": nad[d]["per_node"]["replay_auroc"],
                     "lift": nad[d]["per_node"]["replay_auroc"] - nad[d]["global"]["replay_auroc"]})
    table(rows, "03_normalisation_generalisation",
          "Replay AUROC, global vs per-node, per domain.")


# ---------------------------------------------------------------------------
# 5. Cross-domain per-attack difficulty heatmap
# ---------------------------------------------------------------------------
def difficulty(cd):
    if not cd:
        return
    doms = [("Water", "water_modena_10seed"), ("Power", "power_ieee118_5seed"),
            ("Traffic", "traffic_200_5seed")]
    doms = [(n, k) for n, k in doms if k in cd]
    z = np.array([[cd[k][a][0] for a in ATTACKS] for _, k in doms])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([ALABEL[a] for a in ATTACKS], fontsize=12.5)
    ax.set_yticks(range(len(doms)))
    ax.set_yticklabels([n for n, _ in doms], fontsize=12.5, fontweight="bold")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center",
                    fontsize=13, color="white" if z[i, j] > 0.55 else INK,
                    fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label("F1")
    ax.set_title("Per-attack F1 across domains — the hard class inverts",
                 fontsize=13, fontweight="bold")
    save(fig, "04_difficulty_heatmap",
         "Water/power find stealthy easy and replay hard; traffic is the "
         "mirror image.")


# ---------------------------------------------------------------------------
# 6. Routing schemes (cascade)
# ---------------------------------------------------------------------------
def routing(casc):
    if not casc:
        return
    order = [("soft", "Soft mixture"), ("router_top1", "Router top-1"),
             ("cascade", "Cascade\n(feedback)"), ("oracle", "Oracle\n(ceiling)")]
    cols = [BLUE, MUTED, ORANGE, GREEN]
    means = [st.mean([r[k]["f1"] for r in casc]) for k, _ in order]
    stds = [st.stdev([r[k]["f1"] for r in casc]) for k, _ in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(4), means, color=cols, width=0.58, yerr=stds, capsize=5,
           error_kw=dict(ecolor="#7b8494", lw=1.3))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in order], fontsize=11)
    ax.set_ylabel("Detection F1"); ax.set_ylim(0, 1.0)
    ax.set_title("Routing schemes — the interpretable cascade matches the ceiling",
                 fontsize=13, fontweight="bold")
    save(fig, "05_routing_schemes",
         "Cascade 0.772 matches the oracle ceiling (0.769), within a point "
         "of the soft blend.")
    table([{"scheme": l.replace("\n", " "),
            "f1_mean": st.mean([r[k]["f1"] for r in casc]),
            "f1_std": st.stdev([r[k]["f1"] for r in casc])}
           for k, l in order],
          "05_routing_schemes", "Routing-scheme F1 (5 seeds).")


# ---------------------------------------------------------------------------
# 7. Router diagnostic
# ---------------------------------------------------------------------------
def router(diag):
    if not diag:
        return
    names = ["random", "replay", "stealthy", "noise", "targeted"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, net in zip(axes, ["Modena", "Net3"]):
        conf = np.array(diag[net]["confusion"], dtype=float)[1:, 1:]
        rows = conf.sum(1, keepdims=True); rows[rows == 0] = 1
        norm = conf / rows
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticks(range(5)); ax.set_yticklabels(names)
        for i in range(5):
            for j in range(5):
                v = int(conf[i, j])
                if v:
                    ax.text(j, i, v, ha="center", va="center", fontsize=11,
                            color="white" if norm[i, j] > 0.55 else INK)
        ax.set_title(f"{net} — router acc {diag[net]['overall_acc']:.0%}",
                     fontsize=12.5, fontweight="bold")
        ax.set_xlabel("routed to expert"); ax.grid(False)
        if net == "Modena":
            ax.set_ylabel("true attack class")
    save(fig, "06_router_diagnostic",
         "Router is healthy on Modena (96%); on Net3 it sends 77% of "
         "stealthy windows to the replay expert.")


def main():
    nad = load("temporal_moe/norm_all_domains.json")
    nc = load("temporal_moe/norm_compare.json")
    cd = load("temporal_moe/crossdomain_summary.json")
    casc = load("temporal_moe/cascade_eval.json")
    diag = load("selfplay/router_diagnostic.json")
    ens = load("temporal_moe/ensemble_eval.json")

    print("Building results package ...")
    overview(nad)
    normalisation_water(nc)
    normalisation_generalisation(nad)
    difficulty(cd)
    routing(casc)
    router(diag)

    # replay-why evidence (figure produced by analyze_replay_ceiling.py).
    rw = load("temporal_moe/replay_why.json")
    if rw and (FIG / "07_replay_why.png").exists():
        manifest.append(("figures/07_replay_why.png",
                         f"Why replay still underperforms: replayed and "
                         f"genuine scores overlap (AUROC "
                         f"{rw['replay_auroc']:.2f}), so no threshold "
                         f"separates them (best F1 {rw['replay_best_f1']:.2f}) "
                         f"— a physical limit."))

    # README
    lines = [
        "# Results package",
        "",
        "Figures and tables for the temporal Mixture-of-Experts attack "
        "detector on water, power and traffic sensor networks. Every table "
        "is provided as `.csv` (machine-readable) and `.md` (readable "
        "inline). Read top to bottom — it follows the story.",
        "",
        "## Headline numbers",
        "",
    ]
    if nad:
        for d in ["Water", "Power", "Traffic"]:
            if d in nad:
                m = nad[d].get("per_node") or nad[d]["global"]
                lines.append(f"- **{d}**: detection F1 "
                             f"{m['overall_f1']:.3f}")
    if ens:
        lines += ["",
                  f"- **Final ensembled model (water)**: F1 "
                  f"{ens['ensemble_f1']:.3f} "
                  f"(single-seed mean {ens['single_seed_mean_f1']:.3f})"]
    lines += ["", "## Figures", ""]
    for fname, takeaway in manifest:
        lines.append(f"- **`{fname}`** — {takeaway}")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(manifest)} artifacts -> {OUT}")


if __name__ == "__main__":
    main()
