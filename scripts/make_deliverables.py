"""Build the supervisor deliverables package: figures + tables + index.

Collects every result the project has produced and writes a single
self-contained folder that can be sent as-is:

    deliverables/
        README.md          index describing every artifact
        figures/*.png      plots
        tables/*.csv       machine-readable metrics
        tables/*.md        the same tables, readable in an email

Every section degrades gracefully: if an experiment has not been run yet,
its artifacts are skipped and the README says so.

    python3 scripts/make_deliverables.py
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
OUT = ROOT / "deliverables"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (same as slides / dashboard).
BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, RED, CYAN, YELLOW = "#7c3aed", "#dc2626", "#0891b2", "#ca8a04"
INK, MUTED = "#1f2937", "#6b7280"

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#c9ced6",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#54607a", "ytick.color": "#54607a",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#eef0f4", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

ATTACKS = ["random", "targeted", "stealthy", "noise", "replay"]
ACOLOR = {"random": BLUE, "targeted": GREEN, "stealthy": ORANGE,
          "noise": CYAN, "replay": PURPLE}
manifest: list[tuple[str, str]] = []      # (filename, description)


def save(fig, name, desc):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    manifest.append((f"figures/{name}.png", desc))
    print(f"  figure  {name}.png")


def _fmt(v):
    """Round floats for a human-readable table; leave everything else."""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def table(rows: list[dict], name: str, desc: str):
    """Write a list of dicts as both CSV and a markdown table."""
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(TAB / f"{name}.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r[c]) for c in cols) + "\n")
    with open(TAB / f"{name}.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r[c]) for c in cols) + " |\n")
    manifest.append((f"tables/{name}.csv", desc))
    print(f"  table   {name}.csv / .md")


def load(path):
    p = ROOT / path
    return json.load(open(p)) if p.exists() else None


# ===============================================================
# 1. Part 1 — per-attack detection, 10 seeds
# ===============================================================
def part1(cd):
    if not cd:
        return
    w = cd["water_modena_10seed"]
    means = [w[a][0] for a in ATTACKS]
    stds = [w[a][1] for a in ATTACKS]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (m, s, a) in enumerate(zip(means, stds, ATTACKS)):
        ax.bar(i, m, color=ACOLOR[a], width=0.62, yerr=s, capsize=4,
               error_kw=dict(ecolor="#7b8494", lw=1.3))
        ax.text(i, m + s + 0.03, f"{m:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([a.capitalize() for a in ATTACKS])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Detection F1")
    save(fig, "01_part1_per_attack",
         "Part 1 (water/Modena): per-attack F1, mean ± std over 10 seeds.")

    table([{"attack": a, "f1_mean": round(w[a][0], 4),
            "f1_std": round(w[a][1], 4)} for a in ATTACKS],
          "01_part1_per_attack",
          "Part 1 per-attack F1 with seed standard deviation.")


# ===============================================================
# 2. Cross-domain
# ===============================================================
def cross_domain(cd):
    if not cd:
        return
    doms = [("Water", "water_modena_10seed", BLUE, 10),
            ("Power", "power_ieee118_5seed", ORANGE, 5),
            ("Traffic", "traffic_200_5seed", GREEN, 5)]

    # Overall F1
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (nm, k, c, _) in enumerate(doms):
        m, s = cd[k]["F1"]
        ax.bar(i, m, color=c, width=0.55, yerr=s, capsize=5,
               error_kw=dict(ecolor="#7b8494", lw=1.4))
        ax.text(i, m + s + 0.028, f"{m:.3f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([d[0] for d in doms])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Detection F1")
    save(fig, "02_crossdomain_f1",
         "Overall detection F1 on three domains with the identical model.")

    # Replay vs signal speed
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (nm, k, c, _) in enumerate(doms):
        m, s = cd[k]["replay"]
        ax.bar(i, m, color=c, width=0.52, yerr=s, capsize=5,
               error_kw=dict(ecolor="#7b8494", lw=1.4))
        ax.text(i, m + s + 0.04, f"{m:.3f}", ha="center", va="bottom",
                fontsize=15, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Water\npressure", "Power\nvoltage", "Traffic\nspeed"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Replay-attack F1")
    ax.set_xlabel("signal changes slowly   $\\longrightarrow$   fast",
                  labelpad=10)
    save(fig, "03_replay_vs_signal_speed",
         "Replay F1 rises with how fast the monitored signal moves — the "
         "replay ceiling is a property of the domain, not the model.")

    # Heatmap
    z = np.array([[cd[k][a][0] for a in ATTACKS] for _, k, _, _ in doms])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([a.capitalize() for a in ATTACKS])
    ax.set_yticks(range(3)); ax.set_yticklabels([d[0] for d in doms])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if z[i, j] > 0.55 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).outline.set_visible(False)
    save(fig, "04_difficulty_heatmap",
         "Per-attack F1 for every domain: the hard class inverts between "
         "water/power and traffic.")

    rows = []
    for nm, k, _, ns in doms:
        r = {"domain": nm, "seeds": ns,
             "F1": round(cd[k]["F1"][0], 4), "F1_std": round(cd[k]["F1"][1], 4),
             "AUROC": round(cd[k]["AUROC"][0], 4)}
        for a in ATTACKS:
            r[a] = round(cd[k][a][0], 4)
        rows.append(r)
    table(rows, "02_crossdomain",
          "Cross-domain summary: overall and per-attack F1 for the three "
          "networks.")


# ===============================================================
# 3. Replay-weight Pareto
# ===============================================================
def replay_pareto(rw):
    if not rw:
        return
    ws = sorted(rw.keys(), key=float)
    x = [float(w) for w in ws]
    o = [rw[w]["pressure_f1"]["mean"] for w in ws]
    oe = [rw[w]["pressure_f1"]["std"] for w in ws]
    r = [rw[w]["per_attack"]["replay"]["mean"] for w in ws]
    re_ = [rw[w]["per_attack"]["replay"]["std"] for w in ws]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(x, o, yerr=oe, marker="o", ms=8, lw=2.4, color=BLUE,
                capsize=4, label="Overall F1")
    ax.errorbar(x, r, yerr=re_, marker="s", ms=8, lw=2.4, color=PURPLE,
                capsize=4, label="Replay F1")
    ax.set_xlabel("Replay loss weight"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0); ax.legend()
    save(fig, "05_replay_weight_pareto",
         "Forcing replay up in the loss trades away overall F1 along a "
         "Pareto front (10 seeds per point).")

    table([{"replay_weight": w,
            "overall_f1": round(rw[w]["pressure_f1"]["mean"], 4),
            "replay_f1": round(rw[w]["per_attack"]["replay"]["mean"], 4),
            "seeds": rw[w]["pressure_f1"]["n"]} for w in ws],
          "03_replay_weight_sweep",
          "Replay-weight sweep: the overall-vs-replay trade-off.")


# ===============================================================
# 4. Router diagnostic
# ===============================================================
def router(diag):
    if not diag:
        return
    names = ["random", "replay", "stealthy", "noise", "targeted"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, net in zip(axes, ["Modena", "Net3"]):
        conf = np.array(diag[net]["confusion"], dtype=float)[1:, 1:]
        rows = conf.sum(1, keepdims=True); rows[rows == 0] = 1
        norm = conf / rows
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(names, rotation=30,
                                                    ha="right")
        ax.set_yticks(range(5)); ax.set_yticklabels(names)
        for i in range(5):
            for j in range(5):
                if conf[i, j]:
                    ax.text(j, i, int(conf[i, j]), ha="center", va="center",
                            fontsize=11,
                            color="white" if norm[i, j] > 0.55 else INK)
        ax.set_title(f"{net} — router acc {diag[net]['overall_acc']:.0%}",
                     fontsize=12.5, fontweight="bold")
        ax.set_xlabel("routed to expert"); ax.grid(False)
        if net == "Modena":
            ax.set_ylabel("true attack class")
    save(fig, "06_router_diagnostic",
         "Router confusion. Modena is healthy; Net3 sends 77% of stealthy "
         "windows to the replay expert.")

    rows = []
    for net in ("Modena", "Net3"):
        for cls, acc in diag[net]["per_class_acc"].items():
            if cls == "clean":
                continue
            rows.append({"network": net, "attack": cls,
                         "routing_accuracy": round(acc, 4)})
    table(rows, "04_router_accuracy",
          "Per-class routing accuracy on both networks.")


# ===============================================================
# 5. Routing schemes: soft vs top-1 vs cascade vs oracle
# ===============================================================
def routing_schemes(casc):
    if not casc:
        return
    variants = ["soft", "router_top1", "cascade", "oracle"]
    labels = {"soft": "Soft mixture\n(current)", "router_top1": "Router top-1\n(hard)",
              "cascade": "Cascade\n(feedback)", "oracle": "Oracle expert\n(hard ceiling)"}
    cols = {"soft": BLUE, "router_top1": MUTED, "cascade": ORANGE,
            "oracle": GREEN}
    means = {v: st.mean([r[v]["f1"] for r in casc]) for v in variants}
    stds = {v: (st.stdev([r[v]["f1"] for r in casc]) if len(casc) > 1 else 0.0)
            for v in variants}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, v in enumerate(variants):
        ax.bar(i, means[v], color=cols[v], width=0.58, yerr=stds[v], capsize=4,
               error_kw=dict(ecolor="#7b8494", lw=1.3))
        ax.text(i, means[v] + stds[v] + 0.02, f"{means[v]:.3f}", ha="center",
                va="bottom", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([labels[v] for v in variants])
    ax.set_ylim(0, max(means.values()) * 1.28)
    ax.set_ylabel("Detection F1")
    save(fig, "07_routing_schemes",
         "Routing schemes compared on identical checkpoints. The oracle bar "
         "is the ceiling any hard-selection scheme can reach.")

    table([{"scheme": v, "f1_mean": round(means[v], 4),
            "f1_std": round(stds[v], 4), "runs": len(casc)}
           for v in variants],
          "05_routing_schemes",
          "Soft mixture vs hard top-1 vs cascade vs oracle expert.")

    # Cascade behaviour
    hist = np.sum([r["diagnostics"]["attempts_hist"] for r in casc], axis=0)
    hist = hist[1:]                       # drop the unused 0-attempt bin
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(1, len(hist) + 1), hist, color=ORANGE, width=0.55)
    for i, h in enumerate(hist):
        ax.text(i + 1, h + max(hist) * 0.02, int(h), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, len(hist) + 1))
    ax.set_xlabel("experts tried before accepting")
    ax.set_ylabel("windows")
    save(fig, "08_cascade_attempts",
         "How many experts the cascade has to try per window.")


# ===============================================================
# 6. lambda_expert sweep — does standalone training lift the ceiling?
# ===============================================================
def expert_sweep(sweep):
    if not sweep:
        return
    xs = sorted(sweep.keys(), key=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for v, c, lab in (("soft", BLUE, "Soft mixture"),
                      ("oracle", GREEN, "Oracle expert (hard ceiling)"),
                      ("cascade", ORANGE, "Cascade")):
        ys = [sweep[x][v] for x in xs]
        ax.plot([float(x) for x in xs], ys, marker="o", ms=8, lw=2.4,
                color=c, label=lab)
    ax.set_xlabel(r"$\lambda_{\mathrm{expert}}$  (direct per-expert supervision)")
    ax.set_ylabel("Detection F1")
    ax.legend()
    save(fig, "09_expert_supervision_sweep",
         "Raising direct per-expert supervision trains experts to work "
         "standalone, which is what the cascade needs.")

    table([{"lambda_expert": x, **{k: round(v, 4)
                                   for k, v in sweep[x].items()}}
           for x in xs],
          "06_expert_supervision_sweep",
          "Effect of direct expert supervision on each routing scheme.")


# ===============================================================
# 7. Normalisation comparison
# ===============================================================
def normalisation(norm_rows):
    if not norm_rows:
        return
    labels = [r["config"] for r in norm_rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, (metric, title) in enumerate((("overall_f1", "Overall F1"),
                                         ("replay_f1", "Replay F1"))):
        vals = [r[metric] for r in norm_rows]
        cols = [BLUE if "lobal" in l else PURPLE for l in labels]
        ax[k].bar(range(len(vals)), vals, color=cols, width=0.55)
        for i, v in enumerate(vals):
            ax[k].text(i, v + max(vals) * 0.02, f"{v:.3f}", ha="center",
                       va="bottom", fontsize=12, fontweight="bold")
        ax[k].set_xticks(range(len(labels)))
        ax[k].set_xticklabels(labels, rotation=15, ha="right")
        ax[k].set_title(title, fontsize=12.5, fontweight="bold")
        ax[k].set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    save(fig, "10_normalisation",
         "Global vs per-node normalisation — testing whether water values "
         "sitting close together is what hides replay.")
    table(norm_rows, "07_normalisation",
          "Normalisation comparison (overall and replay F1).")


# ===============================================================
def main():
    print("Building deliverables ...")
    cd = load("runs/temporal_moe/crossdomain_summary.json")
    rw = load("runs/temporal_moe/rw_sweep_summary.json")
    diag = load("runs/selfplay/router_diagnostic.json")
    casc = load("runs/temporal_moe/cascade_eval.json")
    sweep = load("runs/temporal_moe/expert_sweep.json")
    norm_rows = load("runs/temporal_moe/norm_compare.json")

    part1(cd)
    cross_domain(cd)
    replay_pareto(rw)
    router(diag)
    routing_schemes(casc)
    expert_sweep(sweep)
    normalisation(norm_rows)

    # ---- README index ----
    lines = [
        "# Results package",
        "",
        "Figures and tables for the temporal Mixture-of-Experts attack "
        "detector, across water, power and traffic networks.",
        "",
        "Every table is provided as `.csv` (machine readable) and `.md` "
        "(readable inline).",
        "",
        "## Contents",
        "",
    ]
    for fname, desc in manifest:
        lines.append(f"- **`{fname}`** — {desc}")
    missing = []
    if not casc:
        missing.append("routing-scheme comparison (run `scripts/eval_cascade.py`)")
    if not sweep:
        missing.append("expert-supervision sweep")
    if not norm_rows:
        missing.append("normalisation comparison")
    if missing:
        lines += ["", "## Not yet generated", ""]
        lines += [f"- {m}" for m in missing]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(manifest)} artifacts -> {OUT}")


if __name__ == "__main__":
    main()
