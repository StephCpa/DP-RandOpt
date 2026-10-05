"""Figures for docs/REPORT.md from the CSVs in results/ (static PNGs).

Palette: the dataviz skill's reference categorical slots in fixed order,
validated (light surface): blue, orange, aqua, yellow, magenta. Three slots sit
below 3:1 contrast, so every series also gets a distinct marker and a direct
end label, and the numbers are tabulated in the report (table view).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "docs", "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})

TITLES = {"countdown_like": "Countdown-like (general skill)",
          "countdown_specialist": "Countdown-like (specialists)",
          "gsm8k_like": "GSM8K-like (near ceiling)"}


def load_e2():
    d = pd.concat([pd.read_csv(os.path.join(RES, "e2_privacy_utility.csv")),
                   pd.read_csv(os.path.join(RES, "e2_privacy_utility_specialist.csv"))])
    r = pd.concat([pd.read_csv(os.path.join(RES, "e2_references.csv")),
                   pd.read_csv(os.path.join(RES, "e2_references_specialist.csv"))])
    return d, r


def line(ax, x, y, i, label, end_label=True):
    ax.plot(x, y, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6.5, mec=SURFACE, mew=1.5,
            solid_capstyle="round", label=label, zorder=3)
    if end_label:
        ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)


def ref(ax, y, text, style):
    ax.axhline(y, color=INK2, lw=1, ls=style, zorder=1)
    ax.annotate(text, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=INK2)


def fig_privacy_utility(d, r, k=10):
    scen = ["countdown_like", "countdown_specialist", "gsm8k_like"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)
    designs = [("D1", "D1 Gumbel top-K"), ("D0_pre100", "D0 noisy scores, public pre-screen"),
               ("D2", "D2 sharded"), ("D3_s1.0", "D3 one pick + neighbours"), ("D7", "D7 subset EM")]
    for row, n in enumerate((200, 1000)):
        for col, s in enumerate(scen):
            ax = axes[row, col]
            x = d[(d.scenario == s) & (d.n == n) & (d.k == k)]
            for i, (name, lab) in enumerate(designs):
                y = x[x.design == name].sort_values("eps")
                line(ax, y.eps.values, 100 * y.acc.values, i, lab, end_label=False)
            rr = r[(r.scenario == s) & (r.n == n) & (r.k == k)].set_index("design").acc
            ax.axhline(100 * rr["nonprivate"], color=INK2, lw=1.2, ls="-", zorder=1,
                       label=f"plain RandOpt, K={k} (not private)")
            ax.axhline(100 * rr["public_q50"], color=INK2, lw=1.2, ls=":", zorder=1,
                       label="public-only RandOpt, 50 examples")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8])
            ax.set_xticklabels(["1", "2", "4", "8"])
            ax.set_title(f"{TITLES[s]}, n = {n}  (base {100 * x.base.iloc[0]:.0f}%)", loc="left", fontsize=10)
            if row == 1:
                ax.set_xlabel("privacy budget ε (δ′ = 1e-5)")
            if col == 0:
                ax.set_ylabel("test accuracy of the vote (%)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Simulated privacy-utility of the ensemble designs, K = {k}", y=1.08, x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "privacy_utility.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_k_curve(d, r, n=1000, eps=4):
    scen = ["countdown_like", "countdown_specialist", "gsm8k_like"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, s in zip(axes, scen):
        rr = r[(r.scenario == s) & (r.n == n)]
        x = d[(d.scenario == s) & (d.n == n) & (d.eps == eps)]
        ks = [1, 5, 10, 50]
        series = [("plain RandOpt", [rr[(rr.k == k) & (rr.design == "nonprivate")].acc.iloc[0] for k in ks]),
                  ("D1, ε=4", [x[(x.k == k) & (x.design == "D1")].acc.iloc[0] for k in ks]),
                  ("D3, ε=4", [x[(x.k == 1) & (x.design == "D1")].acc.iloc[0]]
                   + [x[(x.k == k) & (x.design == "D3_s1.0")].acc.iloc[0] for k in ks[1:]]),
                  ("public-only (50)", [rr[(rr.k == k) & (rr.design == "public_q50")].acc.iloc[0] for k in ks])]
        for i, (lab, y) in enumerate(series):
            line(ax, np.array(ks), 100 * np.array(y), i, lab, end_label=False)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(TITLES[s], loc="left")
        ax.set_xlabel("ensemble size K")
    axes[0].set_ylabel("test accuracy of the vote (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Ensemble size (N = 2000, n = {n}): the simulated vote gain peaks at K = 5-10",
                 y=1.1, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "k_curve.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_temperature():
    t = pd.read_csv(os.path.join(RES, "e3_temperature.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, s in zip(axes, ["countdown_like", "gsm8k_like"]):
        x = t[(t.scenario == s) & (t.N == 5000)]
        for i, n in enumerate([50, 200, 1000]):
            y = x[x.n == n].sort_values("eps0")
            e = y.eps0.replace(np.inf, 64).values
            line(ax, e, 100 * y.test_acc.values, i, f"n = {n} private examples", end_label=False)
            star = y.gibbs_eps0_star.iloc[0]
            ax.axvline(star, color=SERIES[i], lw=1, ls=":", zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 1, 4, 16, 64])
        ax.set_xticklabels(["0.25", "1", "4", "16", "argmax"])
        ax.set_xlim(0.2, 120)
        ax.set_title(TITLES[s] + ", N = 5000, K = 1", loc="left")
        ax.set_xlabel("selection ε₀ (dotted: worst-case optimum, Cor. 6)")
    axes[0].set_ylabel("expected test accuracy of the pick (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Softer selection does not beat argmax on held-out accuracy",
                 y=1.08, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "temperature.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d, r = load_e2()
    fig_privacy_utility(d, r)
    fig_k_curve(d, r)
    fig_temperature()
    print("figures written to", FIG)
