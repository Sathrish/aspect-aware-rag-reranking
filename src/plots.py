"""Figures for the README. Colour follows the system, in a fixed order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e6e5e1"
COLORS = {"BM25": "#2a78d6", "Dense": "#eb6834", "CrossEncoder": "#1baf7a", "Aspect-Aware": "#eda100"}


def _color(system):
    return COLORS[system.split(" + ")[-1]]


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


def plot_retrieval(summary, path, metrics=("NDCG@5", "P@5", "MRR", "ACS")):
    """One panel per metric; bars grouped by first-stage retriever."""
    n_q = int(summary["Queries"].iloc[0])
    values = summary.set_index("System")
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.2), facecolor=SURFACE)
    width = 0.26
    for ax, metric in zip(axes, metrics):
        _style(ax)
        for g, stage in enumerate(("BM25", "Dense")):
            for j, name in enumerate((stage, f"{stage} + CrossEncoder", f"{stage} + Aspect-Aware")):
                x, v = g + (j - 1) * (width + 0.02), values.loc[name, metric]
                ax.bar(x, v, width, color=_color(name))
                ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks([0, 1], ["BM25 top-20 pool", "Dense top-20 pool"])
        ax.set_ylim(0, 1)
        ax.set_title(metric, fontsize=12, fontweight="bold", color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    labels = ["BM25 (first stage only)", "Dense (first stage only)", "+ CrossEncoder", "+ Aspect-Aware"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Retrieval and re-ranking on NFCorpus (test split, {n_q} queries, top-5)",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def plot_lambda(sweep, path, metrics=("NDCG@5", "ACS", "Diversity")):
    """Small multiples (one y-axis each) instead of a dual-axis chart."""
    n_q = int(sweep["Queries"].iloc[0])
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 3.8), facecolor=SURFACE)
    for ax, metric in zip(axes, metrics):
        _style(ax)
        ax.plot(sweep["lambda"], sweep[metric], "-o", color=COLORS["Aspect-Aware"],
                linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        lo, hi = sweep[metric].min(), sweep[metric].max()
        pad = max((hi - lo) * 0.25, 0.005)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(list(sweep["lambda"]))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("λ  (0 = aspect coverage only, 1 = relevance only)", fontsize=9, color=MUTED)
        ax.set_title(metric, fontsize=12, fontweight="bold", color=INK, loc="left")
    fig.suptitle(f"Aspect-Aware re-ranker: sensitivity to λ ({n_q} queries; y-axes are zoomed)",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def plot_rag(scores, path):
    """Left: mean ROUGE-L per system. Right: per-query scores as a dot plot."""
    systems = [c for c in scores.columns if c != "Query"]
    means = scores[systems].mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1, 1.5]})
    _style(a)
    for i, s in enumerate(systems):
        a.bar(i, means[s], 0.55, color=_color(s))
        a.text(i, means[s] + 0.003, f"{means[s]:.4f}", ha="center", fontsize=10, color=INK)
    a.set_xticks(range(len(systems)), [s.replace(" + ", "\n+ ") for s in systems])
    a.set_ylim(0, max(means.max() * 1.3, 0.05))
    a.set_title(f"Mean ROUGE-L F1 ({len(scores)} queries)", fontsize=12, fontweight="bold",
                color=INK, loc="left")

    _style(b, grid_axis="x")
    y = np.arange(len(scores))[::-1]
    offsets = np.linspace(0.27, -0.27, len(systems))
    for s, off in zip(systems, offsets):
        b.scatter(scores[s], y + off, s=42, color=_color(s), edgecolor=SURFACE, linewidth=1,
                  label=s, zorder=3)
    for yy in y[:-1]:
        b.axhline(yy - 0.5, color=GRID, linewidth=0.6)
    b.set_yticks(y, [q if len(q) <= 46 else q[:44] + "…" for q in scores["Query"]])
    b.tick_params(axis="y", labelsize=8.5)
    b.set_xlim(-0.02, max(scores[systems].max().max() * 1.08, 0.1))
    b.set_xlabel("ROUGE-L F1", fontsize=9, color=MUTED)
    b.set_title("Per-query ROUGE-L F1", fontsize=12, fontweight="bold", color=INK, loc="left")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(systems), frameon=False,
             fontsize=9)
    fig.suptitle("End-to-end RAG with flan-t5-base", fontsize=13, fontweight="bold",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)
