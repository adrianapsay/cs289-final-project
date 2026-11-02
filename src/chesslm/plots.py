"""Figures for comparing runs. One shared style so every figure in the report looks the same."""

import matplotlib.pyplot as plt

# Colors follow the thing shown, in a fixed order: first run blue, second orange, third green, fourth yellow.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BANDS = ["<1200", "1200-1800", "1800-2400", "2400+"]


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def _label_bars(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height), (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=8, color=TEXT, xytext=(0, 2), textcoords="offset points")


def histogram(values, xlabel, path, edges=None):
    """Distribution of one quantity (e.g. ratings). Optional dashed lines mark boundaries such as rating bands."""
    fig, ax = plt.subplots(figsize=(7, 3), facecolor=SURFACE)
    ax.hist(values, bins=40, color=COLORS[0], edgecolor=SURFACE, linewidth=1)
    for edge in edges or []:
        ax.axvline(edge, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel(xlabel, color=MUTED)
    _style(ax, "count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def compare_metrics(summary, runs, path):
    """Three panels (accuracy, legal-move rate, ACPL), one bar per run. Each panel has its own axis."""
    panels = [("puzzle_accuracy", "puzzle accuracy (%)", 100, "{:.1f}"),
              ("legal_rate", "legal-move rate (%)", 100, "{:.1f}"),
              ("acpl", "ACPL (centipawns, lower is better)", 1, "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), facecolor=SURFACE)
    for ax, (column, label, scale, fmt) in zip(axes, panels):
        values = [summary.loc[summary.run == run, column].iloc[0] * scale for run in runs]
        bars = ax.bar(runs, values, color=COLORS[: len(runs)], edgecolor=SURFACE, linewidth=2)
        _label_bars(ax, bars, fmt)
        _style(ax, label)
        ax.set_ylim(bottom=0)  # bars always start at zero
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def accuracy_by_band(summary, runs, path):
    """Puzzle accuracy per rating band, one bar per run in each band."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        row = summary[summary.run == run].iloc[0]
        values = [row[f"accuracy_{band}"] * 100 for band in BANDS]
        positions = [b + (i - (len(runs) - 1) / 2) * width for b in range(len(BANDS))]
        bars = ax.bar(positions, values, width, label=run, color=COLORS[i], edgecolor=SURFACE, linewidth=2)
        _label_bars(ax, bars, "{:.1f}")
    ax.set_xticks(range(len(BANDS)), BANDS)
    ax.set_xlabel("puzzle rating band", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, "puzzle accuracy (%)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def outcome_breakdown(summary, runs, path):
    """For each run, the share of answers that were legal, illegal, or unparseable (stacked to 100%)."""
    outcomes = [("legal_rate", "legal"), ("illegal_rate", "illegal"), ("unparseable_rate", "unparseable")]
    fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=SURFACE)
    left = [0.0] * len(runs)
    for color, (column, label) in zip(COLORS, outcomes):
        values = [summary.loc[summary.run == run, column].iloc[0] * 100 for run in runs]
        ax.barh(runs, values, left=left, label=label, color=color, edgecolor=SURFACE, linewidth=2)
        for y, (start, value) in enumerate(zip(left, values)):
            if value >= 6:  # label segments wide enough to hold text
                ax.text(start + value / 2, y, f"{value:.0f}%", ha="center", va="center", fontsize=8, color=TEXT)
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of answers (%)", color=MUTED)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax, "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def loss_curve(log, path):
    """Training and validation loss over training steps."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    train = log.dropna(subset=["loss"])
    ax.plot(train.step, train.loss, color=COLORS[0], linewidth=2, label="training loss")
    if "eval_loss" in log:  # short runs may end before the first validation check
        val = log.dropna(subset=["eval_loss"])
        ax.plot(val.step, val.eval_loss, color=COLORS[1], linewidth=2, marker="o", markersize=8, label="validation loss")
    ax.set_xlabel("training step", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, "loss")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def reward_curve(log, path):
    """Average reward of the sampled answers over RL training steps (one series, so no legend)."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    rows = log.dropna(subset=["reward"])
    ax.plot(rows.step, rows.reward, color=COLORS[0], linewidth=2)
    ax.set_xlabel("training step", color=MUTED)
    _style(ax, "average reward of sampled answers")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig
