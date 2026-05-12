import os

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "figure.dpi": 160,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    plt.close(fig)


def diagnostic_bar():
    labels = ["EEF DP\n(OSC)", "Joint-delta DP\n(no adapter)", "Joint-delta DP\n+ adapter\n(ours)"]
    values = [49 / 50, 0.0, 50 / 50]
    colors = ["#33658a", "#b75d69", "#2f855a"]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Can MH test success rate")
    ax.set_title("The failure is the action interface, not only the policy")
    ax.grid(axis="x", visible=False)
    for bar, text in zip(bars, ["49/50", "0/50", "50/50"]):
        y = max(bar.get_height(), 0.02)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.035,
            text,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    save(fig, "fig2_can_diagnostic.png")


def full_pipeline_bar():
    tasks = ["Can\nMH", "Can\nPH", "Lift\nPH", "Lift\nMH", "Square\nPH", "Transport\nPH"]
    ours = np.array([50 / 50, 45 / 50, 50 / 50, 49 / 50, 27 / 50, 0 / 50])
    # Diffusion Policy paper Table I, state policy, DiffusionPolicy-C row.
    # Values are the reported max success rates from max / average-last-10.
    paper_dp = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 0.94])
    counts = ["50/50", "45/50", "50/50", "49/50", "27/50", "0/50"]
    x = np.arange(len(tasks))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ours_bars = ax.bar(
        x - width / 2, ours, color="#2f855a", width=width,
        label="Joint-delta DP + adapter (ours)")
    paper_bars = ax.bar(
        x + width / 2, paper_dp, color="#33658a", width=width,
        label="DP paper, state policy")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test success rate")
    ax.set_title("Full joint-delta pipeline versus original DP paper")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.grid(axis="x", visible=False)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5df",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    save(fig, "fig3_full_pipeline.png")


def oracle_replay():
    tasks = ["Can\nMH", "Can\nPH", "Lift\nPH", "Lift\nMH", "Square\nPH",
             "Square\nMH", "Tool\nHang", "Transport\nPH", "Transport\nMH"]
    success = np.array([38, 43, 46, 43, 29, 30, 2, 17, 0]) / 50
    delta_mae = np.array([0.019667, 0.033875, 0.008774, 0.011748, 0.036040,
                          0.013600, 0.032994, 0.017372, 0.169473])
    x = np.arange(len(tasks))
    fig, ax1 = plt.subplots(figsize=(8.6, 3.9))
    ax1.bar(x, success, color="#33658a", width=0.62, label="Success")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Oracle replay success")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.grid(axis="x", visible=False)
    ax2 = ax1.twinx()
    ax2.plot(x, delta_mae, color="#b75d69", marker="o", linewidth=2.2, label="Delta MAE")
    ax2.set_ylim(0, 0.18)
    ax2.set_ylabel("Mean |actual delta - desired delta|")
    ax1.set_title("Adapter replay quality on held-out demonstrations")
    handles = [ax1.patches[0], ax2.lines[0]]
    ax1.legend(
        handles,
        ["Success", "Delta MAE"],
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5df",
        loc="upper right",
    )
    save(fig, "fig4_oracle_replay.png")


def closed_loop_heatmap():
    tasks = ["Can MH", "Can PH", "Lift PH", "Lift MH", "Square PH", "Square MH", "ToolHang PH", "Transport PH"]
    data = np.array([
        [50, 40, 12, 9, 11, 6, 5, 10],
        [45, 31, 2, 8, 4, 2, 1, 1],
        [50, 47, 31, 32, 6, 5, 9, 14],
        [49, 50, 46, 49, 44, 41, 42, 38],
        [24, 16, np.nan, 0, 2, 2, 0, 3],
        [31, 26, np.nan, 9, 4, 3, 0, 0],
        [np.nan, np.nan, np.nan, 0, 0, 0, 0, 0],
        [0, 2, np.nan, 0, 0, 0, np.nan, 0],
    ], dtype=float) / 50.0
    masked = np.ma.masked_invalid(data)
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad("#e9e9e9")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([str(i) for i in range(1, 9)])
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Closed-loop inner steps k")
    ax.set_title("Closed-loop execution often hurts task success")
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                label = "pending"
                color = "#555555"
            else:
                label = f"{int(round(data[i, j] * 50))}"
                color = "black" if data[i, j] < 0.65 else "white"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Success rate")
    save(fig, "fig5_closed_loop.png")


if __name__ == "__main__":
    diagnostic_bar()
    full_pipeline_bar()
    oracle_replay()
    closed_loop_heatmap()
