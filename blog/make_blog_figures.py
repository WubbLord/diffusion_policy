import os
import base64
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BLOG_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT_DIR, exist_ok=True)
INLINE_IMAGE_NAMES = [
    "fig2_can_diagnostic.png",
    "fig3_full_pipeline.png",
    "fig4_oracle_replay.png",
    "fig5_closed_loop.png",
]

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


def inline_assets():
    html_path = os.path.join(BLOG_DIR, "index.html")
    css_path = os.path.join(OUT_DIR, "style.css")

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read().strip()

    style_block = (
        '  <style data-inline-name="style.css">\n'
        f"{css}\n"
        "  </style>"
    )
    html = re.sub(
        r'  <link rel="stylesheet" href="assets/style\.css">\n?',
        style_block + "\n",
        html,
    )
    html = re.sub(
        r'  <style data-inline-name="style\.css">.*?  </style>',
        style_block,
        html,
        flags=re.DOTALL,
    )

    for name in INLINE_IMAGE_NAMES:
        image_path = os.path.join(OUT_DIR, name)
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"

        html = html.replace(
            f'<img src="assets/{name}"',
            f'<img src="{data_url}" data-inline-name="{name}"',
        )
        html = re.sub(
            rf'<img src="data:image/png;base64,[^"]+" data-inline-name="{re.escape(name)}"',
            f'<img src="{data_url}" data-inline-name="{name}"',
            html,
        )

    html = re.sub(
        r'alt="Closed-loop k sweep success [^"]+"',
        'alt="Closed-loop k sweep success line graph"',
        html,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def diagnostic_bar():
    labels = ["EEF DP\n(OSC)", "Joint-delta DP\n(no adapter)", "Joint-delta DP\n+ adapter\n(ours)"]
    values = [49 / 50, 0.0, 50 / 50]
    colors = ["#33658a", "#b75d69", "#2f855a"]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Can MH test success rate")
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
    tasks = ["Can\nPH", "Can\nMH", "Lift\nPH", "Lift\nMH", "Square\nPH",
             "Square\nMH", "Tool\nHang\nPH", "Transport\nPH", "Transport\nMH"]
    ours = np.array([45, 50, 50, 49, 27, 23, 0, 0, 5]) / 50
    # Diffusion Policy paper Table I, state policy, DiffusionPolicy-C row.
    # Values are the reported max success rates from max / average-last-10.
    paper_dp = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 0.97, 0.50, 0.94, 0.68])
    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.7, 4.4))
    ax.bar(
        x - width / 2, ours, color="#2f855a", width=width,
        label="Joint-delta DP + adapter (ours)")
    ax.bar(
        x + width / 2, paper_dp, color="#33658a", width=width,
        label="DP paper, state policy")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test success rate")
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
    tasks = ["Can\nPH", "Can\nMH", "Lift\nPH", "Lift\nMH", "Square\nPH",
             "Square\nMH", "Tool\nHang", "Transport\nPH", "Transport\nMH"]
    success = np.array([43, 38, 46, 43, 29, 30, 2, 17, 0]) / 50
    delta_mae = np.array([0.033875, 0.019667, 0.008774, 0.011748, 0.036040,
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


def closed_loop_line_graph():
    k = np.arange(1, 9)
    task_colors = {
        "Can": "#33658a",
        "Lift": "#2f855a",
        "Square": "#b75d69",
        "Tool Hang": "#8b5a2b",
        "Transport": "#6b5fb5",
    }
    series = {
        "Can": {
            "ph": [45, 31, 2, 8, 4, 2, 1, 1],
            "mh": [50, 40, 12, 9, 11, 6, 5, 10],
        },
        "Lift": {
            "ph": [50, 47, 31, 32, 6, 5, 9, 14],
            "mh": [49, 50, 46, 49, 44, 41, 42, 38],
        },
        "Square": {
            "ph": [24, 16, 1, 0, 2, 2, 0, 3],
            "mh": [31, 26, 10, 9, 4, 3, 0, 0],
        },
        "Tool Hang": {
            "ph": [0, 0, 0, 0, 0, 0, 0, 0],
        },
        "Transport": {
            "ph": [0, 2, 0, 0, 0, 0, np.nan, 0],
        },
    }
    style_by_dataset = {
        "ph": ":",
        "mh": "-",
    }

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for task, task_series in series.items():
        for dataset_type, counts in task_series.items():
            rates = np.array(counts, dtype=float) / 50.0
            if np.all(np.isnan(rates)):
                continue
            ax.plot(
                k,
                rates,
                color=task_colors[task],
                linestyle=style_by_dataset[dataset_type],
                linewidth=2.4,
                marker="o",
                markersize=4.5,
                label=f"{task} {dataset_type.upper()}",
            )

    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(k)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_xlabel("Closed-loop inner steps")
    ax.set_ylabel("Test success rate")
    ax.grid(axis="x", visible=False)

    task_handles = [
        Line2D([0], [0], color=color, linewidth=3, label=task)
        for task, color in task_colors.items()
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", linestyle=":", linewidth=2.4, label="PH"),
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.4, label="MH"),
    ]
    task_legend = ax.legend(
        handles=task_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5df",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
    )
    ax.add_artist(task_legend)
    ax.legend(
        handles=style_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5df",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
    )
    save(fig, "fig5_closed_loop.png")


if __name__ == "__main__":
    diagnostic_bar()
    full_pipeline_bar()
    oracle_replay()
    closed_loop_line_graph()
    inline_assets()
