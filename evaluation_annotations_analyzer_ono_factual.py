"""
The barplot generates non-factual evaluation results only
evaluation_annotations_analyzer.py

Loads: Annotations_Factual.csv
Columns (detected from your file):
  Repo, Q#, Question,
  Claude_AA, RepoWise_AA, ChatGPT_AA,
  Claude_SK, RepoWise_SK, ChatGPT_SK,
  Claude_NIK, RepoWise_NIK, ChatGPT_NIK

What it computes:
1) Pairwise Krippendorff’s α (nominal) between annotators + average pairwise α
2) Fleiss’ κ (generalized; supports missing per-item ratings)

It reports agreement:
- Per system/model: Claude, RepoWise, ChatGPT
- Overall across ALL system-question judgments (i.e., treats each (Repo,Q#,System) as an item)

Outputs (CSV) into --outdir (default: ./iaa_results):
- iaa_summary.csv
- pairwise_alpha_<SCOPE>.csv  (SCOPE = Claude / RepoWise / ChatGPT / OVERALL)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------------------
# Load Excel
# -------------------------------
file_path = "RepoWise_Annotations.xlsx"
df = pd.read_excel(file_path)

systems = ["RepoWise", "Claude", "ChatGPT", "Copilot"]

annotator_map = {
    "A1": {
        "RepoWise": "RepoWise_A1",
        "Claude": "Claude_A1",
        "ChatGPT": "ChatGPT_A1",
        "Copilot": "Copilot_A1",
    },
    "A2": {
        "RepoWise": "RepoWise_A2",
        "Claude": "Claude_A2",
        "ChatGPT": "ChatGPT_A2",
        "Copilot": "Copilot_A2",
    },
}

# -------------------------------
# Non-factual rank counts (1-4)
# -------------------------------
rank_counts = {a: {} for a in annotator_map}
for annot, cols in annotator_map.items():
    for sys, col in cols.items():
        rank_counts[annot][sys] = [
            int((df[col] == 1).sum()),
            int((df[col] == 2).sum()),
            int((df[col] == 3).sum()),
            int((df[col] == 4).sum()),
        ]

def normalize(counts):
    s = sum(counts)
    return [c / s for c in counts] if s else [0, 0, 0, 0]

# -------------------------------
# Build Y positions (Non-factual only)
# -------------------------------
labels = []
y_positions = []
system_line = []

s1, s2, s3, s4 = [], [], [], []  # Rank1..Rank4

gap_between_systems = 1
gap_within_system = 0.14
bar_height = 0.95

y = 0.0
nonfactual_y_min = None
nonfactual_y_max = None

for sys in systems:
    # A1
    y_a1 = y
    r1, r2, r3_, r4_ = normalize(rank_counts["A1"][sys])
    y_positions.append(y_a1)
    labels.append("A1")
    system_line.append(sys)
    s1.append(r1); s2.append(r2); s3.append(r3_); s4.append(r4_)

    # A2
    y_a2 = y + bar_height + gap_within_system
    r1, r2, r3_, r4_ = normalize(rank_counts["A2"][sys])
    y_positions.append(y_a2)
    labels.append("A2")
    system_line.append("")
    s1.append(r1); s2.append(r2); s3.append(r3_); s4.append(r4_)

    if nonfactual_y_min is None:
        nonfactual_y_min = y_a1 - bar_height / 2
    nonfactual_y_max = y_a2 + bar_height / 2

    y += gap_between_systems + 1.9

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8))
y_arr = np.array(y_positions)

# Colors (kept same as your original)
color_rank1 = "#2E7D32"    # green
color_rank2 = "#FFA726"    # orange
color_rank3 = "#D32F2F"    # red
color_rank4 = "#6A1B9A"    # purple

edge_color = "#2C2C2C"
edge_width = 1.2

# Bars (horizontal stacked)
ax.barh(y_arr, s1, height=bar_height, color=color_rank1, edgecolor=edge_color, linewidth=edge_width)
ax.barh(y_arr, s2, height=bar_height, left=s1, color=color_rank2, edgecolor=edge_color, linewidth=edge_width)
ax.barh(y_arr, s3, height=bar_height, left=np.array(s1) + np.array(s2), color=color_rank3, edgecolor=edge_color, linewidth=edge_width)
ax.barh(y_arr, s4, height=bar_height, left=np.array(s1) + np.array(s2) + np.array(s3), color=color_rank4, edgecolor=edge_color, linewidth=edge_width)

# Percentage labels
for y_pos, a1, a2, a3, a4 in zip(y_arr, s1, s2, s3, s4):
    if a1 > 0.05:
        ax.text(a1/2, y_pos, f"{a1*100:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    if a2 > 0.05:
        ax.text(a1 + a2/2, y_pos, f"{a2*100:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="black")
    if a3 > 0.05:
        ax.text(a1 + a2 + a3/2, y_pos, f"{a3*100:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    if a4 > 0.05:
        ax.text(a1 + a2 + a3 + a4/2, y_pos, f"{a4*100:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

# Axes
ax.set_xlim(0, 1.18)
ax.set_ylim(y_arr[0] - 0.8, y_arr[-1] + 0.8)

# Y-axis two-line labels (A1 shows system name, A2 blank second line)
two_line_labels = []
for lab, syslab in zip(labels, system_line):
    if lab == "A1":
        two_line_labels.append(f"A1\n{syslab}")
    else:
        two_line_labels.append("A2\n")

ax.set_yticks(y_arr)
ax.set_yticklabels(two_line_labels, ha="right", fontsize=9, linespacing=1.15)
ax.tick_params(axis="y", pad=10)

# Grid
ax.grid(axis="x", linestyle="--", alpha=0.4, color="gray", linewidth=0.8)

# Background region (Non-factual only)
nonfactual_region = plt.Rectangle(
    (0, nonfactual_y_min),
    1.18,
    nonfactual_y_max - nonfactual_y_min,
    facecolor="#FFF3E0",
    alpha=0.25,
    zorder=0
)
ax.add_patch(nonfactual_region)

# Legend (Non-factual only)
nonfactual_handles = [
    Patch(facecolor=color_rank1, edgecolor=edge_color, label="Rank 1"),
    Patch(facecolor=color_rank2, edgecolor=edge_color, label="Rank 2"),
    Patch(facecolor=color_rank3, edgecolor=edge_color, label="Rank 3"),
    Patch(facecolor=color_rank4, edgecolor=edge_color, label="Rank 4"),
]

# Place legend in the middle of the plot vertically (stable placement)
ymin, ymax = ax.get_ylim()
nonfactual_legend_y = (nonfactual_y_min + nonfactual_y_max) / 2
nonfactual_y_frac = (nonfactual_legend_y - ymin) / (ymax - ymin)

ax.legend(
    handles=nonfactual_handles,
    title="Non-Factual",
    loc="center right",
    bbox_to_anchor=(0.985, nonfactual_y_frac),
    frameon=True,
    shadow=True,
    fontsize=8,
    title_fontsize=9,
    fancybox=True,
    framealpha=0.95
)

plt.tight_layout()

plt.savefig(
    "evaluation_findings.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300
)

plt.show()
