"""Render the website's overall-success charts from the reported CSV aggregates.

Run with the repository virtual environment; matplotlib is only needed to rebuild
the standalone SVG figures, not to serve or view the website.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


SITE = Path(__file__).resolve().parents[1]
COLORS = {"ours": "#5943a0", "baseline": "#78668f", "ablation": "#c7bedb"}
ABLATION_NOTES = {
    "DLP": "no explicit 3D geometry",
    "VAE-GS": "no object-centric structure",
}


def render_chart(scores, benchmark, setting, baselines, ablations, filename, baseline_note=""):
    """All figures share a zero-based 0–100% scale and the same label margin."""
    compact = not baselines
    comparison_count = len(ablations) + len(baselines)
    height = 2.55 if compact else 4.35 + 0.5 * (comparison_count - 4)
    figure, axes = plt.subplots(figsize=(6.1, height))
    figure.subplots_adjust(left=0.34, right=0.95, top=0.96, bottom=0.24 if compact else 0.17)
    axes.set_xlim(0, 100)
    axes.set_ylim(1.8 + len(ablations) if compact else 3.3 + comparison_count, -0.5)
    axes.set_yticks([])
    axes.set_xticks([0, 25, 50, 75, 100])
    axes.tick_params(axis="x", labelsize=11.5, colors="#626572", length=3, pad=6)
    axes.set_xlabel("Overall success rate (%)", fontsize=12.5, color="#626572", labelpad=9)
    for side in ["top", "left", "right"]:
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color("#dedee7")

    def heading(label, y):
        axes.text(-52, y, label, ha="left", va="center", fontsize=12.5, color="#626572", clip_on=False)

    def bar(method, category, y):
        score = scores[benchmark, setting, method]
        note = ABLATION_NOTES.get(method) if category == "ablation" else None
        axes.text(-5, y - 0.16 if note else y, method, ha="right", va="center", fontsize=15,
                  weight="bold" if category == "ours" else "normal", color="#252633", clip_on=False)
        if note:
            axes.text(-5, y + 0.32, f"({note})", ha="right", va="center",
                      fontsize=9.5, color="#626572", clip_on=False)
        for width, color in [(100, "#f0eef6"), (score, COLORS[category])]:
            patch = FancyBboxPatch((0, y - 0.34), width, 0.68,
                                  boxstyle="round,pad=0,rounding_size=1.1",
                                  mutation_aspect=0.1, facecolor=color, edgecolor="none")
            axes.add_patch(patch)
        axes.text(score - 2.2, y, f"{score:.1f}%", ha="right", va="center", fontsize=13.5,
                  weight="bold", color="#252633" if category == "ablation" else "white")

    bar("ParticleSplat", "ours", 0.1)
    heading("Ablations (same EC-Diffuser policy network)", 1.18)
    for index, method in enumerate(ablations):
        bar(method, "ablation", 2.0 + index)
    if baselines:
        divider = 1.83 + len(ablations)
        axes.plot([-52, 100], [divider, divider], color="#c7bfda", linewidth=1, clip_on=False)
        heading(f"Baselines ({baseline_note})", divider + 0.67)
        for index, method in enumerate(baselines):
            bar(method, "baseline", divider + 1.47 + index)

    output = SITE / "static" / "charts" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", facecolor="white", metadata={
        "Title": f"{benchmark} {setting}: overall policy success rates",
        "Description": "Reported ParticleSplat manuscript aggregates. Purple: ParticleSplat; light lavender: representation ablations, shown above muted-purple baselines. DLP has no explicit 3D geometry; VAE-GS has no object-centric structure. Group headings explain the policy networks. All bars use a 0–100% scale.",
        "Date": None,
    })
    plt.close(figure)
    print(output.relative_to(SITE))


def main():
    with (SITE / "static" / "policy-results.csv").open(newline="") as source:
        scores = {(row["benchmark"], row["view_setting"], row["method"]): float(row["mean_success_percent"])
                  for row in csv.DictReader(source) if row["task"] == "Overall (reported)"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "particlesplat-overall-sr"})
    render_chart(scores, "RLBench", "single-view",
                 ["GNFactor", "ManiGaussian"], ["DLP", "VAE-GS"], "rlbench-overall.svg",
                 baseline_note="different policy networks")
    render_chart(scores, "MimicGen", "multi-view",
                 ["3D-DLP", "EquiDiff"], ["DLP"], "mimicgen-overall.svg",
                 baseline_note="3D-DLP: EC-Diffuser; EquiDiff: own policy")
    render_chart(scores, "RLBench", "multi-view", [], ["DLP", "VAE-GS"], "rlbench-multiview.svg")


if __name__ == "__main__":
    main()
