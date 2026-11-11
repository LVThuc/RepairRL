"""Build eight slide figures from exported measured data; no model inference."""

import json
from pathlib import Path
import hashlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path("docs/seminar")
OUT = ROOT / "figures"
INK = "#16283F"
MUTED = "#56697F"
BG = "#F7F9FC"
BLUE = "#2463CE"
TEAL = "#087F82"
PURPLE = "#7447BD"
ORANGE = "#BC6517"
RED = "#B43D42"
GREEN = "#16734C"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor(BG)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.045, 0.94, title, fontsize=27, weight="bold", va="top")
    ax.text(0.045, 0.878, subtitle, fontsize=14, color=MUTED, va="top")
    return fig, ax


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body="",
    color=BLUE,
    fill="white",
    title_size=18,
    body_size=13,
):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=fill,
        edgecolor=color,
        linewidth=1.7,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h - 0.03,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        weight="bold",
        color=color,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.025,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            linespacing=1.6,
        )
    return p


def arrow(ax, a, b, color=MUTED, label=None):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color=color
        )
    )
    if label:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.024,
            label,
            ha="center",
            fontsize=12,
            color=color,
        )


def foot(ax, text):
    ax.text(0.045, 0.04, text, fontsize=11.5, color=MUTED, va="bottom")


def figure1():
    fig, ax = canvas(
        "RepairRL: one warm start, two post-training branches",
        "DPO and GRPO independently start from M1. Every model is evaluated with the same V1 protocol.",
    )
    box(
        ax,
        0.05,
        0.44,
        0.18,
        0.22,
        "M0 · Instruct",
        "Qwen2.5-Coder\n1.5B-Instruct",
        BLUE,
    )
    box(
        ax, 0.32, 0.44, 0.18, 0.22, "M1 · SFT", "2,000 verified\nbug / gold pairs", TEAL
    )
    arrow(ax, (0.238, 0.55), (0.31, 0.55), TEAL, "SFT")
    box(
        ax,
        0.59,
        0.65,
        0.18,
        0.16,
        "DPO-hard",
        "Gold vs failed M1 repair",
        PURPLE,
        body_size=12,
    )
    box(
        ax,
        0.59,
        0.26,
        0.18,
        0.16,
        "GRPO",
        "Online execution rewards",
        ORANGE,
        body_size=12,
    )
    arrow(ax, (0.508, 0.59), (0.58, 0.73), PURPLE)
    arrow(ax, (0.508, 0.49), (0.58, 0.34), ORANGE)
    box(
        ax,
        0.835,
        0.40,
        0.12,
        0.24,
        "Evaluate",
        "RunBugRun\n200 tasks\n\nHumanEvalPack\n164 tasks",
        BLUE,
        title_size=17,
        body_size=12,
    )
    arrow(ax, (0.778, 0.73), (0.855, 0.65), PURPLE)
    arrow(ax, (0.778, 0.34), (0.855, 0.39), ORANGE)
    ax.plot([0.14, 0.14, 0.89], [0.43, 0.15, 0.15], color=BLUE, lw=1.2, alpha=0.6)
    arrow(ax, (0.89, 0.15), (0.89, 0.39), BLUE)
    ax.plot([0.41, 0.41, 0.89], [0.43, 0.15, 0.15], color=TEAL, lw=1.2, alpha=0.6)
    ax.text(
        0.46,
        0.105,
        "M0 and M1 also receive the held-out evaluation",
        ha="center",
        fontsize=13,
        color=MUTED,
    )
    foot(
        ax,
        "Additional comparator: untouched pretrained Coder. Agent feedback is inference-only; it does not update model weights.",
    )
    return fig


def figure2():
    fig, ax = canvas(
        "From published examples to verified training pairs",
        "RunBugRun v2 · Python · short self-contained programs · deterministic problem-group splits",
    )
    levels = [
        (0.78, 0.86, "4,000", "Extracted candidates"),
        (0.61, 0.86 * 3913 / 4000, "3,913", "After context / statement filters"),
        (0.44, 0.86 * 3044 / 4000, "3,044", "Execution-verified + deduplicated"),
    ]
    for y, width, count, label in levels:
        x = 0.5 - width / 2
        ax.add_patch(
            Polygon(
                [
                    (x, y),
                    (x + width, y),
                    (x + width - 0.035, y - 0.115),
                    (x + 0.035, y - 0.115),
                ],
                facecolor=BLUE if count == "4,000" else TEAL,
                alpha=0.96,
            )
        )
        ax.text(
            0.5,
            y - 0.036,
            count,
            ha="center",
            va="center",
            fontsize=25,
            weight="bold",
            color="white",
        )
        ax.text(
            0.5, y - 0.083, label, ha="center", va="center", fontsize=14, color="white"
        )
    ax.text(
        0.85, 0.635, "−87 context / statement", fontsize=12, color=MUTED, ha="center"
    )
    ax.text(
        0.85,
        0.465,
        "−869 execution / duplicates",
        fontsize=12,
        color=MUTED,
        ha="center",
    )
    for x, w, title, body, color in [
        (0.045, 0.23, "2,000 train", "1,219 problems", TEAL),
        (0.305, 0.19, "200 dev", "147 problems", BLUE),
        (0.525, 0.19, "200 test", "144 problems", PURPLE),
        (0.745, 0.21, "644 unused", "Accepted; not trained", MUTED),
    ]:
        box(ax, x, 0.12, w, 0.16, title, body, color, title_size=20, body_size=12)
        arrow(ax, (0.5, 0.315), (x + w / 2, 0.29), color)
    foot(
        ax,
        "Gold passes all selected tests; buggy code fails ≥1. Test suites: 2–10 cases. Train/dev/test problem IDs are disjoint.",
    )
    return fig


def figure3():
    fig, ax = canvas(
        "Three learning signals for the same repair task",
        "Same input contract: problem specification + buggy Python program → full replacement program",
    )
    box(ax, 0.26, 0.73, 0.48, 0.095, "Shared repair prompt", "", BLUE, title_size=18)
    cols = [
        (0.04, TEAL, "SFT · demonstration"),
        (0.365, PURPLE, "DPO · preference"),
        (0.69, ORANGE, "GRPO · execution"),
    ]
    for x, col, title in cols:
        box(ax, x, 0.15, 0.27, 0.48, title, "", col, title_size=18)
        arrow(ax, (0.5, 0.72), (x + 0.135, 0.65), col)
    ax.text(
        0.175, 0.52, "Prompt + gold fixed code", ha="center", fontsize=15, weight="bold"
    )
    ax.text(
        0.175,
        0.43,
        "Predict response tokens\nMask prompt loss",
        ha="center",
        fontsize=15,
        linespacing=1.6,
    )
    arrow(ax, (0.175, 0.35), (0.175, 0.285), TEAL)
    ax.text(
        0.175,
        0.23,
        "Update LoRA by cross entropy",
        ha="center",
        fontsize=13,
        color=TEAL,
    )
    ax.text(
        0.5, 0.53, "Chosen: verified gold fix", ha="center", fontsize=15, color=GREEN
    )
    ax.text(
        0.5, 0.465, "Rejected: failed M1 repair", ha="center", fontsize=15, color=RED
    )
    ax.text(
        0.5,
        0.385,
        "Fixed offline pairs\nFrozen M1 reference",
        ha="center",
        fontsize=14,
        linespacing=1.5,
    )
    arrow(ax, (0.5, 0.32), (0.5, 0.275), PURPLE)
    ax.text(
        0.5, 0.23, "Update relative preference", ha="center", fontsize=14, color=PURPLE
    )
    for i in range(4):
        box(
            ax,
            0.709 + i * 0.063,
            0.435,
            0.04,
            0.075,
            f"C{i + 1}",
            "",
            ORANGE,
            title_size=12,
        )
    ax.text(0.825, 0.54, "Sample 4 current-policy programs", ha="center", fontsize=13)
    arrow(ax, (0.825, 0.42), (0.825, 0.38), ORANGE)
    ax.text(
        0.825,
        0.335,
        "Execute trusted tests\nCompute r₁, r₂, r₃, r₄",
        ha="center",
        fontsize=14,
        linespacing=1.5,
    )
    arrow(ax, (0.825, 0.27), (0.825, 0.24), ORANGE)
    ax.text(
        0.825,
        0.195,
        "Update using group-relative rewards",
        ha="center",
        fontsize=12,
        color=ORANGE,
    )
    foot(
        ax,
        "C1–C4 are schematic; rewards come from execution. Hidden tests stay outside the generated program.",
    )
    return fig


def curve(title, subtitle, steps, counts, color, selected, footer):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.065, 0.94, title, fontsize=27, weight="bold", va="top")
    fig.text(0.065, 0.877, subtitle, fontsize=14, color=MUTED)
    ax = fig.add_axes([0.08, 0.19, 0.84, 0.59])
    ax.set_facecolor("white")
    ax.plot(steps, counts, color=color, lw=3, marker="o", markersize=8)
    ax.axhline(20, color=MUTED, ls="--", lw=1.2, label="M1 parent: 20/60")
    for x, y in zip(steps, counts):
        ax.annotate(
            f"{y}/60",
            (x, y),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=14,
            color=color,
        )
    i = steps.index(selected)
    ax.scatter(
        [selected], [counts[i]], s=210, facecolors="none", edgecolors=color, lw=2
    )
    ax.set(
        xlabel="Optimizer step",
        ylabel="Fully repaired dev tasks (out of 60)",
        ylim=(0, 30),
        xticks=steps,
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower left", frameon=False, fontsize=13)
    fig.text(0.065, 0.07, footer, fontsize=12, color=MUTED)
    return fig


def figure4(data):
    return curve(
        "DPO dev progression: early peak, later decline",
        "Repair pass@1 on a fixed 60-problem dev set — not training preference accuracy",
        [r["step"] for r in data],
        [r["passed"] for r in data],
        PURPLE,
        66,
        "Selected checkpoint: step 66 (tied at 132; earlier wins). Later dev decline is consistent with overfitting or mismatch, not proof of cause.",
    )


def figure5(data):
    data = [r for r in data if r["step"] <= 1200]
    return curve(
        "GRPO: dev success peaks at step 800",
        "Fixed 60-problem dev screen · snapshots every 200 steps · completed evaluations through step 1,200",
        [r["step"] for r in data],
        [round(r["pass1"] * 60) for r in data],
        ORANGE,
        800,
        "Step 800: 25/60 (41.7%). Steps 1,000 and 1,200: 24/60 (40.0%). These are dev results, not the held-out comparison.",
    )


def figure6(rows):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(
        0.065,
        0.94,
        "Held-out repair: compare the models, not training loss",
        fontsize=27,
        weight="bold",
        va="top",
    )
    fig.text(
        0.065,
        0.877,
        "Pass@1 · completed V1 / FP16 evaluations · DPO step 66 and interim GRPO step 800",
        fontsize=14,
        color=MUTED,
    )
    ax = fig.add_axes([0.08, 0.21, 0.84, 0.57])
    names = ["M0 Instruct", "M1 SFT", "DPO-hard s66", "GRPO s800"]
    x = np.arange(4)
    width = 0.34
    for off, b, col, label in [
        (-width / 2, "runbugrun", TEAL, "RunBugRun · n=200"),
        (width / 2, "humanevalpack", BLUE, "HumanEvalPack · n=164"),
    ]:
        y = [
            next(
                r["metrics"]["pass@1"] * 100
                for r in rows
                if r["model"] == n and r["benchmark"] == b
            )
            for n in names
        ]
        bars = ax.bar(x + off, y, width, color=col, label=label)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in y], padding=5, fontsize=15)
    ax.set(
        xticks=x,
        xticklabels=["M0\nInstruct", "M1\nSFT", "DPO-hard\nstep 66", "GRPO\nstep 800"],
        ylabel="Tasks fully repaired on first attempt (%)",
        ylim=(0, 80),
    )
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=14)
    fig.text(
        0.065,
        0.07,
        "Benchmarks differ in format and distribution. Single trained configurations; this is not a universal algorithm ranking.",
        fontsize=12,
        color=MUTED,
    )
    return fig


def figure7(rows):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.23, top=0.76, wspace=0.18)
    fig.text(
        0.065,
        0.94,
        "Does execution feedback add successful repairs?",
        fontsize=27,
        weight="bold",
        va="top",
    )
    fig.text(
        0.065,
        0.877,
        "First attempt versus success within at most 3 sequential feedback-guided attempts",
        fontsize=14,
        color=MUTED,
    )
    names = ["M0 Instruct", "M1 SFT", "DPO-hard s66", "GRPO s800"]
    x = np.arange(4)
    w = 0.35
    for ax, b, title in zip(
        axes,
        ["runbugrun", "humanevalpack"],
        ["RunBugRun · 200 tasks", "HumanEvalPack · 164 tasks"],
        strict=True,
    ):
        for offset, key, col, label in [
            (-w / 2, "pass@1", "#9BBADB", "First attempt"),
            (w / 2, "agent@3", BLUE, "Within 3 attempts"),
        ]:
            vals = [
                next(
                    r["metrics"][key] * 100
                    for r in rows
                    if r["model"] == n and r["benchmark"] == b
                )
                for n in names
            ]
            bars = ax.bar(x + offset, vals, w, color=col, label=label)
            ax.bar_label(
                bars, labels=[f"{v:.1f}" for v in vals], padding=4, fontsize=12
            )
        ax.set(
            xticks=x,
            xticklabels=["M0", "M1", "DPO\ns66", "GRPO\ns800"],
            ylim=(0, 80),
            title=title,
            ylabel="Tasks solved (%)",
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    fig.text(
        0.065,
        0.055,
        "Agent@3 is not independent-sample pass@3. V1 retries receive only a sanitized test summary; no hidden inputs or expected outputs.",
        fontsize=11.5,
        color=MUTED,
    )
    return fig


def figure8(case):
    fig, ax = canvas(
        "Actual repair: a cup swap must update the state only once",
        "RunBugRun 21346 / p00047 · measured first attempts · curated example, not a representative sample",
    )
    box(ax, 0.035, 0.39, 0.255, 0.38, "Original bug", "", RED, title_size=19)
    ax.text(
        0.055,
        0.68,
        "if x == base:\n    base = y\nif y == base:\n    base = x",
        family="DejaVu Sans Mono",
        fontsize=15,
        va="top",
        linespacing=1.7,
    )
    ax.text(
        0.055,
        0.435,
        "Two independent branches can\nundo the same swap.",
        fontsize=13,
        color=MUTED,
        linespacing=1.5,
    )
    arrow(ax, (0.297, 0.58), (0.33, 0.58))
    items = [
        ("PT-base", "Pretrained", "Two == branches retained", RED),
        ("M0 Instruct", "M0 Instruct", "range(int(input()))", RED),
        ("M1 SFT", "M1 SFT", "Two != branches", RED),
        ("DPO-hard s66", "DPO-hard · step 66", "if … elif …", GREEN),
        ("GRPO s800", "GRPO · step 800", "if … elif …", GREEN),
    ]
    for i, (key, label, excerpt, col) in enumerate(items):
        y = 0.72 - i * 0.118
        ax.add_patch(
            FancyBboxPatch(
                (0.345, y - 0.065),
                0.335,
                0.096,
                boxstyle="round,pad=0.008,rounding_size=.009",
                facecolor="white",
                edgecolor="#D6DFEA",
            )
        )
        ax.text(0.36, y + 0.004, label, fontsize=14, weight="bold", color=col)
        ax.text(
            0.36,
            y - 0.033,
            excerpt,
            fontsize=12,
            color=MUTED,
            family="DejaVu Sans Mono",
        )
        fraction = case["models"][key]["first_fraction"]
        passed = round(fraction * 3)
        ax.text(
            0.663,
            y - 0.01,
            f"{passed}/3",
            ha="right",
            fontsize=19,
            weight="bold",
            color=col,
        )
    box(ax, 0.73, 0.43, 0.235, 0.34, "Gold / passing repair", "", GREEN, title_size=17)
    ax.text(
        0.745,
        0.67,
        "if x == base:\n    base = y\nelif y == base:\n    base = x",
        family="DejaVu Sans Mono",
        fontsize=14,
        va="top",
        linespacing=1.7,
    )
    ax.text(
        0.847,
        0.34,
        "DPO & GRPO\n3/3 selected tests passed",
        ha="center",
        fontsize=16,
        color=GREEN,
        weight="bold",
        linespacing=1.5,
    )
    arrow(ax, (0.685, 0.36), (0.72, 0.48), GREEN)
    ax.text(
        0.16,
        0.24,
        "Task: track the ball’s location\nas cup positions are swapped.",
        ha="center",
        fontsize=14,
        linespacing=1.5,
    )
    foot(
        ax,
        "Code excerpts summarize the recorded full programs. M0 invents a leading count; M1 changes comparisons. Hidden scoring cases are omitted.",
    )
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    rows = json.loads((ROOT / "report/results.json").read_text())
    cases = json.loads((ROOT / "report/case-studies.json").read_text())
    dpo = json.loads((ROOT / "report/dpo-dev.json").read_text())
    grpo = json.loads((ROOT / "report/grpo-dev.json").read_text())
    figures = [
        ("01_overall_pipeline", figure1()),
        ("02_dataset_filtering_funnel", figure2()),
        ("03_sft_dpo_grpo_workflow", figure3()),
        ("04_dpo_dev_progression", figure4(dpo)),
        ("05_grpo_dev_progression", figure5(grpo)),
        ("06_model_pass1_comparison", figure6(rows)),
        ("07_agent3_comparison", figure7(rows)),
        (
            "08_actual_repair_case",
            figure8(next(c for c in cases if c["id"] == "21346")),
        ),
    ]
    with PdfPages(OUT / "RepairRL_seminar_figures.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=fig.get_facecolor())
            fig.savefig(OUT / f"{name}.svg", facecolor=fig.get_facecolor())
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    sourcefiles = [
        ROOT / "report" / n
        for n in ["results.json", "case-studies.json", "dpo-dev.json", "grpo-dev.json"]
    ]
    (OUT / "sources.json").write_text(
        json.dumps(
            {
                "scope": "Completed matched V1 held-out results; dev curves frozen through330/1200; no invented training outcomes",
                "sources": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sourcefiles
                },
                "figures": [n for n, _ in figures],
            },
            indent=2,
        )
    )
    print("Saved", len(figures), "figures as PNG/SVG and one combined PDF")


if __name__ == "__main__":
    main()
