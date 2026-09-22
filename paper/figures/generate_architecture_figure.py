"""Generate the reproducible HomiCSx architecture figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent


def box(ax, x, y, width, height, title, detail, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        detail,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#333333",
        linespacing=1.25,
    )


def arrow(ax, start, end, color="#555555", style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            linestyle=style,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def main() -> None:
    figure, ax = plt.subplots(figsize=(7.15, 3.55), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    modules = (
        (0.02, "Geometry", "Seeded 2D/3D\nperiodic inclusions", "#2166ac"),
        (0.22, "Mesh", "Gmsh periodic nodes\nand physical tags", "#1b9e77"),
        (0.42, "FE problem", "Materials, UFL forms\nand MPC constraints", "#6a51a3"),
        (0.62, "Driver", "Load cases, nonlinear\nsteps and averaging", "#d95f02"),
        (0.82, "Results", "Macro histories, fields\nand XDMF output", "#b2182b"),
    )
    width, height, y = 0.16, 0.28, 0.55
    for x, title, detail, color in modules:
        box(ax, x, y, width, height, title, detail, color)
    for index in range(len(modules) - 1):
        arrow(
            ax,
            (modules[index][0] + width, y + height / 2),
            (modules[index + 1][0], y + height / 2),
        )

    box(
        ax,
        0.27,
        0.14,
        0.20,
        0.22,
        "Extension interfaces",
        "Custom materials\nand load callables",
        "#4d4d4d",
    )
    box(
        ax,
        0.53,
        0.14,
        0.20,
        0.22,
        "Typed hooks",
        "Documented lifecycle\nand shared state",
        "#4d4d4d",
    )
    arrow(ax, (0.37, 0.36), (0.50, 0.55), color="#777777", style="--")
    arrow(ax, (0.63, 0.36), (0.70, 0.55), color="#777777", style="--")

    ax.text(
        0.5,
        0.94,
        "HomiCSx end-to-end computational homogenization workflow",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        0.5,
        0.045,
        "Deterministic tests and Abaqus comparisons verify the supported path",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#444444",
    )
    ax.plot([0.08, 0.92], [0.09, 0.09], color="#999999", linewidth=1.0)

    for extension in ("pdf", "png"):
        path = OUTPUT / f"architecture_workflow.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
