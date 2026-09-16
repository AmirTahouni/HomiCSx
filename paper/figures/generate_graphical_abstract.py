"""Generate the optional SoftwareX graphical abstract."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent


def box(ax, x, title, detail, color):
    patch = FancyBboxPatch(
        (x, 0.31),
        0.17,
        0.42,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.085, 0.60, title, ha="center", va="center", fontsize=17,
            weight="bold", color=color)
    ax.text(x + 0.085, 0.43, detail, ha="center", va="center", fontsize=12,
            color="#24323d", linespacing=1.35)


def main():
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = [0.015, 0.215, 0.415, 0.615, 0.815]
    items = [
        ("Geometry", "seeded periodic\nparticulate cells", "#16697a"),
        ("Mesh", "matching opposite\nboundaries", "#287271"),
        ("FEM", "linear and\nfinite strain", "#8f5d2e"),
        ("Homogenize", "2D/3D macro\nresponse", "#9b3a4a"),
        ("Reuse", "fields, histories,\nand validation", "#554971"),
    ]
    for x, (title, detail, color) in zip(positions, items):
        box(ax, x, title, detail, color)

    for left, right in zip(positions[:-1], positions[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left + 0.174, 0.52),
                (right - 0.006, 0.52),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color="#52616b",
            )
        )

    ax.text(
        0.5,
        0.87,
        "HomiCSx: one inspectable computational-homogenization pipeline",
        ha="center",
        va="center",
        fontsize=23,
        weight="bold",
        color="#17252a",
    )
    ax.text(
        0.5,
        0.14,
        "Custom materials  •  load histories  •  typed hooks",
        ha="center",
        va="center",
        fontsize=16,
        color="#334e58",
    )

    fig.savefig(HERE / "graphical_abstract.png", dpi=300, bbox_inches="tight",
                pad_inches=0.05)
    fig.savefig(HERE / "graphical_abstract.pdf", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote graphical abstract under {HERE}")


if __name__ == "__main__":
    main()
