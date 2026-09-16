"""Generate the SoftwareX macro-response validation figure.

Run from the repository root with::

    python paper/figures/generate_validation_figure.py

Only conventional macroscopic shear-stress histories are plotted.  The source
data are the compact, committed HomiCSx and Abaqus validation results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "abaqus"
OUTPUT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def history_xy(history: list[dict]) -> tuple[list[float], list[float]]:
    return (
        [float(row["time"]) for row in history],
        [float(row["macro_p12"]) for row in history],
    )


def plot_pair(
    ax, homicsx, abaqus, label: str, color: str, line_width: float = 2.0
) -> None:
    time_hx, stress_hx = history_xy(homicsx)
    time_abq, stress_abq = history_xy(abaqus)
    ax.plot(
        time_hx,
        stress_hx,
        color=color,
        linewidth=line_width,
        label=f"{label} — HomiCSx",
    )
    ax.plot(
        time_abq[::5],
        stress_abq[::5],
        color=color,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.9,
        markersize=3.2,
        linestyle="none",
        label=f"{label} — Abaqus",
    )


def main() -> None:
    homogeneous_hx = load("viscoelastic_homicsx_results.json")
    homogeneous_abq = load("viscoelastic_abaqus_results.json")
    three_d_hx = load("viscoelastic_homicsx_3d_results.json")
    three_d_abq = load("viscoelastic_abaqus_3d_results.json")
    heterogeneous_hx = {
        row["case_id"]: row
        for row in load("viscoelastic_homicsx_heterogeneous_results.json")["cases"]
    }
    heterogeneous_abq = {
        row["case_id"]: row
        for row in load("viscoelastic_abaqus_heterogeneous_results.json")["cases"]
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "font.family": "DejaVu Sans",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), constrained_layout=True)

    plot_pair(
        axes[0],
        homogeneous_hx["end_to_end_history"],
        homogeneous_abq["history"],
        "Homogeneous 2D",
        "#2468a2",
        line_width=4.0,
    )
    plot_pair(
        axes[0],
        three_d_hx["end_to_end_history"],
        three_d_abq["history"],
        "Homogeneous 3D",
        "#d95f02",
        line_width=2.0,
    )
    axes[0].set_title("(a) Homogeneous relaxation")

    plot_pair(
        axes[1],
        heterogeneous_hx["visco_centered_circle_vf20"]["history"],
        heterogeneous_abq["visco_centered_circle_vf20"]["history"],
        "Centered inclusion",
        "#1b9e77",
    )
    plot_pair(
        axes[1],
        heterogeneous_hx["visco_periodic_split_circle"]["history"],
        heterogeneous_abq["visco_periodic_split_circle"]["history"],
        "Periodic split inclusion",
        "#7b3294",
    )
    axes[1].set_title("(b) Heterogeneous 2D relaxation")

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Macroscopic shear stress $\overline{P}_{12}$")
        ax.grid(True, color="#d9d9d9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for extension in ("pdf", "png"):
        path = OUTPUT / f"viscoelastic_validation.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
