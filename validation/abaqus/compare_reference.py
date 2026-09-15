"""Recompute the compact HomiCSx--Abaqus validation gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
METRIC_COLUMNS = {
    "macro_energy": ("homicsx_macro_energy", "abaqus_macro_energy"),
    "macro_p11": ("homicsx_macro_p11", "abaqus_macro_p11"),
    "macro_p22": ("homicsx_macro_p22", "abaqus_macro_p22"),
    "macro_p33": ("homicsx_macro_p33", "abaqus_macro_p33"),
    "mean_j": ("homicsx_mean_j", "abaqus_mean_j"),
}


def relative_difference_percent(reference: float, comparison: float) -> float:
    """Return ``100 * (comparison-reference) / abs(reference)``."""
    if reference == 0.0:
        raise ValueError("relative difference is undefined for a zero reference")
    return 100.0 * (comparison - reference) / abs(reference)


def evaluate(
    results_path: Path = HERE / "reference_results.csv",
    metadata_path: Path = HERE / "metadata.json",
) -> dict:
    """Evaluate every case and return a serialization-friendly summary."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 9:
        raise ValueError(f"expected 9 comparison cases, found {len(rows)}")
    if len({row["point_id"] for row in rows}) != len(rows):
        raise ValueError("point_id values must be unique")

    maxima = {}
    failures = []
    for metric, (homicsx_column, abaqus_column) in METRIC_COLUMNS.items():
        differences = [
            abs(
                relative_difference_percent(
                    float(row[homicsx_column]), float(row[abaqus_column])
                )
            )
            for row in rows
        ]
        maxima[metric] = max(differences)
        limit = float(metadata["metrics"][metric]["limit_percent"])
        failures.extend(
            f"{row['point_id']}:{metric}={difference:.6g}%>{limit:g}%"
            for row, difference in zip(rows, differences)
            if difference > limit
        )

    return {
        "comparison_id": metadata["comparison_id"],
        "case_count": len(rows),
        "maximum_absolute_difference_percent": maxima,
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    summary = evaluate()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
