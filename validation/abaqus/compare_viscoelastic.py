"""Compare HomiCSx and Abaqus homogeneous shear-relaxation histories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .canonical_common import HERE


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _curve_metrics(reference, comparison) -> dict:
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    peak = float(np.max(np.abs(reference)))
    difference = comparison - reference
    return {
        "peak_normalized_max_percent": float(100.0 * np.max(np.abs(difference)) / peak),
        "peak_normalized_rms_percent": float(
            100.0 * np.sqrt(np.mean(difference**2)) / peak
        ),
        "endpoint_percent": float(
            100.0 * abs(difference[-1]) / abs(reference[-1])
        ),
    }


def evaluate(
    homicsx_path: Path = HERE / "viscoelastic_homicsx_results.json",
    abaqus_path: Path = HERE / "viscoelastic_abaqus_results.json",
) -> dict:
    case = _load(HERE / "viscoelastic_case.json")
    homicsx = _load(homicsx_path)
    abaqus = _load(abaqus_path)
    end_to_end = homicsx["end_to_end_history"]
    discrete = homicsx["discrete_material_history"]
    abq_history = abaqus["history"]
    times = np.asarray([row["time"] for row in end_to_end])
    for name, history in (("discrete", discrete), ("abaqus", abq_history)):
        other_times = np.asarray([row["time"] for row in history])
        if not np.allclose(times, other_times, rtol=0.0, atol=5.0e-7):
            raise ValueError("{} time grid does not match HomiCSx".format(name))
    end_stress = [row["macro_p12"] for row in end_to_end]
    material_metrics = _curve_metrics(
        [row["macro_p12"] for row in discrete], end_stress
    )
    abaqus_metrics = _curve_metrics(end_stress, [row["macro_p12"] for row in abq_history])
    max_j_error = max(
        abs(row["mean_j"] - 1.0) for history in (end_to_end, abq_history) for row in history
    )
    limits = case["acceptance"]
    failures = []
    for label, metrics in (("material_recurrence", material_metrics), ("abaqus", abaqus_metrics)):
        for metric, key in (
            ("peak_normalized_max_percent", "stress_curve_peak_normalized_max_percent"),
            ("peak_normalized_rms_percent", "stress_curve_peak_normalized_rms_percent"),
            ("endpoint_percent", "endpoint_stress_percent"),
        ):
            if metrics[metric] > limits[key]:
                failures.append("{}:{}={:.6g}%".format(label, metric, metrics[metric]))
    if max_j_error > limits["mean_j_absolute_error"]:
        failures.append("mean_j_absolute_error={:.6g}".format(max_j_error))
    return {
        "passed": not failures,
        "homi_end_to_end_vs_material_recurrence": material_metrics,
        "abaqus_vs_homicsx_end_to_end": abaqus_metrics,
        "maximum_mean_j_absolute_error": max_j_error,
        "failures": failures,
    }


def evaluate_heterogeneous(
    homicsx_path: Path = HERE / "viscoelastic_homicsx_heterogeneous_results.json",
    abaqus_path: Path = HERE / "viscoelastic_abaqus_heterogeneous_results.json",
) -> dict:
    case = _load(HERE / "viscoelastic_case.json")
    homicsx = {row["case_id"]: row for row in _load(homicsx_path)["cases"]}
    abaqus = {row["case_id"]: row for row in _load(abaqus_path)["cases"]}
    expected = {row["case_id"] for row in case["heterogeneous_cases"]}
    if set(homicsx) != expected or set(abaqus) != expected:
        raise ValueError("heterogeneous result case IDs do not match the manifest")
    limits = case["heterogeneous_acceptance"]
    failures = []
    summaries = {}
    for case_id in sorted(expected):
        hx_history = homicsx[case_id]["history"]
        abq_history = abaqus[case_id]["history"]
        hx_times = np.asarray([row["time"] for row in hx_history])
        abq_times = np.asarray([row["time"] for row in abq_history])
        if not np.allclose(hx_times, abq_times, rtol=0.0, atol=5.0e-7):
            raise ValueError("{} time grids do not match".format(case_id))
        metrics = _curve_metrics(
            [row["macro_p12"] for row in hx_history],
            [row["macro_p12"] for row in abq_history],
        )
        j_error = max(
            abs(hx["mean_j"] - abq["mean_j"])
            for hx, abq in zip(hx_history, abq_history)
        )
        summaries[case_id] = dict(metrics, maximum_mean_j_absolute_error=j_error)
        for metric, key in (
            ("peak_normalized_max_percent", "stress_curve_peak_normalized_max_percent"),
            ("peak_normalized_rms_percent", "stress_curve_peak_normalized_rms_percent"),
            ("endpoint_percent", "endpoint_stress_percent"),
        ):
            if metrics[metric] > limits[key]:
                failures.append("{}:{}={:.6g}%".format(case_id, metric, metrics[metric]))
        if j_error > limits["mean_j_absolute_error"]:
            failures.append("{}:mean_j={:.6g}".format(case_id, j_error))
    return {"passed": not failures, "cases": summaries, "failures": failures}


def main() -> int:
    summary = {
        "homogeneous": evaluate(),
        "heterogeneous": evaluate_heterogeneous(),
    }
    summary["passed"] = summary["homogeneous"]["passed"] and summary["heterogeneous"]["passed"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
