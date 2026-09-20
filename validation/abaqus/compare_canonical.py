"""Compare separately generated canonical HomiCSx and Abaqus results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .canonical_common import (
    HERE,
    frobenius_error_percent,
    load_cases,
    relative_error_percent,
)


def _indexed(cases):
    return {case["case_id"]: case for case in cases}


def evaluate(
    homicsx_path: Path = HERE / "canonical_homicsx_results.json",
    abaqus_path: Path = HERE / "canonical_abaqus_results.json",
) -> dict:
    manifest = load_cases()
    limits = manifest["acceptance"]
    homicsx = json.loads(homicsx_path.read_text(encoding="utf-8"))
    abaqus = json.loads(abaqus_path.read_text(encoding="utf-8"))
    hx_linear = _indexed(homicsx["linear_cases"])
    abq_linear = _indexed(abaqus["linear_cases"])
    expected_ids = {case["case_id"] for case in manifest["linear_cases"]}
    if set(hx_linear) != expected_ids or set(abq_linear) != expected_ids:
        raise ValueError("result case IDs do not match the canonical manifest")

    failures = []
    linear_summary = {}
    for case_id in sorted(expected_ids):
        hx_case = hx_linear[case_id]
        abq_case = abq_linear[case_id]
        stiffness_error = frobenius_error_percent(
            hx_case["stiffness"], abq_case["stiffness"]
        )
        if stiffness_error > limits["linear_stiffness_frobenius_percent"]:
            failures.append("{}:stiffness={:.6g}%".format(case_id, stiffness_error))
        probe_summary = {}
        for probe_name in manifest["macro_strain_probes"]:
            hx_probe = hx_case["probes"][probe_name]
            abq_probe = abq_case["probes"][probe_name]
            stress_scale = max(np.linalg.norm(hx_probe["macro_stress"]), 1.0e-12)
            stress_error = float(
                100.0
                * np.linalg.norm(
                    np.asarray(abq_probe["macro_stress"])
                    - np.asarray(hx_probe["macro_stress"])
                )
                / stress_scale
            )
            energy_error = float(
                relative_error_percent(
                    hx_probe["macro_energy"], abq_probe["macro_energy"]
                )
            )
            if stress_error > limits["linear_probe_stress_percent"]:
                failures.append(
                    "{}:{}:stress={:.6g}%".format(case_id, probe_name, stress_error)
                )
            if energy_error > limits["linear_probe_energy_percent"]:
                failures.append(
                    "{}:{}:energy={:.6g}%".format(case_id, probe_name, energy_error)
                )
            probe_summary[probe_name] = {
                "stress_error_percent": stress_error,
                "energy_error_percent": energy_error,
            }
        linear_summary[case_id] = {
            "stiffness_frobenius_error_percent": stiffness_error,
            "probes": probe_summary,
        }

    hx_nl = homicsx["nonlinear_case"]
    abq_nl = abaqus["nonlinear_case"]
    nonlinear_summary = {}
    for metric in ("macro_energy", "macro_p12", "mean_j"):
        error = float(relative_error_percent(hx_nl[metric], abq_nl[metric]))
        nonlinear_summary[metric + "_error_percent"] = error
        limit = limits["nonlinear_{}_percent".format(metric)]
        if error > limit:
            failures.append("nonlinear:{}={:.6g}%".format(metric, error))

    return {
        "passed": not failures,
        "linear_cases": linear_summary,
        "nonlinear_case": nonlinear_summary,
        "failures": failures,
    }


def main() -> int:
    summary = evaluate()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
