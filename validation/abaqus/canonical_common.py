"""Shared, dependency-light helpers for the canonical Abaqus suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_cases(path: Path = HERE / "canonical_cases.json") -> dict:
    """Load and minimally validate the canonical case manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [case["case_id"] for case in data["linear_cases"]]
    ids.append(data["nonlinear_case"]["case_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("canonical case_id values must be unique")
    if set(data["macro_strain_probes"]) != {"eps11", "eps22", "gamma12"}:
        raise ValueError("the three canonical plane-strain probes are required")
    return data


def stiffness_to_probe_metrics(stiffness, probes):
    """Return conventional macro stress and energy for each strain probe."""
    matrix = np.asarray(stiffness, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("plane-strain stiffness must have shape (3, 3)")
    output = {}
    for name, values in probes.items():
        strain = np.asarray(values, dtype=float)
        stress = matrix @ strain
        output[name] = {
            "macro_strain": strain.tolist(),
            "macro_stress": stress.tolist(),
            "macro_energy": float(0.5 * strain @ stress),
        }
    return output


def relative_error_percent(reference, comparison, *, floor=1.0e-12):
    """Absolute relative error in percent, with a scale floor near zero."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    denominator = np.maximum(np.abs(reference), floor)
    return 100.0 * np.abs(comparison - reference) / denominator


def frobenius_error_percent(reference, comparison):
    """Relative Frobenius-norm error in percent."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    norm = np.linalg.norm(reference)
    if norm == 0.0:
        raise ValueError("reference matrix has zero norm")
    return float(100.0 * np.linalg.norm(comparison - reference) / norm)
