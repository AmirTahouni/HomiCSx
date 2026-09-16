import json

from validation.abaqus.canonical_common import HERE, load_cases
from validation.abaqus.compare_canonical import evaluate
from validation.abaqus.compare_viscoelastic import (
    evaluate as evaluate_viscoelastic,
    evaluate_3d as evaluate_viscoelastic_3d,
    evaluate_heterogeneous as evaluate_viscoelastic_heterogeneous,
)


def test_canonical_abaqus_reference_passes_all_gates():
    summary = evaluate()
    assert summary["passed"], summary["failures"]


def test_canonical_suite_uses_only_conventional_macro_quantities():
    manifest = load_cases()
    serialized = json.dumps(manifest).lower()
    forbidden = ("q99", "percentile", "localization")
    assert not any(term in serialized for term in forbidden)
    assert set(manifest["nonlinear_case"]["compared_quantities"]) == {
        "macro_energy",
        "macro_p12",
        "mean_j",
    }


def test_canonical_result_case_ids_match_manifest():
    manifest = load_cases()
    expected = {case["case_id"] for case in manifest["linear_cases"]}
    for filename in ("canonical_homicsx_results.json", "canonical_abaqus_results.json"):
        data = json.loads((HERE / filename).read_text(encoding="utf-8"))
        assert {case["case_id"] for case in data["linear_cases"]} == expected


def test_viscoelastic_abaqus_reference_passes_all_gates():
    summary = evaluate_viscoelastic()
    assert summary["passed"], summary["failures"]
    heterogeneous = evaluate_viscoelastic_heterogeneous()
    assert heterogeneous["passed"], heterogeneous["failures"]
    three_dimensional = evaluate_viscoelastic_3d()
    assert three_dimensional["passed"], three_dimensional["failures"]


def test_viscoelastic_validation_uses_macro_history_only():
    case = json.loads(
        (HERE / "viscoelastic_case.json").read_text(encoding="utf-8")
    )
    assert set(case["compared_quantities"]) == {
        "time",
        "macro_f12",
        "macro_p12",
        "mean_j",
    }
