import json

from validation.abaqus.canonical_common import HERE, load_cases
from validation.abaqus.compare_canonical import evaluate


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
