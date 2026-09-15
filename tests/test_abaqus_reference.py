import json
from pathlib import Path

import pytest

from validation.abaqus.compare_reference import (
    evaluate,
    relative_difference_percent,
)


VALIDATION_DIR = Path(__file__).parents[1] / "validation" / "abaqus"


def test_relative_difference_uses_homicsx_as_reference():
    assert relative_difference_percent(4.0, 4.1) == pytest.approx(2.5)
    assert relative_difference_percent(4.0, 3.9) == pytest.approx(-2.5)
    with pytest.raises(ValueError, match="zero reference"):
        relative_difference_percent(0.0, 1.0)


def test_compact_abaqus_reference_passes_macroscopic_response_gates():
    summary = evaluate()
    metadata = json.loads((VALIDATION_DIR / "metadata.json").read_text())

    assert summary["case_count"] == 9
    assert summary["passed"]
    assert summary["failures"] == []
    assert summary["maximum_absolute_difference_percent"] == pytest.approx(
        metadata["expected_maximum_absolute_difference_percent"]
    )
