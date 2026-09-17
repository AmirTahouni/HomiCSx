import pytest

from examples.geometry_generation import run_example as run_geometry_example
from examples.linear_periodic_2d import run_example as run_linear_example
from examples.linear_periodic_3d import run_example as run_linear_3d_example
from examples.hyperelastic_periodic_2d import run_example as run_hyperelastic_example
from examples.viscoelastic_periodic_2d import run_example as run_viscoelastic_example


def test_scriptable_linear_example():
    summary = run_linear_example()
    assert summary["shape"] == [3, 3]
    assert summary["trace"] > 0.0
    assert summary["relative_symmetry_error"] < 2.0e-2


def test_scriptable_linear_example_with_quadrilaterals():
    summary = run_linear_example(quadrilateral=True)
    assert summary["shape"] == [3, 3]
    assert summary["cell_type"] == "quadrilateral"
    assert summary["relative_symmetry_error"] < 2.0e-2


def test_scriptable_geometry_example():
    summary = run_geometry_example()
    assert summary["generated_original_inclusions"] == 4
    assert summary["generated_total_inclusions"] >= 4
    assert summary["generated_phase_ids"] == [0, 1]
    assert summary["prescribed_shape"] == "ellipse"
    assert summary["prescribed_orientation_radians"] == pytest.approx(
        3.141592653589793 / 6.0
    )


def test_scriptable_linear_3d_example():
    summary = run_linear_3d_example()
    assert summary["shape"] == [6, 6]
    assert summary["trace"] > 0.0
    assert summary["cell_type"] == "tetrahedron"
    assert summary["relative_symmetry_error"] < 2.0e-2


def test_scriptable_viscoelastic_example():
    summary = run_viscoelastic_example()
    assert summary["steps"] == 3
    assert summary["final_macro_p12"] < summary["initial_macro_p12"]
    assert summary["hook_matches_history"]


def test_scriptable_hyperelastic_example():
    summary = run_hyperelastic_example()
    assert summary["steps"] == 2
    assert summary["final_macro_p12"] > 0.0
    assert summary["final_macro_energy"] > 0.0
    assert summary["final_mean_j"] == pytest.approx(1.0, rel=5e-3)
