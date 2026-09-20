from types import SimpleNamespace

import dolfinx
import numpy as np
import pytest

from homicsx.core.homogenization import AdaptiveSettings
from homicsx.homogenization import driver as driver_module


def test_linear_driver_run_honors_explicit_overrides(monkeypatch):
    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(driver_module, "_solve_linear_homogenization", fake_solver)
    driver = driver_module.LinearHomogenizationDriver.__new__(
        driver_module.LinearHomogenizationDriver
    )
    driver.mesh_obj = SimpleNamespace(comm=SimpleNamespace(size=1))
    driver.cell_tags = object()
    driver.facet_tags = object()
    driver.assignment = object()
    driver.settings = SimpleNamespace(petsc_options={"configured": True})
    driver.physical_tags = object()
    driver.domain_size = (1.0, 1.0)
    driver.matrix_phase_id = 0
    driver.mode = "complete"

    result = driver.run(mode="partial", petsc_options={"ksp_type": "preonly"})

    assert result == "result"
    assert captured["mode"] == "partial"
    assert captured["petsc_options"] == {"ksp_type": "preonly"}


def test_linear_driver_run_falls_back_to_configured_options(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        driver_module,
        "_solve_linear_homogenization",
        lambda **kwargs: captured.update(kwargs),
    )
    driver = driver_module.LinearHomogenizationDriver.__new__(
        driver_module.LinearHomogenizationDriver
    )
    driver.mesh_obj = SimpleNamespace(comm=SimpleNamespace(size=1))
    driver.cell_tags = object()
    driver.facet_tags = object()
    driver.assignment = object()
    driver.settings = SimpleNamespace(petsc_options={"configured": True})
    driver.physical_tags = object()
    driver.domain_size = (1.0, 1.0)
    driver.matrix_phase_id = 0
    driver.mode = "partial"

    driver.run()

    assert captured["mode"] == "partial"
    assert captured["petsc_options"] == {"configured": True}


def test_linear_driver_rejects_distributed_execution():
    driver = driver_module.LinearHomogenizationDriver.__new__(
        driver_module.LinearHomogenizationDriver
    )
    driver.mesh_obj = SimpleNamespace(comm=SimpleNamespace(size=2))

    try:
        driver.run()
    except RuntimeError as exc:
        assert "one MPI rank only" in str(exc)
    else:
        raise AssertionError("distributed execution must be rejected explicitly")


def _history_driver(cell_type):
    driver = driver_module.NonlinearHomogenizationDriver.__new__(
        driver_module.NonlinearHomogenizationDriver
    )
    driver.mesh_obj = SimpleNamespace(
        comm=SimpleNamespace(size=1),
        topology=SimpleNamespace(cell_type=cell_type),
    )
    driver.assignment = SimpleNamespace(has_history_dependence=lambda: True)
    return driver


def test_history_dependent_driver_rejects_quadrilateral_cells():
    driver = _history_driver(dolfinx.mesh.CellType.quadrilateral)
    with pytest.raises(NotImplementedError, match="triangle or tetrahedron"):
        driver.run(plot_summary=False)


def test_history_dependent_driver_rejects_xdmf_field_reconstruction():
    driver = _history_driver(dolfinx.mesh.CellType.triangle)
    with pytest.raises(NotImplementedError, match="history-independent"):
        driver.run(xdmf_opt=True, plot_summary=False)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_nonlinear_driver_rejects_invalid_tangent_interval(value):
    driver = _history_driver(dolfinx.mesh.CellType.triangle)
    with pytest.raises((TypeError, ValueError), match="tangent_every"):
        driver.run(tangent_every=value, plot_summary=False)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_nonlinear_driver_rejects_invalid_tangent_delta(value):
    driver = _history_driver(dolfinx.mesh.CellType.triangle)
    with pytest.raises(ValueError, match="tangent_delta"):
        driver.run(tangent_delta=value, plot_summary=False)


@pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf])
def test_nonlinear_driver_rejects_invalid_max_strain(value):
    driver = _history_driver(dolfinx.mesh.CellType.triangle)
    with pytest.raises(ValueError, match="max_strain"):
        driver.run(max_strain=value, plot_summary=False)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_nonlinear_driver_rejects_invalid_strain_rate(value):
    driver = _history_driver(dolfinx.mesh.CellType.triangle)
    with pytest.raises(ValueError, match="strain_rate"):
        driver.run(strain_rate=value, plot_summary=False)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"initial_step_ratio": 0.0}, "step ratios"),
        ({"initial_step_ratio": 0.3, "max_step_ratio": 0.2}, "step ratios"),
        ({"min_step": 0.0}, "min_step"),
        ({"target_iters_min": 9, "target_iters_max": 8}, "iteration targets"),
        ({"growth_factor": 1.0}, "growth_factor"),
        ({"cutback_factor": 1.0}, "cutback_factor"),
    ],
)
def test_adaptive_settings_reject_invalid_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveSettings(**kwargs)
