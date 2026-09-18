from types import SimpleNamespace

import dolfinx
import pytest

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
