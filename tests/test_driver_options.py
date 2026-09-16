from types import SimpleNamespace

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
    driver.mesh_obj = object()
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
    driver.mesh_obj = object()
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
