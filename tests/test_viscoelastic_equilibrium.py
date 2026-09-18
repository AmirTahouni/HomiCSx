import numpy as np

from homicsx import (
    GeometryInput,
    MaterialAssignment,
    MeshSettings,
    NeoHookeanIsotropic,
    NonlinearHomogenizationDriver,
    PhysicalTags,
    ProblemSettings,
    generate_mesh,
    particulate_geometry_generator,
)
from homicsx.core.homogenization import AdaptiveSettings
from homicsx.core.material import ViscoelasticGeneralizedMaxwell


def _final_fluctuation_for_rate(
    strain_rate, tangent_every=10_000, shear=0.02
):
    geometry_input = GeometryInput(
        dim=2,
        dispersion="mono",
        shape="circle",
        volume_fraction=0.12,
        clearance=0.01,
        domain_size=(1.0, 1.0),
        num_particles=1,
        seed=7,
    )
    geometry = particulate_geometry_generator(geometry_input)
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=0.08,
            max_size=0.16,
            physical_tags=tags,
            verbosity=0,
        ),
    )
    equilibrium = NeoHookeanIsotropic(young_modulus=10.0, poisson_ratio=0.25)
    matrix = ViscoelasticGeneralizedMaxwell(
        equilibrium_material=equilibrium,
        num_branches=1,
        shear_moduli=[8.0],
        relaxation_times=[0.1],
    )
    inclusion = NeoHookeanIsotropic(young_modulus=100.0, poisson_ratio=0.25)
    driver = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(materials_by_phase={0: matrix, 1: inclusion}),
        settings=ProblemSettings(
            dim=2,
            kinematics="finite_strain",
            two_dimensional_formulation="plane_strain",
            petsc_options={"snes_rtol": 1e-10, "snes_atol": 1e-12},
        ),
        physical_tags=tags,
        domain_size=geometry_input.domain_size,
        matrix_phase_id=0,
        quad_degree=2,
        enable_hooks=True,
    )
    captured = []

    def capture_converged_field(data):
        captured.append(data.u.x.array.copy())

    driver.add_post_convergence_hook(capture_converged_field)

    def held_shear(unused_parameter):
        return np.array([[1.0, shear], [0.0, 1.0]])

    result = driver.run(
        strain_rate=strain_rate,
        tangent_every=tangent_every,
        max_strain=0.01,
        custom_loads={"held_shear": held_shear},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=1.0,
            min_step=0.01,
            max_step_ratio=1.0,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )
    assert len(captured) == 1
    history = result.histories["held_shear"]
    return captured[0], history["Ceff"][-1], history["Pbar"][-1]


def test_maxwell_branches_change_heterogeneous_equilibrium_field_with_rate():
    fast, _, _ = _final_fluctuation_for_rate(strain_rate=10.0)  # dt = 0.001
    slow, _, _ = _final_fluctuation_for_rate(strain_rate=0.01)  # dt = 1.0
    difference = np.linalg.norm(fast - slow)
    scale = max(np.linalg.norm(fast), np.linalg.norm(slow))
    assert difference / scale > 1.0e-2


def test_heterogeneous_viscoelastic_step_tangent_is_finite():
    _, tangent, _ = _final_fluctuation_for_rate(
        strain_rate=10.0, tangent_every=1
    )
    assert tangent.shape == (4, 4)
    assert np.all(np.isfinite(tangent))
    assert np.linalg.norm(tangent) > 1.0


def test_heterogeneous_viscoelastic_tangent_matches_fresh_step_replays():
    delta = 1.0e-5
    _, tangent, _ = _final_fluctuation_for_rate(
        strain_rate=10.0, tangent_every=1, shear=0.02
    )
    _, _, stress_plus = _final_fluctuation_for_rate(
        strain_rate=10.0, shear=0.02 + delta
    )
    _, _, stress_minus = _final_fluctuation_for_rate(
        strain_rate=10.0, shear=0.02 - delta
    )
    independently_replayed_column = (
        stress_plus.reshape(-1) - stress_minus.reshape(-1)
    ) / (2.0 * delta)
    np.testing.assert_allclose(
        tangent[:, 1], independently_replayed_column, rtol=2.0e-4, atol=2.0e-5
    )


def _final_3d_fluctuation_for_rate(strain_rate):
    geometry_input = GeometryInput(
        dim=3,
        dispersion="mono",
        shape="sphere",
        volume_fraction=0.04,
        clearance=0.02,
        domain_size=(1.0, 1.0, 1.0),
        num_particles=1,
        seed=11,
    )
    geometry = particulate_geometry_generator(geometry_input)
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=0.18,
            max_size=0.30,
            physical_tags=tags,
            verbosity=0,
            periodic_mesh=True,
        ),
    )
    matrix = ViscoelasticGeneralizedMaxwell(
        equilibrium_material=NeoHookeanIsotropic(
            young_modulus=10.0, poisson_ratio=0.25
        ),
        num_branches=1,
        shear_moduli=[8.0],
        relaxation_times=[0.1],
    )
    inclusion = NeoHookeanIsotropic(young_modulus=100.0, poisson_ratio=0.25)
    driver = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(materials_by_phase={0: matrix, 1: inclusion}),
        settings=ProblemSettings(
            dim=3,
            kinematics="finite_strain",
            petsc_options={"snes_rtol": 1e-10, "snes_atol": 1e-12},
        ),
        physical_tags=tags,
        domain_size=geometry_input.domain_size,
        matrix_phase_id=0,
        quad_degree=2,
        enable_hooks=True,
    )
    captured = []
    driver.add_post_convergence_hook(lambda data: captured.append(data.u.x.array.copy()))

    def held_shear(unused_parameter):
        deformation = np.eye(3)
        deformation[0, 1] = 0.015
        return deformation

    driver.run(
        strain_rate=strain_rate,
        tangent_every=10_000,
        max_strain=0.01,
        custom_loads={"held_shear_3d": held_shear},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=1.0,
            min_step=0.01,
            max_step_ratio=1.0,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )
    assert len(captured) == 1
    return captured[0]


def test_maxwell_branches_change_3d_heterogeneous_equilibrium_field_with_rate():
    fast = _final_3d_fluctuation_for_rate(strain_rate=10.0)
    slow = _final_3d_fluctuation_for_rate(strain_rate=0.01)
    difference = np.linalg.norm(fast - slow)
    scale = max(np.linalg.norm(fast), np.linalg.norm(slow))
    assert difference / scale > 1.0e-2
