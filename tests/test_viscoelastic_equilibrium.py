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
    strain_rate,
    tangent_every=10_000,
    tangent_delta=1.0e-6,
    shear=0.02,
    deformation=None,
    step_ratio=1.0,
    mutate_post_stress_snapshot=False,
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

    if mutate_post_stress_snapshot:
        def mutate_detached_snapshot(data):
            for phase_states in data.material_states.values():
                for material_state in phase_states.values():
                    for name in material_state.state_variable_names:
                        material_state.get_state(name)[...] = 1.0e6

        driver.add_post_stress_hook(mutate_detached_snapshot)

    def held_shear(unused_parameter):
        if deformation is not None:
            return np.asarray(deformation, dtype=float)
        return np.array([[1.0, shear], [0.0, 1.0]])

    result = driver.run(
        strain_rate=strain_rate,
        tangent_every=tangent_every,
        tangent_delta=tangent_delta,
        max_strain=0.01,
        custom_loads={"held_shear": held_shear},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=step_ratio,
            min_step=1.0e-6,
            max_step_ratio=1.0,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )
    history = result.histories["held_shear"]
    assert len(captured) == len(history["load_param"])
    return captured[-1], history["Ceff"][-1], history["Pbar"][-1]


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


def test_all_heterogeneous_viscoelastic_tangent_columns_match_fresh_step_replays():
    delta = 1.0e-5
    base_deformation = np.array([[1.0, 0.02], [0.0, 1.0]])
    _, tangent, _ = _final_fluctuation_for_rate(
        strain_rate=10.0,
        tangent_every=1,
        tangent_delta=delta,
        deformation=base_deformation,
    )
    replayed = np.zeros_like(tangent)
    for column in range(4):
        plus = base_deformation.copy().reshape(-1)
        minus = base_deformation.copy().reshape(-1)
        plus[column] += delta
        minus[column] -= delta
        _, _, stress_plus = _final_fluctuation_for_rate(
            strain_rate=10.0, deformation=plus.reshape(2, 2)
        )
        _, _, stress_minus = _final_fluctuation_for_rate(
            strain_rate=10.0, deformation=minus.reshape(2, 2)
        )
        replayed[:, column] = (
            stress_plus.reshape(-1) - stress_minus.reshape(-1)
        ) / (2.0 * delta)
    np.testing.assert_allclose(tangent, replayed, rtol=3.0e-4, atol=3.0e-5)


def test_viscoelastic_tangent_is_stable_to_perturbation_size():
    _, fine, _ = _final_fluctuation_for_rate(
        strain_rate=10.0, tangent_every=1, tangent_delta=1.0e-6
    )
    _, coarse, _ = _final_fluctuation_for_rate(
        strain_rate=10.0, tangent_every=1, tangent_delta=1.0e-5
    )
    np.testing.assert_allclose(fine, coarse, rtol=2.0e-4, atol=2.0e-5)


def test_real_viscoelastic_tangent_uses_one_sided_fallback():
    # The negative F_00 perturbation has det(F) < 0 and cannot be solved for the
    # logarithmic Neo-Hookean energy.  The driver must retain the valid forward
    # perturbation and form a finite one-sided derivative instead.
    deformation = np.array([[0.1, 0.0], [0.0, 1.0]])
    _, tangent, _ = _final_fluctuation_for_rate(
        strain_rate=10.0,
        tangent_every=1,
        tangent_delta=0.2,
        deformation=deformation,
    )
    assert np.all(np.isfinite(tangent[:, 0]))


def test_multistep_viscoelastic_tangent_uses_nontrivial_previous_state():
    _, tangent, _ = _final_fluctuation_for_rate(
        strain_rate=10.0, tangent_every=1, step_ratio=0.5
    )
    assert np.all(np.isfinite(tangent))


def test_post_stress_state_snapshot_cannot_contaminate_tangent():
    _, reference_tangent, reference_stress = _final_fluctuation_for_rate(
        strain_rate=10.0, tangent_every=1
    )
    _, mutated_tangent, mutated_stress = _final_fluctuation_for_rate(
        strain_rate=10.0,
        tangent_every=1,
        mutate_post_stress_snapshot=True,
    )
    np.testing.assert_allclose(mutated_stress, reference_stress, atol=1.0e-12)
    np.testing.assert_allclose(mutated_tangent, reference_tangent, atol=1.0e-10)


def _final_3d_fluctuation_for_rate(
    strain_rate, tangent_every=10_000, tangent_delta=1.0e-6, shear=0.015
):
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
        deformation[0, 1] = shear
        return deformation

    result = driver.run(
        strain_rate=strain_rate,
        tangent_every=tangent_every,
        tangent_delta=tangent_delta,
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
    history = result.histories["held_shear_3d"]
    return captured[0], history["Ceff"][-1], history["Pbar"][-1]


def test_maxwell_branches_change_3d_heterogeneous_equilibrium_field_with_rate():
    fast, _, _ = _final_3d_fluctuation_for_rate(strain_rate=10.0)
    slow, _, _ = _final_3d_fluctuation_for_rate(strain_rate=0.01)
    difference = np.linalg.norm(fast - slow)
    scale = max(np.linalg.norm(fast), np.linalg.norm(slow))
    assert difference / scale > 1.0e-2


def test_3d_viscoelastic_shear_tangent_matches_fresh_step_replays():
    delta = 1.0e-5
    _, tangent, _ = _final_3d_fluctuation_for_rate(
        strain_rate=10.0, tangent_every=1, tangent_delta=delta
    )
    _, _, stress_plus = _final_3d_fluctuation_for_rate(
        strain_rate=10.0, shear=0.015 + delta
    )
    _, _, stress_minus = _final_3d_fluctuation_for_rate(
        strain_rate=10.0, shear=0.015 - delta
    )
    replayed_column = (
        stress_plus.reshape(-1) - stress_minus.reshape(-1)
    ) / (2.0 * delta)
    np.testing.assert_allclose(
        tangent[:, 1], replayed_column, rtol=5.0e-4, atol=5.0e-5
    )
