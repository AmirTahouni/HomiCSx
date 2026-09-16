import numpy as np

from homicsx import (
    GeometryInput,
    LinearElasticIsotropic,
    LinearHomogenizationDriver,
    MaterialAssignment,
    MeshSettings,
    PhysicalTags,
    ProblemSettings,
    NeoHookeanIsotropic,
    NonlinearHomogenizationDriver,
    generate_mesh,
    particulate_geometry_generator,
)
from homicsx.core.homogenization import AdaptiveSettings


def _run_linear_homogenization(
    matrix_material,
    inclusion_material,
    *,
    min_size=0.06,
    max_size=0.12,
    dim=2,
):
    geometry_input = GeometryInput(
        dim=dim,
        dispersion="mono",
        shape="circle" if dim == 2 else "sphere",
        volume_fraction=0.05 if dim == 2 else 0.02,
        clearance=0.01,
        domain_size=(1.0,) * dim,
        num_particles=1,
        seed=42,
    )
    geometry = particulate_geometry_generator(geometry_input)
    physical_tags = PhysicalTags()
    mesh_settings = MeshSettings(
        min_size=min_size,
        max_size=max_size,
        physical_tags=physical_tags,
        verbosity=0,
    )
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=mesh_settings,
    )
    assignment = MaterialAssignment(
        materials_by_phase={
            0: matrix_material,
            1: inclusion_material,
        }
    )
    settings = ProblemSettings(
        dim=dim,
        kinematics="small_strain",
        two_dimensional_formulation="plane_strain" if dim == 2 else None,
    )
    return LinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=assignment,
        settings=settings,
        physical_tags=physical_tags,
        domain_size=geometry_input.domain_size,
        matrix_phase_id=0,
    ).run()


def test_linear_homogenization_pipeline_returns_symmetric_stiffness():
    """Exercise the supported public API from geometry through homogenization."""
    result = _run_linear_homogenization(
        LinearElasticIsotropic(young_modulus=1.0, poisson_ratio=0.2),
        LinearElasticIsotropic(young_modulus=10.0, poisson_ratio=0.2),
    )

    assert result.C_hom.shape == (3, 3)
    assert np.all(np.isfinite(result.C_hom))
    relative_skew = np.linalg.norm(result.C_hom - result.C_hom.T) / np.linalg.norm(result.C_hom)
    assert relative_skew < 5e-3


def test_homogeneous_plane_strain_recovers_analytical_stiffness():
    """A geometrically heterogeneous mesh must recover a homogeneous material."""
    young_modulus = 2.5
    poisson_ratio = 0.3
    material = LinearElasticIsotropic(
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    result = _run_linear_homogenization(
        material,
        material,
        min_size=0.025,
        max_size=0.05,
    )

    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    expected = np.array(
        [
            [lame_lambda + 2.0 * shear_modulus, lame_lambda, 0.0],
            [lame_lambda, lame_lambda + 2.0 * shear_modulus, 0.0],
            [0.0, 0.0, shear_modulus],
        ]
    )

    relative_error = np.linalg.norm(result.C_hom - expected) / np.linalg.norm(expected)
    assert relative_error < 5e-3

def test_homogeneous_3d_recovers_analytical_stiffness():
    """The full six-load-case 3D solver must pass a homogeneous patch test."""
    young_modulus = 3.0
    poisson_ratio = 0.25
    material = LinearElasticIsotropic(
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    result = _run_linear_homogenization(
        material,
        material,
        min_size=0.05,
        max_size=0.10,
        dim=3,
    )

    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    expected = np.array(
        [
            [lame_lambda + 2.0 * shear_modulus, lame_lambda, lame_lambda, 0.0, 0.0, 0.0],
            [lame_lambda, lame_lambda + 2.0 * shear_modulus, lame_lambda, 0.0, 0.0, 0.0],
            [lame_lambda, lame_lambda, lame_lambda + 2.0 * shear_modulus, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, shear_modulus, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, shear_modulus, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, shear_modulus],
        ]
    )

    assert result.C_hom.shape == (6, 6)
    relative_error = np.linalg.norm(result.C_hom - expected) / np.linalg.norm(expected)
    # The unstructured 3D periodic mesh carries a small discretization error;
    # the full constitutive tensor must remain within 1.5% in Frobenius norm.
    assert relative_error < 1.5e-2


def test_homogeneous_finite_strain_recovers_neo_hookean_response():
    """A one-step homogeneous RVE must recover analytical W, PK1, and J."""
    geometry_input = GeometryInput(
        dim=2,
        dispersion="mono",
        shape="circle",
        volume_fraction=0.05,
        clearance=0.01,
        domain_size=(1.0, 1.0),
        num_particles=1,
        seed=42,
    )
    geometry = particulate_geometry_generator(geometry_input)
    physical_tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=0.05,
            max_size=0.10,
            physical_tags=physical_tags,
            verbosity=0,
        ),
    )
    material = NeoHookeanIsotropic(young_modulus=2.5, poisson_ratio=0.3)
    assignment = MaterialAssignment(materials_by_phase={0: material, 1: material})
    settings = ProblemSettings(
        dim=2,
        kinematics="finite_strain",
        two_dimensional_formulation="plane_strain",
        petsc_options={
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 20,
        },
    )
    driver = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=assignment,
        settings=settings,
        physical_tags=physical_tags,
        domain_size=geometry_input.domain_size,
        matrix_phase_id=0,
        quad_degree=4,
        enable_hooks=False,
    )

    applied_strain = 0.10

    def uniaxial_tension(load):
        return np.diag([1.0 + load, 1.0])

    result = driver.run(
        tangent_every=2,
        max_strain=applied_strain,
        custom_loads={"uniaxial_tension": uniaxial_tension},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=1.0,
            min_step=1e-8,
            max_step_ratio=1.0,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )

    history = result.histories["uniaxial_tension"]
    assert len(history["load_param"]) == 1
    assert history["converged"][0] > 0

    expected_F = np.diag([1.0 + applied_strain, 1.0])
    expected_energy = material.evaluate_energy(expected_F, dim=2)
    expected_stress = material.get_quadrature_point_stress(None, expected_F, 0)
    expected_J = np.linalg.det(expected_F)

    np.testing.assert_allclose(history["Fbar"][-1], expected_F, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(history["Wbar"][-1], expected_energy, rtol=5e-3)
    np.testing.assert_allclose(history["Pbar"][-1], expected_stress, rtol=5e-3, atol=5e-5)
    np.testing.assert_allclose(history["Jbar"][-1], expected_J, rtol=5e-3)
