import numpy as np

from homicsx import (
    GeometryInput,
    LinearElasticIsotropic,
    LinearHomogenizationDriver,
    MaterialAssignment,
    MeshSettings,
    PhysicalTags,
    ProblemSettings,
    generate_mesh,
    particulate_geometry_generator,
)


def _run_linear_homogenization(matrix_material, inclusion_material):
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
    mesh_settings = MeshSettings(
        min_size=0.06,
        max_size=0.12,
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
        dim=2,
        kinematics="small_strain",
        two_dimensional_formulation="plane_strain",
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
    result = _run_linear_homogenization(material, material)

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
    assert relative_error < 1.5e-2
