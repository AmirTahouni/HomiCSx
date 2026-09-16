import numpy as np
import pytest

from dolfinx import mesh, fem
from mpi4py import MPI

from homicsx.core.material import (
    LinearElasticIsotropic,
    NeoHookeanIsotropic,
    MaterialAssignment,
    ViscoelasticGeneralizedMaxwell,
)
from homicsx.materials.assignment import (
    _validate_material_assignment as validate_material_assignment,
)
from homicsx.materials.coefficients import (
    _build_linear_elastic_coefficients as build_linear_elastic_coefficients,
    _build_hyperelastic_coefficients as build_hyperelastic_coefficients,
)
from homicsx.mesh.tagging import PhysicalTags


def create_simple_test_mesh():
    """
    Create a simple 2D mesh with artificial cell tags:
    half matrix, half inclusion.
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    num_cells = domain.topology.index_map(tdim).size_local

    values = np.zeros(num_cells, dtype=np.int32)

    connectivity = domain.topology.connectivity(tdim, 0)
    x = domain.geometry.x

    for cell in range(num_cells):
        vertices = connectivity.links(cell)
        coords = x[vertices]
        center = coords.mean(axis=0)

        if center[0] < 0.5:
            values[cell] = 1  # matrix
        else:
            values[cell] = 11  # phase 1 (offset=10)

    cell_tags = mesh.meshtags(
        domain,
        tdim,
        np.arange(num_cells, dtype=np.int32),
        values,
    )

    return domain, cell_tags


@pytest.mark.parametrize("material_type", [LinearElasticIsotropic, NeoHookeanIsotropic])
@pytest.mark.parametrize(
    "young_modulus, poisson_ratio",
    [(0.0, 0.3), (-1.0, 0.3), (1.0, -1.0), (1.0, 0.5), (np.nan, 0.3)],
)
def test_isotropic_materials_reject_invalid_elastic_constants(
    material_type, young_modulus, poisson_ratio
):
    with pytest.raises(ValueError):
        material_type(young_modulus=young_modulus, poisson_ratio=poisson_ratio)


@pytest.mark.parametrize(
    "num_branches, shear_moduli, relaxation_times",
    [
        (0, [], []),
        (1, [], [1.0]),
        (1, [1.0], []),
        (1, [0.0], [1.0]),
        (1, [1.0], [0.0]),
        (1, [np.nan], [1.0]),
    ],
)
def test_generalized_maxwell_rejects_invalid_branch_data(
    num_branches, shear_moduli, relaxation_times
):
    with pytest.raises(ValueError):
        ViscoelasticGeneralizedMaxwell(
            equilibrium_material=NeoHookeanIsotropic(10.0, 0.25),
            num_branches=num_branches,
            shear_moduli=shear_moduli,
            relaxation_times=relaxation_times,
        )


def test_material_assignment_and_coefficients():
    """
    Test:
    - material assignment
    - DG0 coefficient creation
    - mixed material support
    """
    mesh_, cell_tags = create_simple_test_mesh()

    physical_tags = PhysicalTags()

    # --- materials ---
    mat_matrix = NeoHookeanIsotropic(young_modulus=1.0, poisson_ratio=0.45)
    mat_particle = LinearElasticIsotropic(young_modulus=10.0, poisson_ratio=0.3)

    assignment = MaterialAssignment(
        materials_by_phase={
            0: mat_matrix,   # matrix
            1: mat_particle  # inclusion
        }
    )

    # --- validate ---
    validate_material_assignment(assignment)

    # --- coefficients ---
    linear_coeffs = build_linear_elastic_coefficients(
        mesh_,
        cell_tags,
        assignment,
        physical_tags,
        matrix_phase_id=0,
    )

    hyper_coeffs = build_hyperelastic_coefficients(
        mesh_,
        cell_tags,
        assignment,
        physical_tags,
        matrix_phase_id=0,
    )

    # --- checks ---
    assert isinstance(linear_coeffs.young_modulus, fem.Function)
    assert isinstance(hyper_coeffs.mu, fem.Function)

    # Make sure fields are non-zero somewhere
    assert np.any(linear_coeffs.young_modulus.x.array > 0.0)
    assert np.any(hyper_coeffs.mu.x.array > 0.0)


@pytest.mark.parametrize(
    "deformation_gradient",
    [
        np.array([[1.20, 0.08], [0.03, 0.92]]),
        np.array(
            [
                [1.12, 0.05, 0.01],
                [0.02, 0.95, 0.04],
                [0.00, 0.03, 1.08],
            ]
        ),
    ],
)
def test_neo_hookean_pk1_is_energy_gradient(deformation_gradient):
    """PK1 must equal the derivative of strain energy with respect to F."""
    material = NeoHookeanIsotropic(young_modulus=7.5, poisson_ratio=0.32)
    dim = deformation_gradient.shape[0]
    analytical_stress = material.get_quadrature_point_stress(
        state=None,
        F=deformation_gradient,
        quad_point_idx=0,
    )

    step = 1e-7
    numerical_stress = np.zeros_like(deformation_gradient)
    for i in range(dim):
        for j in range(dim):
            perturbation = np.zeros_like(deformation_gradient)
            perturbation[i, j] = step
            energy_plus = material.evaluate_energy(
                deformation_gradient + perturbation,
                dim,
            )
            energy_minus = material.evaluate_energy(
                deformation_gradient - perturbation,
                dim,
            )
            numerical_stress[i, j] = (energy_plus - energy_minus) / (2.0 * step)

    np.testing.assert_allclose(
        analytical_stress,
        numerical_stress,
        rtol=2e-7,
        atol=2e-8,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_neo_hookean_reference_configuration_is_stress_and_energy_free(dim):
    material = NeoHookeanIsotropic(young_modulus=7.5, poisson_ratio=0.32)
    identity = np.eye(dim)

    assert material.evaluate_energy(identity, dim) == pytest.approx(0.0, abs=1e-14)
    stress = material.get_quadrature_point_stress(None, identity, 0)
    np.testing.assert_allclose(stress, np.zeros((dim, dim)), atol=1e-14)


def test_generalized_maxwell_recoverable_energy_relaxes_toward_equilibrium():
    equilibrium = NeoHookeanIsotropic(young_modulus=10.0, poisson_ratio=0.25)
    material = ViscoelasticGeneralizedMaxwell(
        equilibrium_material=equilibrium,
        num_branches=1,
        shear_moduli=[3.0],
        relaxation_times=[0.5],
    )
    state = material.initialize_state(1)
    deformation = np.array([[1.0, 0.1], [0.0, 1.0]])
    initial_energy = material.get_quadrature_point_energy(state, deformation, 0)
    material.update_state(
        state,
        deformation[np.newaxis, :, :],
        dt=2.5,
        quad_weights=np.ones(1),
        cell_idx=0,
    )
    relaxed_energy = material.get_quadrature_point_energy(state, deformation, 0)
    equilibrium_energy = equilibrium.evaluate_energy(deformation, dim=2)
    assert initial_energy > relaxed_energy > equilibrium_energy

