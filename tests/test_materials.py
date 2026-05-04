import numpy as np

from dolfinx import mesh, fem
from mpi4py import MPI

from homicsx.core.material import (
    LinearElasticIsotropic,
    NeoHookeanIsotropic,
    MaterialAssignment,
)
from homicsx.materials import (
    validate_material_assignment,
    build_linear_elastic_coefficients,
    build_hyperelastic_coefficients,
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

