import numpy as np

from mpi4py import MPI
from dolfinx import mesh

from homicsx.core.mesh import PhysicalTags
from homicsx.core.material import (
    LinearElasticIsotropic,
    NeoHookeanIsotropic,
    MaterialAssignment,
)
from homicsx.core.fem import ProblemSettings

from homicsx.geometry import generate_mono_3d

from homicsx.mesh import generate_mesh

from homicsx.materials import (
    build_linear_elastic_coefficients, 
    build_hyperelastic_coefficients, 
    validate_material_assignment
)

from homicsx.fem import (
    build_problem,
    build_anchor_and_periodic_constraints,
    solve_problem,
)


def create_test_mesh_and_tags():
    geometry = generate_mono_3d(
        volume_fraction=0.1,
        num_particles=10,
        clearance=0.015,
        domain_size=(1.0, 1.0, 1.0),
        shape="sphere",
        axis_ratios=(1, 1, 1)
    )

    mesh, ct, ft = generate_mesh(
        geometry=geometry,
        min_size=0.03,
        max_size=0.04
    )

    return mesh, ct, ft


def test_full_fem_pipeline():
    """
    Full FEM test:
    - mixed materials
    - periodic constraints
    - assembly
    - solve
    """
    mesh_, cell_tags, facet_tags = create_test_mesh_and_tags()

    physical_tags = PhysicalTags()

    # --- materials ---
    # mat_matrix = NeoHookeanIsotropic(young_modulus=1.0, poisson_ratio=0.45)
    mat_matrix = LinearElasticIsotropic(young_modulus=1.0, poisson_ratio=0.45)
    mat_particle = LinearElasticIsotropic(young_modulus=10.0, poisson_ratio=0.3)

    assignment = MaterialAssignment(
        materials_by_phase={
            0: mat_matrix,
            1: mat_particle,
        }
    )

    validate_material_assignment(assignment)

    # --- FEM settings ---
    settings = ProblemSettings(
        dim=3,
        kinematics="finite_strain",  # hyperelastic active
        two_dimensional_formulation=None,
    )

    # --- function space ---
    from homicsx.fem.assembly import build_displacement_space

    V = build_displacement_space(mesh_, settings)

    # --- constraints ---
    domain_size = (1, 1, 1)

    bcs, mpc = build_anchor_and_periodic_constraints(
        mesh=mesh_,
        facet_tags=facet_tags,
        V=V,
        domain_size=domain_size,
        physical_tags=physical_tags,
        anchor_point=(0, 0, 0),
    )

    # --- build problem ---
    problem = build_problem(
        mesh=mesh_,
        cell_tags=cell_tags,
        assignment=assignment,
        settings=settings,
        physical_tags=physical_tags,
        bcs=bcs,
        mpc=mpc,
        matrix_phase_id=0,
    )

    # --- solve ---
    result = solve_problem(problem)

    # --- checks ---
    assert result.converged
    assert result.solution is not None



