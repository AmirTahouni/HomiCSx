"""Deterministic end-to-end 3D linear homogenization example."""

from __future__ import annotations

import json

import numpy as np

from homicsx import (
    Inclusion,
    LinearElasticIsotropic,
    LinearHomogenizationDriver,
    MaterialAssignment,
    MeshSettings,
    PhysicalTags,
    ProblemSettings,
    RVEGeometry,
    generate_mesh,
)


def run_example() -> dict:
    geometry = RVEGeometry(
        dim=3,
        domain_size=(1.0, 1.0, 1.0),
        inclusions=[
            Inclusion(
                center=(0.5, 0.5, 0.5),
                phase_id=1,
                shape="sphere",
                radii=0.18,
            )
        ],
        metadata={"example": "linear_periodic_3d"},
    )
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=0.18,
            max_size=0.28,
            physical_tags=tags,
            periodic_mesh=True,
            verbosity=0,
        ),
    )
    driver = LinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(
            materials_by_phase={
                0: LinearElasticIsotropic(young_modulus=1.0, poisson_ratio=0.25),
                1: LinearElasticIsotropic(young_modulus=10.0, poisson_ratio=0.25),
            }
        ),
        settings=ProblemSettings(dim=3, kinematics="small_strain"),
        physical_tags=tags,
        domain_size=geometry.domain_size,
        matrix_phase_id=0,
        mode="complete",
    )
    stiffness = np.asarray(driver.run().C_hom, dtype=float)
    relative_symmetry_error = float(
        np.linalg.norm(stiffness - stiffness.T) / np.linalg.norm(stiffness)
    )
    return {
        "shape": list(stiffness.shape),
        "trace": float(np.trace(stiffness)),
        "relative_symmetry_error": relative_symmetry_error,
        "cell_type": domain.topology.cell_type.name,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
