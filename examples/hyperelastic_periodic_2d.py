"""Deterministic heterogeneous Neo-Hookean homogenization example."""

from __future__ import annotations

import json

import numpy as np

from homicsx import (
    AdaptiveSettings,
    Inclusion,
    MaterialAssignment,
    MeshSettings,
    NeoHookeanIsotropic,
    NonlinearHomogenizationDriver,
    PhysicalTags,
    ProblemSettings,
    RVEGeometry,
    generate_mesh,
)


def run_example() -> dict:
    geometry = RVEGeometry(
        dim=2,
        domain_size=(1.0, 1.0),
        inclusions=[
            Inclusion(center=(0.5, 0.5), phase_id=1, shape="circle", radii=0.18)
        ],
        metadata={"example": "hyperelastic_periodic_2d"},
    )
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry,
        MeshSettings(
            min_size=0.08,
            max_size=0.16,
            physical_tags=tags,
            periodic_mesh=True,
            verbosity=0,
        ),
    )
    driver = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(
            materials_by_phase={
                0: NeoHookeanIsotropic(young_modulus=10.0, poisson_ratio=0.25),
                1: NeoHookeanIsotropic(young_modulus=100.0, poisson_ratio=0.25),
            }
        ),
        settings=ProblemSettings(
            dim=2,
            kinematics="finite_strain",
            two_dimensional_formulation="plane_strain",
        ),
        physical_tags=tags,
        domain_size=geometry.domain_size,
        matrix_phase_id=0,
        quad_degree=2,
    )

    def simple_shear(load: float) -> np.ndarray:
        deformation = np.eye(2)
        deformation[0, 1] = load
        return deformation

    result = driver.run(
        tangent_every=10_000,
        max_strain=0.04,
        custom_loads={"simple_shear": simple_shear},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=0.5,
            min_step=0.02,
            max_step_ratio=0.5,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )
    history = result.histories["simple_shear"]
    return {
        "steps": len(history["load_param"]),
        "final_macro_p12": float(history["Pbar"][-1][0, 1]),
        "final_macro_energy": float(history["Wbar"][-1]),
        "final_mean_j": float(history["Jbar"][-1]),
        "cells": domain.topology.index_map(domain.topology.dim).size_global,
        "unconstrained_displacement_dofs": 2
        * domain.topology.index_map(0).size_global,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
