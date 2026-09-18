"""Deterministic heterogeneous generalized-Maxwell relaxation example."""

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
    PostStressData,
    ProblemSettings,
    RVEGeometry,
    ViscoelasticGeneralizedMaxwell,
    generate_mesh,
)


def run_example() -> dict:
    geometry = RVEGeometry(
        dim=2,
        domain_size=(1.0, 1.0),
        inclusions=[
            Inclusion(center=(0.5, 0.5), phase_id=1, shape="circle", radii=0.18)
        ],
        metadata={"example": "viscoelastic_periodic_2d"},
    )
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=0.08,
            max_size=0.16,
            physical_tags=tags,
            periodic_mesh=True,
            verbosity=0,
        ),
    )
    matrix = ViscoelasticGeneralizedMaxwell(
        equilibrium_material=NeoHookeanIsotropic(
            young_modulus=10.0, poisson_ratio=0.25
        ),
        num_branches=1,
        shear_moduli=[4.0],
        relaxation_times=[0.2],
    )
    driver = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(
            materials_by_phase={
                0: matrix,
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
        enable_hooks=True,
    )
    hook_stress = []

    def collect_macro_stress(data: PostStressData) -> None:
        hook_stress.append(float(data.P_avg[0, 1]))

    driver.add_post_stress_hook(collect_macro_stress)

    def held_shear(unused_parameter: float) -> np.ndarray:
        deformation = np.eye(2)
        deformation[0, 1] = 0.02
        return deformation

    result = driver.run(
        strain_rate=1.0,
        tangent_every=10_000,
        max_strain=0.3,
        custom_loads={"held_shear": held_shear},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=1.0 / 3.0,
            min_step=0.1,
            max_step_ratio=1.0 / 3.0,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )
    history = result.histories["held_shear"]
    stresses = [float(stress[0, 1]) for stress in history["Pbar"]]
    return {
        "steps": len(stresses),
        "initial_macro_p12": stresses[0],
        "final_macro_p12": stresses[-1],
        "hook_matches_history": bool(np.allclose(hook_stress, stresses)),
        "cells": domain.topology.index_map(domain.topology.dim).size_global,
        "unconstrained_displacement_dofs": 2
        * domain.topology.index_map(0).size_global,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
