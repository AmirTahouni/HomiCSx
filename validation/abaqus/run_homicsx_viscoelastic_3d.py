"""Run the HomiCSx side of the homogeneous 3D relaxation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from homicsx import (
    MaterialAssignment,
    MeshSettings,
    NeoHookeanIsotropic,
    NonlinearHomogenizationDriver,
    PhysicalTags,
    ProblemSettings,
    RVEGeometry,
    ViscoelasticGeneralizedMaxwell,
    generate_mesh,
)
from homicsx.core.homogenization import AdaptiveSettings

from .canonical_common import HERE
from .run_homicsx_viscoelastic import load_case


def make_material(manifest):
    equilibrium = manifest["equilibrium_material"]
    return ViscoelasticGeneralizedMaxwell(
        equilibrium_material=NeoHookeanIsotropic(
            young_modulus=equilibrium["young_modulus"],
            poisson_ratio=equilibrium["poisson_ratio"],
        ),
        num_branches=len(manifest["branch_shear_moduli"]),
        shear_moduli=list(manifest["branch_shear_moduli"]),
        relaxation_times=list(manifest["relaxation_times"]),
    )


def direct_history(manifest, case):
    material = make_material(manifest)
    state = material.initialize_state(1)
    deformation = np.eye(3)
    deformation[0, 1] = case["gamma12"]
    count = int(round(case["end_time"] / case["time_increment"]))
    output = []
    for step in range(1, count + 1):
        material.update_state(
            state,
            deformation[np.newaxis, :, :],
            case["time_increment"],
            np.ones(1),
            0,
        )
        stress = material.get_quadrature_point_stress(state, deformation, 0)
        output.append(
            {
                "time": step * case["time_increment"],
                "macro_f12": case["gamma12"],
                "macro_p12": float(stress[0, 1]),
                "mean_j": float(np.linalg.det(deformation)),
            }
        )
    return output


def end_to_end_history(manifest, case):
    lengths = np.array(
        [case["domain"]["lx"], case["domain"]["ly"], case["domain"]["lz"]]
    )
    geometry = RVEGeometry(
        dim=3,
        domain_size=lengths,
        inclusions=[],
        metadata={"validation_case_id": case["case_id"]},
    )
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=case["homicsx_min_size"],
            max_size=case["homicsx_max_size"],
            physical_tags=tags,
            verbosity=0,
            periodic_mesh=True,
        ),
    )
    driver = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(materials_by_phase={0: make_material(manifest)}),
        settings=ProblemSettings(
            dim=3,
            kinematics="finite_strain",
            petsc_options={"snes_rtol": 1e-10, "snes_atol": 1e-12},
        ),
        physical_tags=tags,
        domain_size=lengths,
        matrix_phase_id=0,
        quad_degree=2,
        enable_hooks=False,
    )

    def held_shear(unused_parameter):
        deformation = np.eye(3)
        deformation[0, 1] = case["gamma12"]
        return deformation

    step_ratio = case["time_increment"] / case["end_time"]
    result = driver.run(
        strain_rate=1.0,
        tangent_every=10_000,
        max_strain=case["end_time"],
        custom_loads={"shear_relaxation_3d": held_shear},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=step_ratio,
            min_step=case["time_increment"],
            max_step_ratio=step_ratio,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )
    history = result.histories["shear_relaxation_3d"]
    return [
        {
            "time": float(time),
            "macro_f12": float(deformation[0, 1]),
            "macro_p12": float(stress[0, 1]),
            "mean_j": float(jacobian),
        }
        for time, deformation, stress, jacobian in zip(
            history["load_param"], history["Fbar"], history["Pbar"], history["Jbar"]
        )
    ]


def main(output_path: Path = HERE / "viscoelastic_homicsx_3d_results.json") -> int:
    manifest = load_case()
    case = manifest["three_dimensional_case"]
    output = {
        "schema_version": manifest["schema_version"],
        "case_id": case["case_id"],
        "end_to_end_history": end_to_end_history(manifest, case),
        "discrete_material_history": direct_history(manifest, case),
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Wrote {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
