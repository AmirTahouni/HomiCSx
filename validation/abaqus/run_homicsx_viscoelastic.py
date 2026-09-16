"""Run the HomiCSx side of the homogeneous viscoelastic relaxation case."""

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
    generate_mesh,
)
from homicsx.core.homogenization import AdaptiveSettings
from homicsx.core.material import ViscoelasticGeneralizedMaxwell

from .canonical_common import HERE
from .run_homicsx_canonical import make_geometry


CASE_PATH = HERE / "viscoelastic_case.json"


def load_case() -> dict:
    return json.loads(CASE_PATH.read_text(encoding="utf-8"))


def discrete_material_history(case: dict) -> list[dict]:
    """Evaluate the exact constitutive recurrence used by HomiCSx."""
    equilibrium_data = case["equilibrium_material"]
    equilibrium = NeoHookeanIsotropic(
        young_modulus=equilibrium_data["young_modulus"],
        poisson_ratio=equilibrium_data["poisson_ratio"],
    )
    material = ViscoelasticGeneralizedMaxwell(
        equilibrium_material=equilibrium,
        num_branches=len(case["branch_shear_moduli"]),
        shear_moduli=case["branch_shear_moduli"],
        relaxation_times=case["relaxation_times"],
    )
    state = material.initialize_state(1)
    gamma = case["gamma12"]
    deformation = np.array([[1.0, gamma], [0.0, 1.0]])
    count = int(round(case["end_time"] / case["time_increment"]))
    history = []
    for step in range(1, count + 1):
        material.update_state(
            state,
            deformation[np.newaxis, :, :],
            case["time_increment"],
            np.ones(1),
            0,
        )
        stress = material.get_quadrature_point_stress(state, deformation, 0)
        history.append(
            {
                "time": step * case["time_increment"],
                "macro_f12": gamma,
                "macro_p12": float(stress[0, 1]),
                "mean_j": float(np.linalg.det(deformation)),
            }
        )
    return history


def run_end_to_end(case: dict) -> list[dict]:
    """Exercise meshing, periodic constraints, state updates, and averaging."""
    circle_case = {
        "case_id": case["case_id"],
        "circles": [
            {
                "center": [case["domain"]["lx"] / 2.0, case["domain"]["ly"] / 2.0],
                "radius": 0.18,
                "periodic_source_id": None,
            }
        ],
    }
    geometry = make_geometry(circle_case, case["domain"])
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=case["mesh"]["homicsx_min_size"],
            max_size=case["mesh"]["homicsx_max_size"],
            physical_tags=tags,
            verbosity=0,
            periodic_mesh=True,
        ),
    )
    equilibrium_data = case["equilibrium_material"]

    def material():
        return ViscoelasticGeneralizedMaxwell(
            equilibrium_material=NeoHookeanIsotropic(
                young_modulus=equilibrium_data["young_modulus"],
                poisson_ratio=equilibrium_data["poisson_ratio"],
            ),
            num_branches=len(case["branch_shear_moduli"]),
            shear_moduli=list(case["branch_shear_moduli"]),
            relaxation_times=list(case["relaxation_times"]),
        )

    driver = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(materials_by_phase={0: material(), 1: material()}),
        settings=ProblemSettings(
            dim=2,
            kinematics="finite_strain",
            two_dimensional_formulation="plane_strain",
            petsc_options={
                "snes_rtol": 1e-10,
                "snes_atol": 1e-12,
                "snes_max_it": 20,
            },
        ),
        physical_tags=tags,
        domain_size=geometry.domain_size,
        matrix_phase_id=0,
        quad_degree=4,
        enable_hooks=False,
    )
    gamma = case["gamma12"]

    def held_simple_shear(unused_time_parameter):
        return np.array([[1.0, gamma], [0.0, 1.0]])

    step_ratio = case["time_increment"] / case["end_time"]
    result = driver.run(
        strain_rate=1.0,
        tangent_every=10_000,
        max_strain=case["end_time"],
        custom_loads={"shear_relaxation": held_simple_shear},
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
    history = result.histories["shear_relaxation"]
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


def main(output_path: Path = HERE / "viscoelastic_homicsx_results.json") -> int:
    case = load_case()
    output = {
        "schema_version": case["schema_version"],
        "case_id": case["case_id"],
        "end_to_end_history": run_end_to_end(case),
        "discrete_material_history": discrete_material_history(case),
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Wrote {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
