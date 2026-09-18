"""Run the HomiCSx side of the canonical HomiCSx/Abaqus suite.

Run this script inside the documented HomiCSx development environment::

    python -m validation.abaqus.run_homicsx_canonical
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from homicsx import (
    LinearElasticIsotropic,
    LinearHomogenizationDriver,
    MaterialAssignment,
    MeshSettings,
    NeoHookeanIsotropic,
    NonlinearHomogenizationDriver,
    PhysicalTags,
    ProblemSettings,
    generate_mesh,
)
from homicsx.core.homogenization import AdaptiveSettings
from homicsx.core.geometry import Inclusion, RVEGeometry

from .canonical_common import HERE, load_cases, stiffness_to_probe_metrics


def make_geometry(case: dict, domain: dict) -> RVEGeometry:
    """Build an exact deterministic RVE from one manifest case."""
    inclusions = [
        Inclusion(
            center=np.asarray(circle["center"], dtype=float),
            phase_id=1,
            shape="circle",
            radii=np.asarray([circle["radius"]], dtype=float),
            periodic_source_id=circle["periodic_source_id"],
        )
        for circle in case["circles"]
    ]
    physical_area = sum(
        np.pi * circle["radius"] ** 2
        for circle in case["circles"]
        if circle["periodic_source_id"] is None
    )
    domain_size = np.asarray([domain["lx"], domain["ly"]], dtype=float)
    return RVEGeometry(
        dim=2,
        domain_size=domain_size,
        inclusions=inclusions,
        target_volume_fraction=physical_area / float(np.prod(domain_size)),
        realized_volume_fraction=physical_area / float(np.prod(domain_size)),
        metadata={"validation_case_id": case["case_id"]},
    )


def run_linear_case(case: dict, manifest: dict) -> dict:
    geometry = make_geometry(case, manifest["domain"])
    tags = PhysicalTags()
    mesh = manifest["mesh"]
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=mesh["homicsx_min_size"],
            max_size=mesh["homicsx_max_size"],
            physical_tags=tags,
            verbosity=0,
            periodic_mesh=True,
        ),
    )
    materials = manifest["linear_materials"]
    matrix = LinearElasticIsotropic(
        young_modulus=materials["matrix"]["young_modulus"],
        poisson_ratio=materials["matrix"]["poisson_ratio"],
    )
    inclusion_key = case["inclusion_material"]
    inclusion = LinearElasticIsotropic(
        young_modulus=materials[inclusion_key]["young_modulus"],
        poisson_ratio=materials[inclusion_key]["poisson_ratio"],
    )
    result = LinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(materials_by_phase={0: matrix, 1: inclusion}),
        settings=ProblemSettings(
            dim=2,
            kinematics="small_strain",
            two_dimensional_formulation="plane_strain",
        ),
        physical_tags=tags,
        domain_size=geometry.domain_size,
        matrix_phase_id=0,
        mode="complete",
    ).run()
    stiffness = np.asarray(result.C_hom, dtype=float)
    return {
        "case_id": case["case_id"],
        "stiffness": stiffness.tolist(),
        "probes": stiffness_to_probe_metrics(
            stiffness, manifest["macro_strain_probes"]
        ),
    }


def run_nonlinear_case(manifest: dict) -> dict:
    """Run the current nonlinear solver for homogeneous simple shear."""
    case = manifest["nonlinear_case"]
    geometry = make_geometry(manifest["linear_cases"][0], manifest["domain"])
    tags = PhysicalTags()
    mesh_settings = manifest["mesh"]
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=mesh_settings["homicsx_min_size"],
            max_size=mesh_settings["homicsx_max_size"],
            physical_tags=tags,
            verbosity=0,
            periodic_mesh=True,
        ),
    )
    material = NeoHookeanIsotropic(
        young_modulus=case["young_modulus"],
        poisson_ratio=case["poisson_ratio"],
    )
    gamma = case["gamma12"]

    def simple_shear(load):
        deformation = np.eye(2)
        deformation[0, 1] = load
        return deformation

    result = NonlinearHomogenizationDriver(
        mesh_obj=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        assignment=MaterialAssignment(materials_by_phase={0: material, 1: material}),
        settings=ProblemSettings(
            dim=2,
            kinematics="finite_strain",
            two_dimensional_formulation="plane_strain",
        ),
        physical_tags=tags,
        domain_size=geometry.domain_size,
        matrix_phase_id=0,
        quad_degree=2,
    ).run(
        tangent_every=10_000,
        max_strain=gamma,
        custom_loads={"homogeneous_simple_shear": simple_shear},
        from_built_in_loads=[],
        adaptive_settings=AdaptiveSettings(
            initial_step_ratio=1.0,
            min_step=gamma,
            max_step_ratio=1.0,
        ),
        plot_summary=False,
        plot_individual=False,
        save_plots=False,
    )
    history = result.histories["homogeneous_simple_shear"]
    return {
        "case_id": case["case_id"],
        "deformation_gradient": [[1.0, gamma], [0.0, 1.0]],
        "macro_energy": float(history["Wbar"][-1]),
        "macro_p12": float(history["Pbar"][-1][0, 1]),
        "mean_j": float(history["Jbar"][-1]),
        "evaluation": "current HomiCSx nonlinear periodic solver",
    }


def main(output_path: Path = HERE / "canonical_homicsx_results.json") -> int:
    manifest = load_cases()
    output = {
        "schema_version": manifest["schema_version"],
        "linear_cases": [
            run_linear_case(case, manifest) for case in manifest["linear_cases"]
        ],
        "nonlinear_case": run_nonlinear_case(manifest),
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Wrote {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
