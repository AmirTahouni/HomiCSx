"""Generate deterministic XDMF fields for the manuscript ParaView figure."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import meshio
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


OUTPUT = Path(__file__).resolve().parent / "paraview_data"


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    geometry = RVEGeometry(
        dim=2,
        domain_size=(1.0, 1.0),
        inclusions=[
            Inclusion(center=(0.5, 0.5), phase_id=1, shape="circle", radii=0.2)
        ],
        metadata={"figure": "paraview_fields"},
    )
    tags = PhysicalTags()
    domain, cell_tags, facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=0.035,
            max_size=0.075,
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

    previous_directory = Path.cwd()
    try:
        os.chdir(OUTPUT)
        driver.run(
            tangent_every=10_000,
            max_strain=0.25,
            custom_loads={"simple_shear": simple_shear},
            from_built_in_loads=[],
            adaptive_settings=AdaptiveSettings(
                initial_step_ratio=0.2,
                min_step=0.05,
                max_step_ratio=0.2,
            ),
            xdmf_opt=True,
            csv_opt=False,
            plot_summary=False,
            plot_individual=False,
            save_plots=False,
        )
    finally:
        os.chdir(previous_directory)
    h5_path = OUTPUT / "simple_shear_results.h5"
    with h5py.File(h5_path, "r") as data:
        points_2d = np.asarray(data["/Mesh/mesh/geometry"])
        points = np.column_stack((points_2d, np.zeros(len(points_2d))))
        triangles = np.asarray(data["/Mesh/mesh/topology"], dtype=int)
        time_key = sorted(
            data["/Function/total_disp"].keys(),
            key=lambda name: float(name.replace("_", ".", 1)),
        )[-1]
        displacement = np.asarray(data[f"/Function/total_disp/{time_key}"])
        von_mises = np.asarray(data[f"/Function/von_Mises/{time_key}"]).reshape(-1)
        energy = np.asarray(data[f"/Function/energy_density/{time_key}"]).reshape(-1)
    meshio.write(
        OUTPUT / "simple_shear_final.vtu",
        meshio.Mesh(
            points=points,
            cells=[("triangle", triangles)],
            point_data={"total_disp": displacement},
            cell_data={
                "von_Mises": [von_mises],
                "energy_density": [energy],
            },
        ),
    )
    print(f"Wrote ParaView data under {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
