import dolfinx.fem as fem
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

from homicsx.core import (
    RVEGeometry,
    MeshSettings,
    PhysicalTags,
)
from homicsx.mesh import generate_mesh



def _create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k


def _create_pyvista_mesh(mesh):
    points = mesh.geometry.x.copy()
    tdim = mesh.topology.dim

    # Extract connectivity (cell-to-vertex mapping)
    # Note: In dolfinx, connectivity is stored in the mesh.topology object.
    connectivity = mesh.topology.connectivity(tdim, 0).array

    # Set number of nodes per cell and corresponding VTK cell type
    if tdim == 2:
        num_nodes_per_cell = 3
        cell_type = pv.CellType.TRIANGLE  # VTK_TRIANGLE (integer value 5)
    elif tdim == 3:
        num_nodes_per_cell = 4
        cell_type = pv.CellType.TETRA  # VTK_TETRA (integer value 10)
    else:
        raise ValueError("Unsupported mesh topology dimension")

    # Determine number of cells
    num_cells = connectivity.shape[0] // num_nodes_per_cell

    # Create a flat cell array in VTK format:
    # For each cell, the data layout is [n, pt0, pt1, ..., pt(n-1)]
    cells = np.hstack(
        [
            np.full((num_cells, 1), num_nodes_per_cell, dtype=np.int64),
            connectivity.reshape(num_cells, num_nodes_per_cell),
        ]
    ).flatten()

    # Create an array for cell types (one type per cell)
    cell_types = np.full(num_cells, cell_type, dtype=np.uint8)

    # Build the PyVista UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    return grid


def visualize_geometry(geometry: RVEGeometry) -> None:
    """
    Visualizes the RVE geometry with support for multiple phases (including interphases).
    """
    try:
        if geometry.dim==3:
            mesh, ct, ft = generate_mesh(
                geometry=geometry,
                mesh_settings=MeshSettings(
                    min_size=0.03,
                    max_size=0.04,
                )
            )
        elif geometry.dim==2:
            mesh, ct, ft = generate_mesh(
                geometry=geometry,
                mesh_settings=MeshSettings(
                    min_size=0.01,
                    max_size=0.02,
                )
            )
    except Exception as e:
        raise RuntimeError(f'Visualization failed during meshing: {e}')

    physical_tags = PhysicalTags()
    matrix_phase_id = 0


    unique_phases = {matrix_phase_id}
    for inc in geometry.inclusions:
        unique_phases.add(inc.phase_id)
        if inc.has_interphase:
            unique_phases.add(inc.interphase_phase_id)

    unique_phases = sorted(list(unique_phases))

    phase_to_value = {p_id: float(p_id) for p_id in unique_phases}

    tag_to_phase_value = {}
    for p_id in unique_phases:
        cell_tag = physical_tags.cell_tag_for_phase(p_id, matrix_phase_id)
        tag_to_phase_value[cell_tag] = phase_to_value[p_id]

    val_field = _create_piecewise_constant_field(
        mesh, ct, tag_to_phase_value, name="PhaseID"
    )

    grid = _create_pyvista_mesh(mesh)
    grid.cell_data["Phase"] = val_field.x.array

    plotter = pv.Plotter(window_size=[1200, 800], title="HomiCSx Geometry Visualization")

    cmap = plt.get_cmap("tab10")

    tol = 1e-5
    for i, p_id in enumerate(unique_phases):
        val = phase_to_value[p_id]
        phase_grid = grid.threshold([val - tol, val + tol], scalars="Phase")

        if phase_grid.n_cells == 0:
            continue

        if p_id == matrix_phase_id:
            color = "lightblue"
            opacity = 0.3
            label = "Matrix"
        elif any(inc.has_interphase and p_id == inc.interphase_phase_id for inc in geometry.inclusions):
            color = "orange"
            opacity = 0.6
            label = f"Interphase (Phase {p_id})"
        else:
            color = cmap(i % 10)
            opacity = 0.8
            label = f"Inclusion (Phase {p_id})"


        plotter.add_mesh(
            phase_grid, 
            color=color, 
            opacity=opacity, 
            show_edges=False, 
            line_width=0.5,
            label=label,
        )

    plotter.add_legend()
    plotter.view_isometric()
    plotter.add_axes()
    plotter.show(jupyter_backend="html")


__all__ = [
    "visualize_geometry",
]