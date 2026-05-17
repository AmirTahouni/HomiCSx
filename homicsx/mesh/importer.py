from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import numpy as np
import dolfinx
from mpi4py import MPI

from homicsx.core.mesh import PhysicalTags, MeshImportMapping


def _read_gmsh_mesh(mesh_file: str, dim: int):
    """Read GMSH mesh file into dolfinx format."""
    import gmsh
    from dolfinx.io import gmshio
    
    gmsh.initialize()
    gmsh.open(mesh_file)
    
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model,
        comm=MPI.COMM_WORLD,
        rank=0,
        gdim=dim,
    )
    
    gmsh.finalize()
    
    return mesh, cell_tags, facet_tags


def _read_gmsh_mesh(mesh_file: str, dim: int):
    """Read GMSH mesh file into dolfinx format."""
    import gmsh
    from dolfinx.io import gmshio
    from mpi4py import MPI
    
    gmsh.initialize()
    gmsh.open(mesh_file)
    
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model,
        comm=MPI.COMM_WORLD,
        rank=0,
        gdim=dim,
    )
    
    gmsh.finalize()
    
    return mesh, cell_tags, facet_tags


def _remap_cell_tags(mesh, raw_tags, cell_groups: Dict[int, int]):
    """
    Remap cell physical groups to HomiCSx phase IDs.
    
    Uses dolfinx.mesh.meshtags(mesh, dim, entities, values) signature.
    """
    import numpy as np
    import dolfinx.mesh
    
    values = raw_tags.values.copy()
    
    for old_id, new_id in cell_groups.items():
        values[raw_tags.values == old_id] = new_id
    
    # Identify any unmapped groups
    mapped_set = set(cell_groups.keys())
    all_groups = set(np.unique(raw_tags.values))
    unmapped = all_groups - mapped_set
    
    if unmapped:
        print(f"Warning: Unmapped cell groups detected: {unmapped}")
        print(f"  These cells will keep their original tags.")
    
    return dolfinx.mesh.meshtags(mesh, raw_tags.dim, raw_tags.indices, values)


def _remap_facet_tags(mesh, raw_tags, boundary_groups: Dict[int, str]):
    """
    Remap facet physical groups to HomiCSx boundary tags.
    """
    import numpy as np
    import dolfinx.mesh
    
    BOUNDARY_TAG_MAP = {
        "left": 3, "right": 4,
        "bottom": 5, "top": 6,
        "near": 7, "far": 8,
    }
    
    values = raw_tags.values.copy()
    
    for old_id, boundary_name in boundary_groups.items():
        if boundary_name in BOUNDARY_TAG_MAP:
            new_id = BOUNDARY_TAG_MAP[boundary_name]
            values[raw_tags.values == old_id] = new_id
    
    return dolfinx.mesh.meshtags(mesh, raw_tags.dim, raw_tags.indices, values)


def _detect_boundaries_by_coordinates(
    mesh, raw_ft, domain_size: Tuple[float, ...], dim: int
):
    """
    Detect boundary facets by their midpoint coordinates.
    """
    import numpy as np
    import dolfinx.mesh
    
    BOUNDARY_TAG_MAP = {
        "left": 3, "right": 4,
        "bottom": 5, "top": 6,
        "near": 7, "far": 8,
    }
    
    x = mesh.geometry.x
    
    mesh.topology.create_connectivity(dim - 1, dim)
    facet_to_cell = mesh.topology.connectivity(dim - 1, dim)
    
    values = np.zeros(len(raw_ft.indices), dtype=np.int32)
    tol = 1e-10
    
    for i, facet_index in enumerate(raw_ft.indices):
        cells = facet_to_cell.links(facet_index)
        if len(cells) != 1:
            continue
        
        mesh.topology.create_connectivity(dim - 1, 0)
        facet_vertices = mesh.topology.connectivity(dim - 1, 0).links(facet_index)
        midpoint = np.mean(x[facet_vertices], axis=0)
        
        if abs(midpoint[0]) < tol:
            values[i] = BOUNDARY_TAG_MAP["left"]
        elif abs(midpoint[0] - domain_size[0]) < tol:
            values[i] = BOUNDARY_TAG_MAP["right"]
        elif abs(midpoint[1]) < tol:
            values[i] = BOUNDARY_TAG_MAP["bottom"]
        elif abs(midpoint[1] - domain_size[1]) < tol:
            values[i] = BOUNDARY_TAG_MAP["top"]
        elif dim == 3:
            if abs(midpoint[2]) < tol:
                values[i] = BOUNDARY_TAG_MAP["near"]
            elif abs(midpoint[2] - domain_size[2]) < tol:
                values[i] = BOUNDARY_TAG_MAP["far"]
    
    detected = np.sum(values > 0)
    total = len(values)
    print(f"Boundary detection: {detected}/{total} facets identified as boundaries")
    
    return dolfinx.mesh.meshtags(mesh, raw_ft.dim, raw_ft.indices, values)


def import_mesh_auto(
    mesh_file: str,
    dim: int,
    domain_size: Tuple[float, ...],
    physical_tags=None,
):
    """
    Import a GMSH mesh that follows HomiCSx PhysicalTags convention.
    
    The mesh is used AS-IS since it already follows the convention.
    Only boundaries are verified/corrected.
    """
    from homicsx.core.mesh import PhysicalTags
    
    if physical_tags is None:
        physical_tags = PhysicalTags()
    
    mesh, cell_tags, facet_tags = _read_gmsh_mesh(mesh_file, dim)
    
    # Cell tags already follow HomiCSx convention - keep them as-is
    # Matrix = 1, Inclusion phases = 10 + phase_id
    # No remapping needed
    
    # For boundaries: auto-detect by coordinates to ensure correctness
    # (more reliable than trusting GMSH boundary tags)
    boundary_groups = physical_tags.boundary_name_to_tag(dim)
    tag_to_name = {v: k for k, v in boundary_groups.items()}
    
    # Try using GMSH boundary tags first, fall back to auto-detect
    detected_facet_tags = _detect_boundaries_by_coordinates(
        mesh, facet_tags, domain_size, dim
    )
    
    return mesh, cell_tags, detected_facet_tags


def import_mesh_with_mapping(
    mesh_file: str,
    dim: int,
    domain_size: Tuple[float, ...],
    mapping: MeshImportMapping,
):
    """
    Import a GMSH mesh with CUSTOM convention, remapping to HomiCSx.
    
    This is for meshes that DON'T follow HomiCSx PhysicalTags convention.
    The mapping tells us how to convert custom GMSH groups to HomiCSx tags.
    """
    mesh, raw_ct, raw_ft = _read_gmsh_mesh(mesh_file, dim)
    
    # Remap cell tags from custom GMSH groups → HomiCSx phase tags
    cell_tags = _remap_cell_tags(mesh, raw_ct, mapping.cell_groups)
    
    # Remap or auto-detect boundaries
    if mapping.auto_detect_boundaries:
        facet_tags = _detect_boundaries_by_coordinates(mesh, raw_ft, domain_size, dim)
    elif mapping.boundary_groups is not None:
        facet_tags = _remap_facet_tags(mesh, raw_ft, mapping.boundary_groups)
    else:
        facet_tags = raw_ft
    
    return mesh, cell_tags, facet_tags
