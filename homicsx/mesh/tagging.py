from __future__ import annotations

import gmsh
import numpy as np

from homicsx.core.mesh import PhysicalTags


def classify_outer_boundaries(
    *,
    dim: int,
    domain_size: np.ndarray,
    tol: float = 1e-8,
) -> dict[str, list[int]]:
    """
    Classify outer boundary entities of the RVE box using center-of-mass tests.

    In 2D: classify 1D boundary curves
    In 3D: classify 2D boundary surfaces

    Parameters
    ----------
    dim: int
        Spacial dimension of the domain (2 or 3).
    dim_size: np.ndarray
        States the size of the domain in its dimensions:
        - [Lx, Ly] for 2D domains.
        - [Lx, Ly, Lz] for 3D domains.
    tol: float
        The tolerance with respect to which the outer boundaries are being detected and calssified.
    
    Returns
    -------
    dict[str, list[int]]
        Mapping from boundary names to lists of Gmsh entity tags.

        In 2D, the keys are:
        - "left"
        - "right"
        - "bottom"
        - "top"

        In 3D, the keys are:
        - "left"
        - "right"
        - "bottom"
        - "top"
        - "near"
        - "far"

    Notes
    -----
    Boundary classification is based on the center of mass of each boundary
    entity. This approach is well suited to the assumptions:
    axis-aligned rectangular/cuboidal domains with boundaries lying on planes
    of constant x, y, and z.    
    """
    domain_size = np.asarray(domain_size, dtype=float).reshape(-1)
    fdim = dim - 1

    boundaries: dict[str, list[int]] = {
        "left": [],
        "right": [],
        "bottom": [],
        "top": [],
    }
    if dim == 3:
        boundaries["near"] = []
        boundaries["far"] = []

    entities = gmsh.model.getEntities(dim=fdim)

    Lx = float(domain_size[0])
    Ly = float(domain_size[1])
    Lz = float(domain_size[2]) if dim == 3 else None

    for _, tag in entities:
        com = gmsh.model.occ.getCenterOfMass(fdim, tag)

        x = float(com[0])
        y = float(com[1])

        if abs(x - 0.0) < tol:
            boundaries["left"].append(tag)
        elif abs(x - Lx) < tol:
            boundaries["right"].append(tag)

        if abs(y - 0.0) < tol:
            boundaries["bottom"].append(tag)
        elif abs(y - Ly) < tol:
            boundaries["top"].append(tag)

        if dim == 3:
            z = float(com[2])
            if abs(z - 0.0) < tol:
                boundaries["near"].append(tag)
            elif abs(z - Lz) < tol:
                boundaries["far"].append(tag)

    return boundaries


def add_cell_physical_groups(
    *,
    dim: int,
    matrix_entity_tags: list[int],
    phase_entity_tags: dict[int, list[int]],
    matrix_phase_id: int = 1,
    physical_tags: PhysicalTags | None = None,
) -> None:
    """
    Add physical groups for the matrix region and all non-matrix material phases.

    This function assigns Gmsh physical groups to cell regions so that they can
    later be converted into DOLFINx cell tags. One physical group is created
    for the matrix, and one physical group is created for each material phase
    present in the clipped inclusion regions.

    Parameters
    ----------
    dim : int
        Geometric dimension of the regions being tagged.
        Must be 2 for area regions or 3 for volume regions.
    matrix_entity_tags : list[int]
        Gmsh entity tags corresponding to the matrix region after all inclusion
        phases have been subtracted from the outer box.
    phase_entity_tags : dict[int, list[int]]
        Mapping from ``phase_id`` to lists of Gmsh entity tags belonging to that
        material phase.
    matrix_phase_id : int, optional
        Phase identifier representing the matrix material. Defaults to 0.
    physical_tags : PhysicalTags or None, optional
        Tagging convention used to generate physical group IDs and names.
        If ``None``, a default ``PhysicalTags`` instance is used.

    Returns
    -------
    None

    Notes
    -----
    In the default tagging convention:
    - matrix phase -> tag 1
    - phase k -> tag ``phase_tag_offset + k``

    These physical groups later become DOLFINx ``cell_tags`` and are typically
    used to assign material properties phase by phase.
    """
    if physical_tags is None:
        physical_tags = PhysicalTags()

    if matrix_entity_tags:
        gmsh.model.addPhysicalGroup(
            dim,
            matrix_entity_tags,
            physical_tags.cell_tag_for_phase(matrix_phase_id, matrix_phase_id),
            name=physical_tags.cell_name_for_phase(matrix_phase_id, matrix_phase_id),
        )

    for phase_id in sorted(phase_entity_tags):
        entity_tags = list(phase_entity_tags[phase_id])
        if not entity_tags:
            continue

        gmsh.model.addPhysicalGroup(
            dim,
            entity_tags,
            physical_tags.cell_tag_for_phase(phase_id, matrix_phase_id),
            name=physical_tags.cell_name_for_phase(phase_id, matrix_phase_id),
        )


def add_boundary_physical_groups(
    *,
    dim: int,
    boundary_entities: dict[str, list[int]],
    physical_tags: PhysicalTags | None = None,
) -> None:
    """
    Add physical groups for the outer boundaries of the RVE domain.

    This function takes the output of ``classify_outer_boundaries`` and assigns
    named physical groups to the corresponding Gmsh boundary entities. These
    physical groups later become DOLFINx facet tags.

    Parameters
    ----------
    dim : int
        Geometric dimension of the RVE.
        Must be 2 or 3.
    boundary_entities : dict[str, list[int]]
        Mapping from boundary names to lists of Gmsh entity tags.
    physical_tags : PhysicalTags or None, optional
        Tagging convention used to assign physical group IDs to boundaries.
        If ``None``, a default ``PhysicalTags`` instance is used.

    Returns
    -------
    None

    Notes
    -----
    The boundary names follow the axis-aligned convention:
    - 2D: left, right, bottom, top
    - 3D: left, right, bottom, top, near, far

    These physical groups are typically used later for periodic boundary
    conditions or other facet-based constraints in FEniCSx.
    """
    if physical_tags is None:
        physical_tags = PhysicalTags()

    fdim = dim - 1
    tag_map = physical_tags.boundary_name_to_tag(dim)

    pretty_names = {
        "left": "Left",
        "right": "Right",
        "bottom": "Bottom",
        "top": "Top",
        "near": "Near",
        "far": "Far",
    }

    for name, entity_tags in boundary_entities.items():
        if entity_tags:
            gmsh.model.addPhysicalGroup(
                fdim,
                entity_tags,
                tag_map[name],
                name=pretty_names[name],
            )


__all__ = [
    # tagging
    "classify_outer_boundaries",
    "add_cell_physical_groups",
    "add_boundary_physical_groups",
]            