from __future__ import annotations

from collections import defaultdict

import gmsh
import numpy as np
from mpi4py import MPI

from homicsx.core.geometry import Inclusion, RVEGeometry
from homicsx.core.mesh import PhysicalTags, MeshSettings
from .converters import gmsh_model_to_dolfinx_mesh
from .tagging import (
    add_boundary_physical_groups,
    add_cell_physical_groups,
    classify_outer_boundaries,
)


def _matrix_phase_id(geometry: RVEGeometry) -> int:
    """
    Return the phase identifier used for the matrix material.

    The current convention is that the first entry of ``geometry.phase_ids``
    represents the matrix phase. If no phase IDs are provided, this function
    falls back to 0.

    Parameters
    ----------
    geometry : RVEGeometry
        Geometry object containing the list of phase identifiers present in the
        RVE.

    Returns
    -------
    int
        Phase identifier to be treated as the matrix phase.

    Notes
    -----
    This is a simple project-level convention rather than a fully general
    material model. It keeps the current architecture lightweight while still
    supporting multi-phase unit cells.
    """
    if geometry.phase_ids:
        return int(geometry.phase_ids[0])
    return 0


def _non_matrix_inclusions_by_phase(
    geometry: RVEGeometry,
    matrix_phase_id: int,
) -> dict[int, list[Inclusion]]:
    """
    Group all non-matrix inclusions by material phase.

    This function filters out inclusions belonging to the matrix phase and
    returns the remaining inclusions grouped by ``phase_id``. The grouping is
    used later to clip and tag each material phase separately in the Gmsh model.

    Parameters
    ----------
    geometry : RVEGeometry
        Geometry object containing all inclusions, including periodic images.
    matrix_phase_id : int
        Phase identifier treated as the matrix material.

    Returns
    -------
    dict[int, list[Inclusion]]
        Mapping from non-matrix ``phase_id`` values to lists of corresponding
        ``Inclusion`` objects.

    Notes
    -----
    Periodic images are kept exactly as stored in ``geometry.inclusions``.
    This function does not attempt to reconstruct original/image relationships;
    it only groups by phase.
    """
    grouped: dict[int, list[Inclusion]] = defaultdict(list)
    for inclusion in geometry.inclusions:
        if int(inclusion.phase_id) == matrix_phase_id:
            continue
        grouped[int(inclusion.phase_id)].append(inclusion)
    return dict(grouped)


def _add_box_domain(dim: int, domain_size: np.ndarray) -> int:
    """
    Create the outer rectangular or cuboidal RVE domain in the Gmsh OCC model.

    Parameters
    ----------
    dim : int
        Geometric dimension of the domain.
        Must be 2 for a rectangle or 3 for a box.
    domain_size : ndarray
        Domain side lengths:
        - ``[Lx, Ly]`` in 2D
        - ``[Lx, Ly, Lz]`` in 3D

    Returns
    -------
    int
        Gmsh OCC entity tag of the created box domain.

    Notes
    -----
    The current implementation assumes that the domain starts at the origin.
    No translated or rotated outer domains are supported at this stage.
    """
    occ = gmsh.model.occ
    if dim == 2:
        return occ.add_rectangle(0.0, 0.0, 0.0, float(domain_size[0]), float(domain_size[1]))
    if dim == 3:
        return occ.add_box(
            0.0,
            0.0,
            0.0,
            float(domain_size[0]),
            float(domain_size[1]),
            float(domain_size[2]),
        )
    raise ValueError(f"Unsupported dimension: {dim}")


def _add_inclusion(dim: int, inclusion: Inclusion, use_inner: bool = False) -> int:
    """Adds a single primitive to OCC and returns its tag."""
    occ = gmsh.model.occ
    radii = inclusion.inner_radii if use_inner else inclusion.outer_radii

    x, y = map(float, inclusion.center[:2])
    if dim == 2:
        r1, r2 = (radii[0], radii[0]) if inclusion.shape == "circle" else radii
        return occ.addDisk(x, y, 0.0, r1, r2)
    else:
        x, y, z = map(float, inclusion.center)
        if inclusion.shape == "sphere":
            return occ.addSphere(x, y, z, radii[0])
        else:  # ellipsoid
            tag = occ.addSphere(x, y, z, 1.0)
            occ.dilate([(3, tag)], x, y, z, radii[0], radii[1], radii[2])
            return tag


def _collect_interface_entities(
    *,
    phase_dimtags: list[tuple[int, int]],
    dim: int,
) -> list[int]:
    """
    Collect lower-dimensional entities representing material interfaces.

    This function extracts the boundaries of the clipped phase regions and keeps
    only the entities of codimension one:
    - curves in 2D
    - surfaces in 3D

    These entities are later used to define mesh-refinement fields near
    material interfaces.

    Parameters
    ----------
    phase_dimtags : list[tuple[int, int]]
        Gmsh dimension-tag pairs representing clipped material phase regions.
    dim : int
        Geometric dimension of the problem.
        Must be 2 or 3.

    Returns
    -------
    list[int]
        Sorted list of unique Gmsh entity tags representing material interfaces.

    Notes
    -----
    If no phase regions are provided, an empty list is returned.
    The function does not distinguish between different non-matrix phases; it
    simply collects all interface entities together.
    """
    if not phase_dimtags:
        return []

    fdim = dim - 1
    boundary_dimtags = gmsh.model.getBoundary(
        phase_dimtags,
        oriented=False,
        recursive=False,
    )

    interface_tags: list[int] = []
    for entity_dim, entity_tag in boundary_dimtags:
        if entity_dim == fdim:
            interface_tags.append(entity_tag)

    return sorted(set(interface_tags))


def _set_mesh_options(
    *,
    min_size: float,
    max_size: float,
    verbosity: int = 0,
    optimize: bool = True,
    smoothing_steps: int = 10,
    dim: int = 3,
    force_quad_hex: bool = False,
) -> None:
    """
    Set global Gmsh mesh-generation options.

    Parameters
    ----------
    min_size : float
        Global minimum characteristic mesh size.
    max_size : float
        Global maximum characteristic mesh size.
    verbosity : int, optional
        Gmsh verbosity level. Defaults to 0.
    optimize : bool, optional
        If ``True``, enable Gmsh mesh optimization and Netgen optimization.
        Defaults to ``True``.
    smoothing_steps : int, optional
        Number of mesh smoothing iterations. Defaults to 10.

    Returns
    -------
    None

    Notes
    -----
    These options define the baseline mesh behavior for the model. Local mesh
    refinement near interfaces, if enabled later, is layered on top of these
    global settings.
    """
    gmsh.option.setNumber("General.Verbosity", int(verbosity))

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(min_size))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(max_size))

    gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
    gmsh.option.setNumber("Mesh.Smoothing", int(smoothing_steps))

    if force_quad_hex:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)

        if dim == 2:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1) 
        else:
            gmsh.option.setNumber("Mesh.Algorithm3D", 1) 


def _set_interface_refinement_field(
    *,
    dim: int,
    interface_entity_tags: list[int],
    min_size: float,
    max_size: float,
    dist_min: float | None = None,
    dist_max: float | None = None,
) -> None:
    """
    Define a simple distance-based mesh refinement field near interfaces.

    This function creates a Gmsh background mesh field that enforces a finer
    mesh near material interfaces and allows a coarser mesh away from them.

    Parameters
    ----------
    dim : int
        Geometric dimension of the problem.
        Must be 2 or 3.
    interface_entity_tags : list[int]
        Gmsh entity tags representing material interfaces:
        curves in 2D or surfaces in 3D.
    min_size : float
        Minimum mesh size to apply close to interfaces.
    max_size : float
        Maximum mesh size to apply far from interfaces.
    dist_min : float or None, optional
        Distance below which ``min_size`` is enforced.
        If ``None``, a default value based on ``max_size`` is used.
    dist_max : float or None, optional
        Distance beyond which ``max_size`` is enforced.
        If ``None``, a default value based on ``max_size`` is used.

    Returns
    -------
    None

    Notes
    -----
    The refinement strategy is simple:
    - one distance field,
    - one threshold field,
    - one background mesh field.

    If ``interface_entity_tags`` is empty, the function returns without
    creating any refinement field.
    """
    if not interface_entity_tags:
        return

    if dist_min is None:
        dist_min = 0.5 * max_size
    if dist_max is None:
        dist_max = 2.0 * max_size

    field = gmsh.model.mesh.field

    field.add("Distance", 1)
    if dim == 2:
        field.setNumbers(1, "CurvesList", interface_entity_tags)
    elif dim == 3:
        field.setNumbers(1, "FacesList", interface_entity_tags)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", float(min_size))
    field.setNumber(2, "SizeMax", float(max_size))
    field.setNumber(2, "DistMin", float(dist_min))
    field.setNumber(2, "DistMax", float(dist_max))

    field.setAsBackgroundMesh(2)


def build_gmsh_model(
    geometry: RVEGeometry,
    *,
    min_size: float,
    max_size: float,
    model_name: str = "rve",
    physical_tags: PhysicalTags | None = None,
    boundary_tol: float = 1e-8,
    refine_interfaces: bool = True,
    interface_dist_min: float | None = None,
    interface_dist_max: float | None = None,
    verbosity: int = 0,
    optimize: bool = True,
    smoothing_steps: int = 10,
    quad_hex: bool = False,
) -> dict:
    """
    Build a Gmsh OCC model from an ``RVEGeometry`` object.

    This function constructs the geometric model of the representative volume
    element, performs phase-wise clipping of inclusion primitives to the outer
    box, subtracts all non-matrix phases from the matrix, assigns physical
    groups to cells and boundaries, and optionally defines a mesh refinement
    field near material interfaces.

    Parameters
    ----------
    geometry : RVEGeometry
        Geometry object describing the RVE domain, inclusions, and material
        phase IDs.
    min_size : float
        Minimum mesh element size, typically used near interfaces.
    max_size : float
        Maximum mesh element size, typically used away from interfaces.
    model_name : str, optional
        Name assigned to the Gmsh model. Defaults to ``"rve"``.
    physical_tags : PhysicalTags or None, optional
        Tagging convention for matrix, phases, and boundaries.
        If ``None``, a default ``PhysicalTags`` instance is used.
    boundary_tol : float, optional
        Tolerance used to classify outer boundary entities by position.
        Defaults to ``1e-8``.
    refine_interfaces : bool, optional
        If ``True``, create a distance-based mesh refinement field around
        material interfaces. Defaults to ``True``.
    interface_dist_min : float or None, optional
        Distance below which the minimum mesh size is enforced in the interface
        refinement field.
    interface_dist_max : float or None, optional
        Distance beyond which the maximum mesh size is enforced in the
        interface refinement field.
    verbosity : int, optional
        Gmsh verbosity level. Defaults to 0.
    optimize : bool, optional
        If ``True``, enable Gmsh mesh optimization. Defaults to ``True``.
    smoothing_steps : int, optional
        Number of requested mesh smoothing iterations. Defaults to 10.

    Returns
    -------
    dict
        Dictionary containing metadata about the constructed model, including:
        - ``"dim"`` : geometric dimension
        - ``"domain_size"`` : RVE side lengths
        - ``"matrix_phase_id"`` : matrix phase identifier
        - ``"matrix_cell_tag"`` : physical tag of the matrix
        - ``"phase_cell_tags"`` : mapping from phase ID to physical cell tag
        - ``"matrix_entity_tags"`` : Gmsh entity tags of the matrix region
        - ``"phase_entity_tags"`` : mapping from phase ID to region tags
        - ``"boundary_entities"`` : outer boundary entity classification
        - ``"interface_entity_tags"`` : collected interface entity tags

    Notes
    -----
    This function builds the geometric and tagging model only. It does not call
    ``gmsh.model.mesh.generate(...)``.

    Important assumptions of the current implementation:
    - the outer domain is rectangular in 2D or cuboidal in 3D,
    - the outer domain starts at the origin,
    - inclusions are axis-aligned,
    - periodic images are already explicitly stored in
      ``geometry.inclusions``.

    Multi-phase behavior
    --------------------
    All non-matrix inclusions are grouped by ``phase_id`` and clipped phase by
    phase. Each phase receives its own physical group, which later becomes a
    distinct DOLFINx cell tag.
    """
    if physical_tags is None:
        physical_tags = PhysicalTags()

    dim = int(geometry.dim)
    domain_size = np.asarray(geometry.domain_size, dtype=float).reshape(-1)
    matrix_phase_id = _matrix_phase_id(geometry)

    gmsh.model.add(model_name)
    _set_mesh_options(
        min_size=min_size,
        max_size=max_size,
        verbosity=verbosity,
        optimize=optimize,
        smoothing_steps=smoothing_steps,
        dim=dim,
        force_quad_hex=quad_hex,
    )

    occ = gmsh.model.occ

    cell_tag = _add_box_domain(dim, domain_size)
    cell_dimtag = (dim, cell_tag)

    inclusions_by_phase = _non_matrix_inclusions_by_phase(geometry, matrix_phase_id)

    phase_primitive_tags: dict[int, list[int]] = {}

    for phase_id, inclusions in inclusions_by_phase.items():
        for inc in inclusions:
            if inc.has_interphase:
                outer_tag = _add_inclusion(dim, inc, use_inner=False)
                inner_tag = _add_inclusion(dim, inc, use_inner=True)

                cut_tags, _ = occ.cut([(dim, outer_tag)], [(dim, inner_tag)], 
                                    removeObject=True, removeTool=False)

                inter_phase_id = inc.interphase_phase_id
                phase_primitive_tags.setdefault(inter_phase_id, []).append(cut_tags[0][1])

                phase_primitive_tags.setdefault(inc.phase_id, []).append(inner_tag)
            else:
                tag = _add_inclusion(dim, inc, use_inner=False)
                phase_primitive_tags.setdefault(inc.phase_id, []).append(tag)

    occ.synchronize()

    phase_dimtags: dict[int, list[tuple[int, int]]] = {}
    all_clipped_phase_dimtags: list[tuple[int, int]] = []

    for phase_id, primitive_tags in phase_primitive_tags.items():
        if not primitive_tags:
            continue

        clipped_dimtags, _ = occ.intersect(
            [cell_dimtag],
            [(dim, tag) for tag in primitive_tags],
            removeObject=False,
            removeTool=True,
        )
        occ.synchronize()

        clipped_dimtags = [(d, t) for d, t in clipped_dimtags if d == dim]
        phase_dimtags[phase_id] = clipped_dimtags
        all_clipped_phase_dimtags.extend(clipped_dimtags)

    if all_clipped_phase_dimtags:
        matrix_dimtags, _ = occ.cut(
            [cell_dimtag],
            all_clipped_phase_dimtags,
            removeObject=True,
            removeTool=False,
        )
        occ.synchronize()
    else:
        matrix_dimtags = [cell_dimtag]

    matrix_entity_tags = [tag for entity_dim, tag in matrix_dimtags if entity_dim == dim]

    phase_entity_tags: dict[int, list[int]] = {}
    for phase_id, dimtags in phase_dimtags.items():

        phase_entity_tags[phase_id] = [tag for d, tag in dimtags if d == dim]

    add_cell_physical_groups(
        dim=dim,
        matrix_entity_tags=matrix_entity_tags,
        phase_entity_tags=phase_entity_tags,
        matrix_phase_id=matrix_phase_id,
        physical_tags=physical_tags,
    )

    boundary_entities = classify_outer_boundaries(
        dim=dim,
        domain_size=domain_size,
        tol=boundary_tol,
    )

    add_boundary_physical_groups(
        dim=dim,
        boundary_entities=boundary_entities,
        physical_tags=physical_tags,
    )

    occ.synchronize()

    interface_entity_tags = _collect_interface_entities(
        phase_dimtags=all_clipped_phase_dimtags,
        dim=dim,
    )

    if refine_interfaces:
        _set_interface_refinement_field(
            dim=dim,
            interface_entity_tags=interface_entity_tags,
            min_size=min_size,
            max_size=max_size,
            dist_min=interface_dist_min,
            dist_max=interface_dist_max,
        )

    phase_cell_tags = {
        phase_id: physical_tags.cell_tag_for_phase(phase_id, matrix_phase_id)
        for phase_id in sorted(phase_entity_tags)
    }

    return {
        "dim": dim,
        "domain_size": domain_size,
        "matrix_phase_id": matrix_phase_id,
        "matrix_cell_tag": physical_tags.cell_tag_for_phase(matrix_phase_id, matrix_phase_id),
        "phase_cell_tags": phase_cell_tags,
        "matrix_entity_tags": matrix_entity_tags,
        "phase_entity_tags": phase_entity_tags,
        "boundary_entities": boundary_entities,
        "interface_entity_tags": interface_entity_tags,
    }


def generate_mesh(
    geometry: RVEGeometry,
    mesh_settings: MeshSettings,
):
    """
    Build, mesh, and convert an RVE geometry into DOLFINx mesh objects.

    This is the main high-level meshing entry point. It initializes Gmsh,
    builds the OCC model from the provided ``RVEGeometry`` and ``MeshSettings`` objects, generates the mesh,
    and converts the result to DOLFINx compatible mesh, cell tags, and facet tags.

    Parameters
    ----------
    geometry : RVEGeometry
        Geometry object describing the RVE to be meshed.
    mesh_settings: MeshSettings
        A structured data object containing all necessary meshing parameters and settings.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        Generated DOLFINx mesh.
    cell_tags : dolfinx.mesh.MeshTags
        Cell-wise tags corresponding to matrix and material phase regions.
    facet_tags : dolfinx.mesh.MeshTags
        Facet-wise tags corresponding to outer domain boundaries.

    Note
    ----
    Internally, this function performs the following steps:
    1. initialize Gmsh,
    2. build the geometric model and physical groups,
    3. generate the mesh on ``model_rank``,
    4. convert the Gmsh model to DOLFINx,
    5. finalize Gmsh if requested.

    This function is intended to be the main user-facing API for the meshing
    module.

    See also
    --------
    homicsx.core.MeshSettings
    """
    gmsh.initialize()
    try:
        if mesh_settings.comm.rank == mesh_settings.model_rank:
            build_gmsh_model(
                geometry,
                min_size=mesh_settings.min_size,
                max_size=mesh_settings.max_size,
                model_name=mesh_settings.model_name,
                physical_tags=mesh_settings.physical_tags,
                boundary_tol=mesh_settings.boundary_tol,
                refine_interfaces=mesh_settings.refine_interfaces,
                interface_dist_min=mesh_settings.interface_dist_min,
                interface_dist_max=mesh_settings.interface_dist_max,
                verbosity=mesh_settings.verbosity,
                optimize=mesh_settings.optimize,
                smoothing_steps=mesh_settings.smoothing_steps,
                quad_hex=mesh_settings.quad_hex,
            )
            gmsh.model.mesh.generate(int(geometry.dim))

            if mesh_settings.view:
                gmsh.fltk.run()

        mesh, cell_tags, facet_tags = gmsh_model_to_dolfinx_mesh(
            dim=int(geometry.dim),
            comm=mesh_settings.comm,
            model_rank=mesh_settings.model_rank,
        )

    finally:
        if mesh_settings.finalize:
            gmsh.finalize()

    return mesh, cell_tags, facet_tags


__all__ = [
   # gmsh_builder
    "build_gmsh_model", 
    "generate_mesh", 
]

