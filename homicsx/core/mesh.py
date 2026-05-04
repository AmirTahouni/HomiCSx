from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from mpi4py import MPI


@dataclass(frozen=True)
class PhysicalTags:
    """
    Defines physical-tag convention for the mesh module.

    It centralizes the tag ID for both cell entities (matrix and inclusion phases) and the boundary facets.
    It ensures consistent physical tagging across GMSH and DOLFINx.

    Attributes
    ----------
    matrix: int
        The physical tag used for matrix region. 
        By default set to 1.
    phase_tag_offset: int
        The offset used to distinguish and separate the tags used for matrix/boundary entities and inclusion phases.
        phase_tag = phase_id + phase_tag_offset
    left: int
        The physical tag assigned to the left boundary (x = 0). 
        Fixed to 3.
    right: int
        The physical tag assigned to the right boundary (x = Lx). 
        Fixed to 4.
    bottom: int
        The physical tag assigned to the bottom boundary (y = 0). 
        Fixed to 5.
    top: int
        The physical tag assigned to the top boundary (y = Ly). 
        Fixed to 6.
    near: int
        The physical tag assigned to the near boundary (z = 0, only in 3D). 
        Fixed to 7.
    far: int
        The physical tag assigned to the far boundary (z = Lz, only in 3D). 
        Fixed to 8.

        
    Note
    ----
    - Boundary physical tags are fixed and dimensional-dependent.
    """
    matrix: int = 1
    phase_tag_offset: int = 10
    left: int = 3
    right: int = 4
    bottom: int = 5
    top: int = 6
    near: int = 7
    far: int = 8

    def cell_tag_for_phase(self, phase_id: int, matrix_phase_id: int = 0) -> int:
        """
        Calculates the unique integer tag for a given material phase.

        Parameters
        ----------
        phase_id : int
            The ID of the phase to tag.
        matrix_phase_id : int, default 0
            The ID that represents the matrix in the geometry model.

        Returns
        -------
        int
            The physical tag (either `matrix` or `offset + phase_id`).
        """
        phase_id = int(phase_id)
        matrix_phase_id = int(matrix_phase_id)

        if phase_id == matrix_phase_id:
            return self.matrix
        return self.phase_tag_offset + phase_id

    def cell_name_for_phase(self, phase_id: int, matrix_phase_id: int = 0, is_interphase: bool = False) -> str:
        """
        Generates a human-readable string name for a material phase.

        Parameters
        ----------
        phase_id : int
            The ID of the phase.
        matrix_phase_id : int, default 0
            The ID designated as the matrix.
        is_interphase : bool, default False
            Whether the phase represents an interphase layer.

        Returns
        -------
        str
            A string name like "Matrix", "Core_Phase_1", or "Interphase_Phase_101".
        """
        if phase_id == matrix_phase_id:
            return "Matrix"
        
        prefix = "Interphase" if is_interphase else "Core"
        return f"{prefix}_Phase_{phase_id}"

    def boundary_name_to_tag(self, dim: int) -> dict[str, int]:
        """
        Returns a mapping of boundary names to their corresponding integer tags.

        Parameters
        ----------
        dim : int
            Dimension of the problem (2 or 3).

        Returns
        -------
        dict[str, int]
            A dictionary where keys are boundary names (e.g., "left", "top") 
            and values are their assigned physical tags.

        Raises
        ------
        ValueError
            If `dim` is not 2 or 3.
        """
        if dim == 2:
            return {
                "left": self.left,
                "right": self.right,
                "bottom": self.bottom,
                "top": self.top,
            }
        if dim == 3:
            return {
                "left": self.left,
                "right": self.right,
                "bottom": self.bottom,
                "top": self.top,
                "near": self.near,
                "far": self.far,
            }
        raise ValueError(f"Unsupported dimension: {dim}")
    
@dataclass
class MeshSettings:
    """
    Container for GMSH mesh generator input parameters and settings.

    Attributes
    ----------
    min_size : float
        Minimum mesh element size, typically used near material interfaces.
    max_size : float
        Maximum mesh element size, typically used in bulk regions.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator used for mesh conversion and distribution.
        Defaults to ``MPI.COMM_WORLD``.
    model_rank : int, optional
        Rank responsible for constructing the Gmsh model and generating the
        mesh. Defaults to 0.
    model_name : str, optional
        Name assigned to the Gmsh model. Defaults to ``"rve"``.
    physical_tags : PhysicalTags or None, optional
        Tagging convention for physical groups. If ``None``, defaults are used.
    boundary_tol : float, optional
        Tolerance used for outer boundary classification. Defaults to ``1e-8``.
    refine_interfaces : bool, optional
        If ``True``, enable simple refinement near material interfaces.
        Defaults to ``True``.
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
        Number of mesh smoothing iterations requested from Gmsh.
        Defaults to 10.
    quad_hex: bool, optional
        Whether quad/hex mesh is to be forced during mesh generation.
        Defaults to False.
    view : bool, optional
        If ``True``, open the Gmsh GUI viewer after mesh generation on the
        model-owning rank. Defaults to ``False``.
    finalize : bool, optional
        If ``True``, finalize Gmsh before returning. Defaults to ``True``.
    """
    min_size: float
    max_size: float
    comm = MPI.COMM_WORLD
    model_rank: int = 0
    model_name: str = "rve"
    physical_tags: PhysicalTags | None = None
    boundary_tol: float = 1e-8
    refine_interfaces: bool = True
    interface_dist_min: float | None = None
    interface_dist_max: float | None = None
    verbosity: int = 0
    optimize: bool = True
    smoothing_steps: int = 10
    quad_hex: bool = False
    view: bool = False
    finalize: bool = True


@dataclass
class MeshImportMapping:
    """
    Mapping from GMSH physical groups to HomiCSx conventions.
    
    Attributes
    ----------
    cell_groups : dict
        Mapping from GMSH physical group ID → HomiCSx phase_id.
        Example: {100: 0, 200: 1} means GMSH group 100 is matrix (phase 0),
        GMSH group 200 is inclusion (phase 1).
    boundary_groups : dict, optional
        Mapping from GMSH physical group ID → boundary name.
        Example: {1: "left", 2: "right", 3: "bottom", 4: "top"}
        If None and auto_detect_boundaries=True, boundaries are detected
        by coordinate matching.
    auto_detect_boundaries : bool
        If True, ignores boundary_groups and detects boundaries by 
        checking facet midpoints against domain_size.
    """
    cell_groups: Dict[int, int]
    boundary_groups: Optional[Dict[int, str]] = None
    auto_detect_boundaries: bool = True


__all__ = [
    # mesh
    "PhysicalTags",
    "MeshSettings",
    "MeshImportMapping",
]

