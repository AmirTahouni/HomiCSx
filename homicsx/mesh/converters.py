from __future__ import annotations

from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh
import gmsh


def gmsh_model_to_dolfinx_mesh(
    *,
    dim: int,
    comm=MPI.COMM_WORLD,
    model_rank: int = 0,
):
    """
    Convert the current active Gmsh model into DOLFINx mesh objects.

    This function wraps ``dolfinx.io.gmshio.model_to_mesh`` for the current
    ``gmsh.model``. It is intended to be called after the Gmsh geometry has
    been constructed, physical groups have been assigned, and mesh generation
    has already been performed.

    Parameters
    ----------
    dim : int
        Geometric dimension of the mesh.
        Must be 2 for planar RVEs or 3 for volumetric RVEs.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator used for mesh distribution. Defaults to
        ``MPI.COMM_WORLD``.
    model_rank : int, optional
        Rank that owns the Gmsh model and provides it during conversion.
        Defaults to 0.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        The converted DOLFINx mesh.
    cell_tags : dolfinx.mesh.MeshTags
        Cell-wise physical tags, typically representing matrix and inclusions
        phases.
    facet_tags : dolfinx.mesh.MeshTags
        Facet-wise physical tags, typically representing outer boundaries.

    Notes
    -----
    This function assumes that the current Gmsh model is valid and already
    meshed. It does not create geometry or generate mesh elements itself.
    """
    return model_to_mesh(gmsh.model, comm, model_rank, gdim=dim)


__all__ = [
    # converters
    "gmsh_model_to_dolfinx_mesh",
]