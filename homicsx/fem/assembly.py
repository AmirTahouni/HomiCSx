from __future__ import annotations

from typing import Any

import ufl
from dolfinx import fem

from homicsx.core.fem import (
    ProblemSettings,
)


def build_displacement_space(mesh, settings: ProblemSettings):
    """
    Build the vector-valued displacement function space.

    Parameters
    ----------
    mesh:
        DOLFINx mesh.
    settings:
        FEM problem settings.

    Returns
    -------
    fem.FunctionSpace
        Vector-valued displacement space.

    Notes
    -----
    This follows the same basic pattern as your current FEM code:
    a standard vector Lagrange space for displacement.
    """
    return fem.functionspace(
        mesh,
        (settings.element_family, settings.element_degree, (settings.dim,)),
    )


__all__ = [
    # assembly
    "build_displacement_space",
]
