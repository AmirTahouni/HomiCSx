from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemSettings:
    """
    Basic settings for continuum-mechanics FEM problems in HomiCSx.

    Attributes
    ----------
    dim:
        Geometric dimension of the mesh. Supported values are 2 and 3.
    kinematics:
        Name of the kinematic setting. Typical values are:

            - "small_strain"
            - "finite_strain"

        This is mainly descriptive metadata. The actual global problem type
        is inferred from the assigned material models during assembly.
    two_dimensional_formulation:
        Optional label for 2D problems. For now, "plane_strain" is the main
        intended use. Use None for 3D.
    element_family:
        Finite-element family used for the displacement field.
    element_degree:
        Polynomial degree of the displacement field.
    quadrature_degree:
        Optional quadrature degree, stored here for future use.
    petsc_options:
        Optional dictionary of PETSc or solver options.
    metadata:
        Optional free-form dictionary for bookkeeping or later extensions.
    """
    dim: int
    kinematics: str = "small_strain"
    two_dimensional_formulation: str | None = None
    element_family: str = "Lagrange"
    element_degree: int = 1
    quadrature_degree: int | None = None
    petsc_options: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("dim must be either 2 or 3.")

        if self.dim == 3:
            if self.two_dimensional_formulation is not None:
                raise ValueError(
                    "two_dimensional_formulation must be None for 3D problems."
                )
            return

        if self.two_dimensional_formulation is None:
            raise ValueError(
                "two_dimensional_formulation must be 'plane_strain' for 2D "
                "problems; plane stress is not supported."
            )
        if self.two_dimensional_formulation == "plane_stress":
            raise NotImplementedError(
                "Plane-stress homogenization is not supported in HomiCSx "
                "the current release; use 'plane_strain' or a 3D model."
            )
        if self.two_dimensional_formulation != "plane_strain":
            raise ValueError(
                "two_dimensional_formulation must be 'plane_strain' for 2D "
                "problems."
            )


__all__ = [
    # fem
    "ProblemSettings",
]
