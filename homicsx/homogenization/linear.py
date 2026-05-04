from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import fem
import ufl

from homicsx.core.fem import ProblemSettings
from homicsx.core.homogenization import LinearHomogenizationResult
from homicsx.core.material import MaterialAssignment
from homicsx.fem.fluctuation import (
    build_linear_periodic_fluctuation_problem,
)


def _elementary_macro_strains(dim: int, mode: str = "complete") -> list[np.ndarray]:
    """
    Return elementary symmetric macroscopic strain tensors.

    Parameters
    ----------
    dim:
        Problem dimension, either 2 or 3.
    mode:
        Load-case mode. Supported values:
            - "complete"
            - "partial"

    Returns
    -------
    list[np.ndarray]
        List of symmetric strain tensors.
    """
    if dim == 2:
        if mode == "complete":
            return [
                np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float),      # Exx
                np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float),      # Eyy
                np.array([[0.0, 0.5], [0.5, 0.0]], dtype=float),      # Exy
            ]
        if mode == "partial":
            return [
                np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float),      # tensile
                np.array([[0.0, 0.5], [0.5, 0.0]], dtype=float),      # shear
            ]
        raise ValueError("For dim=2, mode must be 'complete' or 'partial'.")

    if dim == 3:
        if mode == "complete":
            return [
                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),  # Exx
                np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),  # Eyy
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),  # Ezz
                np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),  # Exy
                np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float),  # Exz
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]], dtype=float),  # Eyz
            ]
        if mode == "partial":
            return [
                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),  # tensile
                np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),  # shear
            ]
        raise ValueError("For dim=3, mode must be 'complete' or 'partial'.")

    raise ValueError("Only dim=2 or dim=3 are supported.")


def _voigt_size(dim: int) -> int:
    """
    Return the size of the symmetric Voigt vector for the given dimension.
    """
    if dim == 2:
        return 3
    if dim == 3:
        return 6
    raise ValueError("Only dim=2 or dim=3 are supported.")


def _stress_to_voigt_components(stress, dim: int) -> list:
    """
    Return stress components in Voigt-style order.

    Parameters
    ----------
    stress:
        UFL stress tensor.
    dim:
        Problem dimension.

    Returns
    -------
    list
        List of UFL scalar expressions.

    Notes
    -----
    The ordering is:
    - 2D: [s11, s22, s12]
    - 3D: [s11, s22, s33, s12, s13, s23]
    """
    if dim == 2:
        return [stress[0, 0], stress[1, 1], stress[0, 1]]

    if dim == 3:
        return [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[0, 1],
            stress[0, 2],
            stress[1, 2],
        ]

    raise ValueError("Only dim=2 or dim=3 are supported.")


def _average_stress_vector(mesh_obj, stress, dim: int) -> np.ndarray:
    """
    Compute the volume-averaged stress vector in Voigt notation.
    """
    volume = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh_obj)))
    components = _stress_to_voigt_components(stress, dim)

    averaged = np.zeros(len(components), dtype=float)
    for i, comp in enumerate(components):
        numerator = fem.assemble_scalar(fem.form(comp * ufl.dx(domain=mesh_obj)))
        averaged[i] = numerator / volume

    return averaged


def _solve_linear_homogenization(
    mesh_obj,
    cell_tags,
    facet_tags,
    assignment: MaterialAssignment,
    settings: ProblemSettings,
    physical_tags: Any,
    domain_size: tuple[float, ...],
    matrix_phase_id: int = 0,
    mode: str = "complete",
    petsc_options: dict[str, Any] | None = None,
) -> LinearHomogenizationResult:
    """
    Computes the effective (homogenized) stiffness tensor of an RVE using periodic BCs.

    The function applies elementary macro-strains to the RVE, solves for the 
    fluctuation displacement fields, and computes the volume-averaged stress 
    to populate the constitutive matrix (C_hom).

    Parameters
    ----------
    mesh_obj : dolfinx.mesh.Mesh
        The computational mesh of the RVE.
    cell_tags : dolfinx.mesh.MeshTags
        Tags for different material phases (Inclusions/Matrix).
    assignment : MaterialAssignment
        Constitutive properties for each phase.
    settings : ProblemSettings
        Simulation parameters (dimension, 2D formulation, etc.).
    mode : str, default "complete"
        - "complete": Solves for all components of the C matrix (3 in 2D, 6 in 3D).
        - "partial": Assumes macro-isotropy; solves only for unique components 
          to reduce computational cost.

    Returns
    -------
    HomogenizationResult
        Contains the homogenized stiffness matrix (C_hom), average stresses, 
        and the full fluctuation displacement fields for each load case.

    Notes
    -----
    - Fluctuation Field: The solution solved here is only the periodic 
      fluctuation 'v'. The total displacement is u = E:x + v.
    - Stability: Currently restricted to linear problems due to known 
      instabilities in nonlinear MPC implementations within dolfinx_mpc.
    """
    # if problem_is_nonlinear(assignment):
    #     raise ValueError('Nonlinear assignment for linear homogenization. All materials must be of type "LinearElasticIsotropic".')
    
    if petsc_options is None and settings.petsc_options:
        petsc_options = settings.petsc_options

    load_cases = _elementary_macro_strains(dim=settings.dim, mode=mode)
    C_hom = np.zeros((_voigt_size(settings.dim), len(load_cases)), dtype=float)

    stress_vectors: list[np.ndarray] = []
    fluctuation_fields: list[Any] = []

    for i, macro_strain in enumerate(load_cases):
        problem, context = build_linear_periodic_fluctuation_problem(
            mesh_obj=mesh_obj,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            assignment=assignment,
            settings=settings,
            physical_tags=physical_tags,
            domain_size=domain_size,
            macro_strain=macro_strain,
            matrix_phase_id=matrix_phase_id,
        )

        context.fluctuation_field.x.array[:] = 0

        solution = problem.solve()

        stress_avg = _average_stress_vector(
            mesh_obj=mesh_obj,
            stress=context.stress_expression,
            dim=settings.dim,
        )

        C_hom[:, i] = stress_avg
        stress_vectors.append(stress_avg)

        v_copy = fem.Function(solution.function_space, name=f"PeriodicFluctuation_case_{i}")
        v_copy.x.array[:] = solution.x.array[:]
        v_copy.x.scatter_forward()
        fluctuation_fields.append(v_copy)
    
    if settings.dim==3:
        if mode=="partial":
            tensile_diag = C_hom[0, 0]
            tensile_off_diag_1 = C_hom[1, 0]
            tensile_off_diag_2 = C_hom[2, 0]
            shear_diag = C_hom[3, 1]

            C_hom = np.ndarray((6, 6))

            C_hom[0, 0] = tensile_diag
            C_hom[1, 0] = tensile_off_diag_1
            C_hom[2, 0] = tensile_off_diag_2
            C_hom[3, 0] = 0
            C_hom[4, 0] = 0
            C_hom[5, 0] = 0

            C_hom[0, 1] = tensile_off_diag_1
            C_hom[1, 1] = tensile_diag
            C_hom[2, 1] = tensile_off_diag_1
            C_hom[3, 1] = 0
            C_hom[4, 1] = 0
            C_hom[5, 1] = 0

            C_hom[0, 2] = tensile_off_diag_2
            C_hom[1, 2] = tensile_off_diag_1
            C_hom[2, 2] = tensile_diag
            C_hom[3, 2] = 0
            C_hom[4, 2] = 0
            C_hom[5, 2] = 0

            C_hom[0, 3] = 0
            C_hom[1, 3] = 0
            C_hom[2, 3] = 0
            C_hom[3, 3] = shear_diag
            C_hom[4, 3] = 0
            C_hom[5, 3] = 0

            C_hom[0, 4] = 0
            C_hom[1, 4] = 0
            C_hom[2, 4] = 0
            C_hom[3, 4] = 0
            C_hom[4, 4] = shear_diag
            C_hom[5, 4] = 0

            C_hom[0, 5] = 0
            C_hom[1, 5] = 0
            C_hom[2, 5] = 0
            C_hom[3, 5] = 0
            C_hom[4, 5] = 0
            C_hom[5, 5] = shear_diag

    elif settings.dim==2:
        if mode=="partial":
            tensile_diag = C_hom[0, 0]
            tensile_off_diag = C_hom[1, 0]
            shear_diag = C_hom[2, 1]

            C_hom = np.ndarray((3, 3))

            C_hom[0, 0] = tensile_diag
            C_hom[1, 0] = tensile_off_diag
            C_hom[2, 0] = 0

            C_hom[0, 1] = tensile_off_diag
            C_hom[1, 1] = tensile_diag
            C_hom[2, 1] = 0

            C_hom[0, 2] = 0
            C_hom[1, 2] = 0
            C_hom[2, 2] = shear_diag

    return LinearHomogenizationResult(
        C_hom=C_hom,
        load_cases=load_cases,
        average_stresses=stress_vectors,
        fluctuation_fields=fluctuation_fields,
        metadata={
            "dim": settings.dim,
            "mode": mode,
            "two_dimensional_formulation": settings.two_dimensional_formulation,
            "num_load_cases": len(load_cases),
            "petsc_options": dict(petsc_options) if petsc_options is not None else None,
        },
    )


__all__ = [
    # # linear
    # "_stress_to_voigt_components",
    # "_average_stress_vector",
    # "_solve_linear_homogenization",
]
