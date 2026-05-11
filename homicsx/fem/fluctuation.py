from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Tuple, List, Callable

import numpy as np
import ufl
from dolfinx import fem
import petsc4py.PETSc as PETSc
import dolfinx
from dolfinx_mpc import LinearProblem

from homicsx.core.mesh import PhysicalTags
from homicsx.core.fem import ProblemSettings
from homicsx.core.material import MaterialAssignment, QuadraturePointEvaluator, MaterialState
from homicsx.materials.coefficients import _build_linear_elastic_coefficients

from .assembly import build_displacement_space
from .constraints import build_anchor_and_periodic_constraints, build_constraints_nonlinear
from .kinematics import small_strain
from .constitutive import _linear_homogenization_stress
from .nonlinear_problem import NonlinearProblemMPC


@dataclass
class LinearFluctuationProblemContext:
    """
    Extra context returned alongside a linear periodic fluctuation problem.

    Attributes
    ----------
    macro_strain:
        Constant macroscopic strain tensor used in the fluctuation problem.
    fluctuation_field:
        Unknown periodic fluctuation field to be solved for.
    stress_expression:
        UFL stress tensor expression based on the solved fluctuation field.
    coefficients:
        Linear-elastic DG0 coefficient bundle.
    metadata:
        Optional bookkeeping dictionary.
    """
    macro_strain: fem.Constant
    fluctuation_field: fem.Function
    stress_expression: Any
    coefficients: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NonlinearFluctuationProblemContext:
    """
    Extra context returned alongside a nonlinear periodic fluctuation problem.
    """
    F_macro: fem.Constant
    fluctuation_field: fem.Function
    quad_evaluator: QuadraturePointEvaluator
    material_states: Optional[Dict[int, Dict[int, MaterialState]]] = None
    time: float = 0.0
    dt: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_linear_periodic_fluctuation_problem(
    mesh_obj,
    cell_tags,
    facet_tags,
    assignment: MaterialAssignment,
    settings: ProblemSettings,
    physical_tags,
    domain_size: tuple[float, ...],
    macro_strain: np.ndarray,
    matrix_phase_id: int = 0,
    anchor_point: tuple[float, ...] | None = None,
    atol: float = 1e-12,
    petsc_options: Dict[str, Any] | None = None,
):
    """
    Build a linear periodic fluctuation problem for a prescribed macroscopic strain.

    Parameters
    ----------
    mesh_obj:
        DOLFINx mesh object.
    cell_tags:
        Cell MeshTags.
    facet_tags:
        Facet MeshTags.
    assignment:
        Phase-wise material assignment.
    settings:
        FEM problem settings.
    physical_tags:
        PhysicalTags convention object.
    domain_size:
        Domain side lengths.
    macro_strain:
        Prescribed macroscopic strain tensor.
    matrix_phase_id:
        Phase id corresponding to the matrix phase.
    anchor_point:
        Point used for the anchor Dirichlet condition.
    atol:
        Tolerance used when locating the anchor point.

    Returns
    -------
    tuple[LinearProblemDefinition, LinearFluctuationProblemContext]
        Linear problem definition plus useful homogenization context.

    Notes
    -----
    This function belongs to the FEM layer because it defines a specific
    variational boundary value problem. The homogenization layer should only
    orchestrate repeated calls to this builder over canonical load cases.
    """
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    linear_coeffs = _build_linear_elastic_coefficients(
        mesh=mesh_obj,
        cell_tags=cell_tags,
        assignment=assignment,
        physical_tags=physical_tags,
        matrix_phase_id=matrix_phase_id,
    )

    V = build_displacement_space(mesh_obj, settings)

    bcs, mpc = build_anchor_and_periodic_constraints(
        mesh=mesh_obj,
        facet_tags=facet_tags,
        V=V,
        domain_size=domain_size,
        physical_tags=physical_tags,
        anchor_point=anchor_point,
        atol=atol,
    )

    Eps = fem.Constant(mesh_obj, np.asarray(macro_strain, dtype=np.float64))

    du = ufl.TrialFunction(V)
    u_ = ufl.TestFunction(V)

    weak_form = ufl.inner(
        _linear_homogenization_stress(
            du,
            Eps,
            linear_coeffs.lambda_,
            linear_coeffs.mu,
        ),
        small_strain(u_),
    ) * ufl.dx(domain=mesh_obj)

    a_form, L_form = ufl.system(weak_form)

    solution_space = mpc.function_space if mpc is not None else V
    v = fem.Function(solution_space, name="PeriodicFluctuation")

    problem = LinearProblem(
        a=a_form,
        L=L_form,
        u=v,
        mpc=mpc,
        bcs=bcs,
        petsc_options=petsc_options,
    )

    stress_expr = _linear_homogenization_stress(
        v,
        Eps,
        linear_coeffs.lambda_,
        linear_coeffs.mu,
    )

    context = LinearFluctuationProblemContext(
        macro_strain=Eps,
        fluctuation_field=v,
        stress_expression=stress_expr,
        coefficients=linear_coeffs,
        metadata={
            "dim": settings.dim,
            "domain_size": tuple(domain_size),
        },
    )

    return problem, context


def build_nonlinear_periodic_fluctuation_problem_with_quadrature(
    mesh_obj: dolfinx.mesh.Mesh,
    cell_tags: dolfinx.mesh.MeshTags,
    facet_tags: dolfinx.mesh.MeshTags,
    assignment: MaterialAssignment,
    settings: ProblemSettings,
    physical_tags: PhysicalTags,
    domain_size: Tuple[float, ...],
    matrix_phase_id: int = 0,
    atol: float = 1e-12,
    petsc_options: Dict[str, Any] | None = None,
    # dt: float = 1.0,
    quad_degree: int = 4,
):
    """Build nonlinear problem with quadrature point evaluation support."""
    
    V = build_displacement_space(mesh_obj, settings)
    
    bcs, mpc = build_constraints_nonlinear(
        mesh=mesh_obj,
        facet_tags=facet_tags,
        V=V,
        domain_size=domain_size,
        physical_tags=physical_tags,
        atol=atol,
    )
    
    V_mpc = mpc.function_space
    u = fem.Function(V_mpc, name="Fluctuation")
    
    # Create quadrature evaluator
    quad_evaluator = QuadraturePointEvaluator(mesh_obj, degree=quad_degree)
    
    # Initialize macroscopic deformation gradient
    if settings.dim == 3:
        F_macro_data = np.eye(3, dtype=PETSc.ScalarType)
    elif settings.dim == 2:
        F_macro_data = np.eye(2, dtype=PETSc.ScalarType)
    
    F_macro = fem.Constant(mesh_obj, F_macro_data)
    
    # Initialize material states if needed
    material_states = None
    if assignment.has_history_dependence():
        material_states = assignment.initialize_states(
            mesh_obj, cell_tags, quad_evaluator, 
            physical_tags=physical_tags, 
            matrix_phase_id=matrix_phase_id
        )
        print(f"  Initialized history variables for {len(material_states)} phases")
        print(f"  Quadrature points per cell: {quad_evaluator.num_quad_points}")
    
    F = ufl.variable(F_macro + ufl.grad(u))
    dx = ufl.Measure("dx", domain=mesh_obj, subdomain_data=cell_tags, 
                     metadata={"quadrature_degree": quad_degree})
    
    # Build energy functional
    Pi = None
    for phase in assignment.materials_by_phase.keys():
        material_model = assignment.materials_by_phase[phase]
        psi = material_model.psi_form(F=F)
        tag = physical_tags.cell_tag_for_phase(phase)
        
        if Pi is None:
            Pi = psi * dx(tag)
        else:
            Pi += psi * dx(tag)
    
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    Residual = ufl.derivative(Pi, u, v)
    Jacobian = ufl.derivative(Residual, u, du)
    
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_monitor": None,
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 40,
        }
    
    problem = NonlinearProblemMPC(
        Residual,
        u,
        mpc,
        bcs=bcs,
        J=Jacobian,
        petsc_options=petsc_options
    )
    
    context = NonlinearFluctuationProblemContext(
        F_macro=F_macro,
        fluctuation_field=u,
        quad_evaluator=quad_evaluator,
        material_states=material_states,
        # dt=dt,
        metadata={
            "dim": settings.dim,
            "domain_size": tuple(domain_size),
        },
    )
    
    return problem, context

__all__ = [
    # fluctuation
    "LinearFluctuationProblemContext",
    "build_linear_periodic_fluctuation_problem",
    "build_nonlinear_periodic_fluctuation_peoblem",
]


