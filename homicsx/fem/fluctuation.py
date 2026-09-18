from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Tuple

import numpy as np
import ufl
from dolfinx import fem
import petsc4py.PETSc as PETSc
import dolfinx
from dolfinx_mpc import LinearProblem

from homicsx.core.mesh import PhysicalTags
from homicsx.core.fem import ProblemSettings
from homicsx.core.material import (
    MaterialAssignment,
    MaterialState,
    QuadraturePointEvaluator,
    ViscoelasticGeneralizedMaxwell,
)
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
    state_coefficients: Optional[Dict[int, list[fem.Function]]] = None
    dt_constant: Optional[fem.Constant] = None
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

    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    dt_constant = fem.Constant(mesh_obj, PETSc.ScalarType(0.0))
    state_coefficients: Dict[int, list[fem.Function]] = {}

    # Build the first-Piola residual phase by phase. History-independent phases
    # use their hyperelastic energy. Generalized-Maxwell phases use the
    # algorithmically updated viscous metric in both residual and Jacobian,
    # while the coefficient itself remains the previous converged state.
    Residual = None
    for phase in assignment.materials_by_phase.keys():
        material_model = assignment.materials_by_phase[phase]
        tag = physical_tags.cell_tag_for_phase(phase)

        if isinstance(material_model, ViscoelasticGeneralizedMaxwell):
            equilibrium_energy = material_model.equilibrium_material.psi_form(F=F)
            phase_stress = ufl.diff(equilibrium_energy, F)
            C = F.T * F
            C_inverse = ufl.inv(C)
            coefficient_space = fem.functionspace(
                mesh_obj, ("DG", 0, (settings.dim, settings.dim))
            )
            branch_coefficients = []
            for branch, (shear_modulus, relaxation_time) in enumerate(
                zip(material_model.shear_moduli, material_model.relaxation_times)
            ):
                previous_cv = fem.Function(
                    coefficient_space, name=f"Cv_phase_{phase}_branch_{branch}"
                )
                previous_cv.x.array[:] = 0.0
                block_size = settings.dim * settings.dim
                local_cells = mesh_obj.topology.index_map(mesh_obj.topology.dim).size_local
                identity = np.eye(settings.dim, dtype=PETSc.ScalarType).reshape(-1)
                for cell_index in range(local_cells):
                    dof = coefficient_space.dofmap.cell_dofs(cell_index)[0]
                    previous_cv.x.array[
                        dof * block_size : (dof + 1) * block_size
                    ] = identity
                previous_cv.x.scatter_forward()
                branch_coefficients.append(previous_cv)
                alpha = ufl.exp(-dt_constant / relaxation_time)
                updated_cv = alpha * previous_cv + (1.0 - alpha) * C
                branch_pk2 = shear_modulus * (ufl.inv(updated_cv) - C_inverse)
                phase_stress += F * branch_pk2
            state_coefficients[phase] = branch_coefficients
        else:
            phase_energy = material_model.psi_form(F=F)
            phase_stress = ufl.diff(phase_energy, F)

        phase_residual = ufl.inner(phase_stress, ufl.grad(v)) * dx(tag)
        Residual = phase_residual if Residual is None else Residual + phase_residual

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
        state_coefficients=state_coefficients or None,
        dt_constant=dt_constant,
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


