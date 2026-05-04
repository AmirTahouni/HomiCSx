from __future__ import annotations

from typing import Any, Optional, Dict, Tuple, List, Callable

import time
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from dolfinx import fem, io, mesh
from dolfinx.io import XDMFFile
import csv
import petsc4py.PETSc as PETSc
from ufl import grad, ln, tr, det, variable, derivative, TestFunction, TrialFunction
from dolfinx_mpc import MultiPointConstraint
import dolfinx
import pyvista as pv

from homicsx.fem import NonlinearProblemMPC

from homicsx import(
    GeometryInput, 
    RVEGeometry,
    PhysicalTags, 
    MeshSettings,
    MaterialAssignment,
    NeoHookeanIsotropic,
    LinearElasticIsotropic,
)

from homicsx.core.homogenization import AdaptiveSettings
from homicsx.core.material import MaterialState, QuadraturePointEvaluator
from homicsx.fem.fluctuation import NonlinearFluctuationProblemContext

import numpy as np
import csv
from mpi4py import MPI
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from petsc4py import PETSc

from dataclasses import dataclass

# =============================================================================
# Kinematics and Utilities
# =============================================================================

def _make_Fbar(load_type: str, a: float, dim: int) -> np.ndarray:
    """
    Build macroscopic deformation gradient for common load paths.
    """
    if dim == 3:
        identity = np.eye(3, dtype=PETSc.ScalarType)
        if load_type == "uniaxial_x":
            F = identity.copy()
            F[0, 0] = 1.0 - a
        elif load_type == "uniaxial_y":
            F = identity.copy()
            F[1, 1] = 1.0 - a
        elif load_type == "uniaxial_z":
            F = identity.copy()
            F[2, 2] = 1.0 - a
        elif load_type == "biaxial":
            F = identity.copy()
            F[0, 0] = 1.0 - a
            F[1, 1] = 1.0 - a
        elif load_type == "triaxial":
            F = identity.copy()
            F[0, 0] = 1.0 - a
            F[1, 1] = 1.0 - a
            F[2, 2] = 1.0 - a
        elif load_type == "shear_xy":
            F = identity.copy()
            F[0, 1] = a
        elif load_type == "shear_xz":
            F = identity.copy()
            F[0, 2] = a
        elif load_type == "shear_yz":
            F = identity.copy()
            F[1, 2] = a
        elif load_type == "iso_stretch":
            F = identity.copy()
            F[0, 0] = 1.0 + a
            F[1, 1] = 1.0 + a
            F[2, 2] = 1.0 + a
        else:
            raise ValueError(f"Unknown load_type for 3D: {load_type}")
    else:  # dim == 2
        identity = np.eye(2, dtype=PETSc.ScalarType)
        if load_type == "uniaxial_x":
            F = identity.copy()
            F[0, 0] = 1.0 - a
        elif load_type == "uniaxial_y":
            F = identity.copy()
            F[1, 1] = 1.0 - a
        elif load_type == "biaxial":
            F = identity.copy()
            F[0, 0] = 1.0 - a
            F[1, 1] = 1.0 - a
        elif load_type == "shear_xy":
            F = identity.copy()
            F[0, 1] = a
        elif load_type == "iso_stretch":
            F = identity.copy()
            F[0, 0] = 1.0 + a
            F[1, 1] = 1.0 + a
        else:
            raise ValueError(f"Unknown load_type for 2D: {load_type}")
    
    return F


def _flatten_tensor(A: np.ndarray) -> np.ndarray:
    """Flatten a 2D tensor to 1D array (row-major)."""
    return A.flatten()


def _perturb_F(Fbar: np.ndarray, idx: int, delta: float) -> np.ndarray:
    """Perturb a component of the deformation gradient."""
    Fp = np.array(Fbar, dtype=float)
    dim = Fp.shape[0]
    i, j = divmod(idx, dim)
    Fp[i, j] += delta
    return Fp


# =============================================================================
# History Management
# =============================================================================

def _init_history() -> dict:
    """Initialize history dictionary for storing simulation results."""
    return {
        "step": [],
        "load_param": [],
        "load_type": [],
        "Fbar": [],
        "Pbar": [],
        "Wbar": [],
        "Jbar": [],
        "converged": [],
        "iters": [],
        "Ceff": []
    }


def _record_history(history, step_idx, a, load_type, Fbar, Pbar, Wbar, Jbar, converged, iters, Ceff):
    """Internal helper to keep the main loop clean."""
    history["step"].append(step_idx)
    history["load_param"].append(float(a))
    history["load_type"].append(load_type)
    history["Fbar"].append(np.array(Fbar, copy=True))
    history["Pbar"].append(np.array(Pbar, copy=True))
    history["Wbar"].append(float(Wbar))
    history["Jbar"].append(float(Jbar))
    history["converged"].append(int(converged))
    history["iters"].append(int(iters))
    history["Ceff"].append(None if Ceff is None else np.array(Ceff, copy=True))


def _save_history_csv(history: dict, filename: str, dim: int):
    """Save history to CSV file."""
    # Header
    header = ["step", "load_type", "load_param"]
    for i in range(dim):
        for j in range(dim):
            header.append(f"F{i+1}{j+1}")
    for i in range(dim):
        for j in range(dim):
            header.append(f"P{i+1}{j+1}")
    header.extend(["Wbar", "Jbar", "converged", "iters"])
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for k in range(len(history["step"])):
            F = history["Fbar"][k]
            P = history["Pbar"][k]
            
            row = [
                history["step"][k],
                history["load_type"][k],
                history["load_param"][k]
            ]
            
            for i in range(dim):
                for j in range(dim):
                    row.append(F[i, j])
            for i in range(dim):
                for j in range(dim):
                    row.append(P[i, j])
            
            row.extend([
                history["Wbar"][k],
                history["Jbar"][k],
                history["converged"][k],
                history["iters"][k]
            ])
            
            writer.writerow(row)


def _get_curve(history: dict, key: str, idx1: int = None, idx2: int = None) -> np.ndarray:
    """
    Extract curve data from history dictionary.
    
    Parameters
    ----------
    history : dict
        History dictionary from load case
    key : str
        Key to extract ("load_param", "Pbar", "Wbar", "Jbar")
    idx1, idx2 : int, optional
        Indices for tensor components
    
    Returns
    -------
    np.ndarray
        Array of values
    """
    # Map old key names to actual history keys
    key_mapping = {
        "load_param": "load_values",
        "Pbar": "P_avg",
        "Wbar": "W_avg",
        "Jbar": "J_avg",
    }
    
    actual_key = key_mapping.get(key, key)
    
    if actual_key not in history:
        # Try to find similar key
        for k in history.keys():
            if key.lower() in k.lower() or actual_key.lower() in k.lower():
                actual_key = k
                break
    
    if actual_key not in history:
        return np.array([])
    
    if actual_key == "load_values":
        return np.array(history[actual_key], dtype=float)
    elif actual_key in ["P_avg", "Pbar"]:
        if idx1 is not None and idx2 is not None:
            return np.array([P[idx1, idx2] for P in history[actual_key]], dtype=float)
        else:
            return np.array(history[actual_key], dtype=float)
    elif actual_key in ["W_avg", "Wbar", "J_avg", "Jbar"]:
        return np.array(history[actual_key], dtype=float)
    else:
        return np.array(history[actual_key], dtype=float)
    

def _get_tangent_component_curve(history: dict, row: int, col: int) -> np.ndarray:
    """Extract a specific tangent component curve from history."""
    vals = [np.nan if C is None else C[row, col] for C in history["Ceff"]]
    return np.array(vals, dtype=float)


# =============================================================================
# Energy and Stress Computation
# =============================================================================

def _compute_phase_volumes(
        domain: dolfinx.mesh.Mesh, 
        cell_tags: dolfinx.mesh.MeshTags, 
        material_assignment: MaterialAssignment,
        physical_tags: PhysicalTags,
):
    """
    Compute the volume of each phase.
    
    Returns:
        phase_volumes: dict mapping phase_id to volume
        total_volume: total domain volume
    """
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    
    phase_volumes = {}
    total_volume = 0.0
    
    for phase_id in material_assignment.materials_by_phase.keys():
        tag = physical_tags.cell_tag_for_phase(phase_id)
        vol_local = fem.assemble_scalar(fem.form(1.0 * dx(tag)))
        # vol = domain.comm.allreduce(vol_local, op=MPI.SUM)
        phase_volumes[phase_id] = vol_local
        total_volume += vol_local
    
    return phase_volumes, total_volume


def _compute_average_P_and_energy_with_state(
    domain,
    u: fem.Function,
    F_macro: fem.Constant,
    material_assignment: MaterialAssignment,
    cell_tags: dolfinx.mesh.MeshTags,
    dim: int,
    quad_evaluator: QuadraturePointEvaluator,
    material_states: Optional[Dict[int, Dict[int, MaterialState]]] = None,
    physical_tags: PhysicalTags = None,
    matrix_phase_id: int = 0,
):
    """
    Compute volume-averaged Piola stress, energy density, and Jacobian 
    with material state awareness.
    """
    if physical_tags is None:
        physical_tags = PhysicalTags()

    # Total deformation gradient: F = F_macro + grad(u)
    Ftot = ufl.variable(F_macro + ufl.grad(u))
    
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    
    # Compute phase volumes using existing function
    phase_volumes, total_volume = _compute_phase_volumes(
        domain, cell_tags, material_assignment, physical_tags=physical_tags
    )
    
    # Initialize averaged quantities
    Wbar = 0.0
    Pbar = np.zeros((dim, dim), dtype=float)
    Jbar = 0.0
    
    # Check if we have history-dependent materials
    has_history = material_assignment.has_history_dependence()
    
    # Create mapping from physical tag to phase_id
    tag_to_phase = {}
    for phase_id in material_assignment.materials_by_phase.keys():
        tag = physical_tags.cell_tag_for_phase(phase_id, matrix_phase_id)
        tag_to_phase[tag] = phase_id
    
    # If no history-dependent materials, use standard assembly (faster)
    if not has_history or material_states is None:
        for phase_id, material in material_assignment.materials_by_phase.items():
            tag = physical_tags.cell_tag_for_phase(phase_id, matrix_phase_id)
            phase_vol = phase_volumes[phase_id]
            weight = phase_vol / total_volume
            
            # Phase-specific energy density
            psi_phase = material.psi_form(Ftot)
            
            # Phase-specific Piola stress
            P_phase = ufl.diff(psi_phase, Ftot)
            
            # Average energy density for this phase
            W_local = fem.assemble_scalar(fem.form(psi_phase * dx(tag)))
            W_phase = domain.comm.allreduce(W_local, op=MPI.SUM)
            Wbar += weight * (W_phase / phase_vol)
            
            # Average Piola stress for this phase
            Pbar_phase = np.zeros((dim, dim), dtype=float)
            for i in range(dim):
                for j in range(dim):
                    val_local = fem.assemble_scalar(fem.form(P_phase[i, j] * dx(tag)))
                    val = domain.comm.allreduce(val_local, op=MPI.SUM)
                    Pbar_phase[i, j] = val / phase_vol
            
            Pbar += weight * Pbar_phase
            
            # Average Jacobian for this phase
            J_local = fem.assemble_scalar(fem.form(ufl.det(Ftot) * dx(tag)))
            J_phase = domain.comm.allreduce(J_local, op=MPI.SUM)
            Jbar += weight * (J_phase / phase_vol)
    
    # For history-dependent materials, use quadrature point evaluation
    else:
        # Compute deformation gradient at quadrature points
        F_by_cell = quad_evaluator.compute_deformation_gradient_at_quad_points(u, F_macro)
        
        # Initialize per-phase accumulators
        P_phase_sum = {phase_id: np.zeros((dim, dim)) for phase_id in material_assignment.materials_by_phase.keys()}
        W_phase_sum = {phase_id: 0.0 for phase_id in material_assignment.materials_by_phase.keys()}
        J_phase_sum = {phase_id: 0.0 for phase_id in material_assignment.materials_by_phase.keys()}
        phase_vol_check = {phase_id: 0.0 for phase_id in material_assignment.materials_by_phase.keys()}
        
        for cell_idx, F_cell in F_by_cell.items():
            physical_tag = cell_tags.values[cell_idx]
            # Map physical tag back to phase_id
            phase_id = tag_to_phase.get(physical_tag)
            if phase_id is None:
                continue
                
            material = material_assignment.materials_by_phase[phase_id]
            
            # Get material state if this phase has history tracking
            state = None
            if material_states is not None and phase_id in material_states:
                state = material_states[phase_id].get(cell_idx)
            
            # Get cell volume (approximate from phase volumes)
            num_cells_in_phase = len(np.where(cell_tags.values == physical_tag)[0])
            cell_vol = phase_volumes[phase_id] / num_cells_in_phase if num_cells_in_phase > 0 else 0.0
            
            # Integrate over quadrature points
            for q in range(quad_evaluator.num_quad_points):
                F_q = F_cell[q, :dim, :dim]
                weight = quad_evaluator.quad_weights[q] * cell_vol
                
                # Compute stress at quadrature point
                if material.requires_history() and state is not None:
                    P_q = material.get_quadrature_point_stress(state, F_q, q)
                else:
                    P_q = material.get_quadrature_point_stress(None, F_q, 0)
                
                # if material.requires_history() and state is not None:
                #     W_q = material.psi_form(state, F_q, q)
                # else:
                #     W_q = material.psi_form(F_q)
                W_q = material.evaluate_energy(F_q, dim)

                # Jacobian
                J_q = np.linalg.det(F_q)
                
                # Accumulate
                P_phase_sum[phase_id] += P_q * weight
                W_phase_sum[phase_id] += W_q * weight
                J_phase_sum[phase_id] += J_q * weight
                phase_vol_check[phase_id] += weight


                # if cell_idx == 1 and q == 0:
                #     print(f"\n[DEBUG] Phase {phase_id}, cell {cell_idx}, qp {q}")
                #     print(f"  F_q =\n{F_q}")
                #     print(f"  det(F_q) = {np.linalg.det(F_q)}")
                #     print(f"  cell_vol = {cell_vol}")
                #     print(f"  quad_weight = {quad_evaluator.quad_weights[q]}")
                #     print(f"  weight = {weight}")
                #     print(f"  P_q (equilibrium) =\n{material.equilibrium_material.get_quadrature_point_stress(state, F_q, q)}")
                #     print(f"  P_q (total) =\n{P_q}")
                # # inclusion_cells = np.where(cell_tags.values == physical_tags.cell_tag_for_phase(1, matrix_phase_id))[0]
                # # print(f"\n[DEBUG] Phase 1 (inclusion) cell indices: {inclusion_cells[:5]}")
                # if cell_idx == 1340 and q == 0:  # adjust to an actual inclusion cell index
                #     print(f"\n[DEBUG] Phase {phase_id}, cell {cell_idx}, qp {q}")
                #     print(f"  F_q =\n{F_q}")
                #     print(f"  P_q =\n{P_q}")
                #     print(f"  material: {material}")
        
        # Compute weighted averages
        for phase_id in material_assignment.materials_by_phase.keys():
            phase_vol = phase_volumes[phase_id]
            weight = phase_vol / total_volume

            # print(f"Phase {phase_id}: phase_volumes = {phase_volumes[phase_id]}, phase_vol_check = {phase_vol_check[phase_id]}")
            
            if phase_vol_check[phase_id] > 0:
                Pbar += weight * (P_phase_sum[phase_id] / phase_vol_check[phase_id])
                Wbar += weight * (W_phase_sum[phase_id] / phase_vol_check[phase_id])
                Jbar += weight * (J_phase_sum[phase_id] / phase_vol_check[phase_id])
    
    return Pbar, Wbar, Jbar


# =============================================================================
# Solver Utilities
# =============================================================================

def _snapshot_material_states(
    material_states: Dict[int, Dict[int, MaterialState]]
) -> Dict[int, Dict[int, MaterialState]]:
    """Create a deep copy of material states."""
    snapshot = {}
    for phase_id, phase_states in material_states.items():
        snapshot[phase_id] = {}
        for cell_idx, state in phase_states.items():
            snapshot[phase_id][cell_idx] = state.copy()
    return snapshot


def _restore_material_states(
    material_states: Dict[int, Dict[int, MaterialState]],
    snapshot: Dict[int, Dict[int, MaterialState]]
) -> None:
    """Restore material states from snapshot."""
    for phase_id, phase_states in snapshot.items():
        for cell_idx, state in phase_states.items():
            material_states[phase_id][cell_idx] = state.copy()


def _update_all_material_states(
    states: Dict[int, Dict[int, MaterialState]],
    F_by_cell: Dict[int, np.ndarray],
    dt: float,
    quad_weights: np.ndarray,
    material_assignment: MaterialAssignment
) -> Tuple[bool, Dict[str, int]]:
    """
    Update material states for all cells.
    
    Returns
    -------
    Tuple[bool, Dict[str, int]]
        (all_converged, iteration_stats)
    """
    all_converged = True
    stats = {"total_cells": 0, "total_iterations": 0, "failed_cells": 0}
    
    for phase_id, phase_states in states.items():
        material = material_assignment.materials_by_phase[phase_id]
        
        for cell_idx, state in phase_states.items():
            if cell_idx in F_by_cell:
                stats["total_cells"] += 1
                F_cell = F_by_cell[cell_idx]
                
                converged, iterations = material.update_state(
                    state, F_cell, dt, quad_weights, cell_idx
                )
                
                if converged:
                    stats["total_iterations"] += iterations
                else:
                    all_converged = False
                    stats["failed_cells"] += 1
                    
    return all_converged, stats


def _solve_once_with_history(
    problem: Any,
    u: dolfinx.fem.Function,
    F_macro: dolfinx.fem.Constant,
    Fbar: np.ndarray,
    context: NonlinearFluctuationProblemContext,
    material_assignment: MaterialAssignment,
    reset_to_zero: bool = True,
    initial_guess: Optional[np.ndarray] = None,  # Add this parameter
) -> Tuple[dolfinx.fem.Function, int, int]:
    """
    Solve with state update after convergence.
    """
    # Update macroscopic deformation gradient
    F_macro.value[...] = Fbar
    
    # Set initial guess
    if initial_guess is not None:
        u.x.array[:] = initial_guess
    elif reset_to_zero:
        u.x.array[:] = 0.0
    # else: keep current u as initial guess
    
    u.x.scatter_forward()
    
    # Solve nonlinear problem
    u_sol, converged, iters = problem.solve()
    u_sol.x.scatter_forward()

    # Get residual norm from SNES
    residual_norm = 0.0
    if hasattr(problem, '_snes'):
        residual_norm = problem._snes.getFunctionNorm()
    
    # If converged, update material states
    if converged > 0 and context.material_states is not None:
        # Compute deformation gradient at quadrature points for all cells
        F_by_cell = context.quad_evaluator.compute_deformation_gradient_at_quad_points(
            u_sol, F_macro
        )
        
        # Update material states for all phases
        all_converged, stats = _update_all_material_states(
            context.material_states,
            F_by_cell,
            context.dt,
            context.quad_evaluator.quad_weights,
            material_assignment
        )
        
        if not all_converged:
            print(f"    Warning: Local return mapping failed for {stats['failed_cells']} cells")
        elif stats['total_iterations'] > 0:
            pass  # Silent for perturbation solves
    
    return u_sol, converged, iters, residual_norm


def _compute_Ceff_fd_with_state(
    problem,
    u: fem.Function,
    F_macro: fem.Constant,
    domain: mesh.Mesh,
    material_assignment: MaterialAssignment,
    cell_tags: mesh.MeshTags,
    Fbar: np.ndarray,
    dim: int,
    quad_evaluator: QuadraturePointEvaluator,
    context: Any,
    delta: float = 1e-6
) -> np.ndarray:
    """
    Compute effective tangent stiffness via finite differences with state awareness.
    """
    dim2 = dim * dim
    
    # Save base state
    u_base = np.copy(u.x.array)
    
    # Deep copy the ENTIRE material_states dictionary
    saved_states = None
    if context.material_states is not None:
        import copy
        saved_states = copy.deepcopy(context.material_states)
    
    # Get base state quantities
    P0, W0, J0 = _compute_average_P_and_energy_with_state(
        domain, u, F_macro, material_assignment, cell_tags, dim,
        quad_evaluator, context.material_states
    )
    P0_vec = _flatten_tensor(P0)
    
    # Initialize tangent
    Ceff = np.zeros((dim2, dim2), dtype=float)
    
    # Use smaller delta for viscoelastic materials
    if material_assignment.has_history_dependence():
        delta = delta #min(delta, 1e-7)
    
    # Perturb each component
    for a in range(dim2):
        # Restore displacement
        u.x.array[:] = u_base[:]
        u.x.scatter_forward()
        
        # Restore material states from deep copy
        if saved_states is not None:
            import copy
            context.material_states = copy.deepcopy(saved_states)
        
        Fp = _perturb_F(Fbar, a, delta)
        
        # Solve with perturbation - use base state as initial guess
        u_sol, converged, iters, residual_norm = _solve_once_with_history(
            problem, u, F_macro, Fp, context, material_assignment, 
            reset_to_zero=False, initial_guess=u_base
        )
        
        if converged <= 0:
            print(f"    Warning: Perturbed solve failed at col {a}, using central difference fallback...")
            # Try backward perturbation
            Fm = _perturb_F(Fbar, a, -delta)
            
            # Restore states again
            u.x.array[:] = u_base[:]
            u.x.scatter_forward()
            if saved_states is not None:
                import copy
                context.material_states = copy.deepcopy(saved_states)
            
            u_sol_m, converged_m, _ = _solve_once_with_history(
                problem, u, F_macro, Fm, context, material_assignment,
                reset_to_zero=False, initial_guess=u_base
            )
            
            if converged_m <= 0:
                Ceff[:, a] = np.nan
                continue
            
            Pm, _, _ = _compute_average_P_and_energy_with_state(
                domain, u_sol_m, F_macro, material_assignment, cell_tags, dim,
                quad_evaluator, context.material_states
            )
            Pm_vec = _flatten_tensor(Pm)
            
            # Use base P0 as forward? No, skip this component
            Ceff[:, a] = np.nan
            continue
        
        Pp, _, _ = _compute_average_P_and_energy_with_state(
            domain, u_sol, F_macro, material_assignment, cell_tags, dim,
            quad_evaluator, context.material_states
        )
        Pp_vec = _flatten_tensor(Pp)
        Ceff[:, a] = (Pp_vec - P0_vec) / delta
    
    # Restore base state
    u.x.array[:] = u_base[:]
    u.x.scatter_forward()
    if saved_states is not None:
        import copy
        context.material_states = copy.deepcopy(saved_states)
    
    return Ceff


# =============================================================================
# Load Case Execution
# =============================================================================

def _run_one_load_case_with_history(
    domain: dolfinx.mesh.Mesh,
    physical_tags: PhysicalTags,
    problem: Any,
    u: dolfinx.fem.Function,
    F_macro: dolfinx.fem.Constant,
    material_assignment: MaterialAssignment,
    cell_tags: dolfinx.mesh.MeshTags,
    load_tag: str,
    load_name: str,
    load_values: np.ndarray,
    load_function: Callable,
    dim: int,
    context: NonlinearFluctuationProblemContext,
    xdmf: bool = False,
    tangent_every: int = 1,
    # tangent_delta: float = 1e-6,
    adaptive_settings: Optional[AdaptiveSettings] = None,
    # physical_dt: float = 1.0,
    strain_rate: Optional[float] = None,
    _hook_data: Optional[Dict] = None,
) -> Tuple[Dict, Optional[List[Tuple[float, Dict]]]]:
    """
    Run a single load case with full material history tracking.
    """
    history = _init_history()
    if adaptive_settings is None:
        adaptive_settings = AdaptiveSettings()
    
    # Initialize state history tracking
    state_history = []
    if context.material_states is not None:
        state_snapshot = _snapshot_material_states(context.material_states)
        state_history.append((0.0, state_snapshot))
    
    settings = adaptive_settings
    target_val = load_values[-1]
    current_a = 0.0
    da = max(target_val * settings.initial_step_ratio, settings.min_step)
    da_min, da_max = settings.min_step, target_val * settings.max_step_ratio
    
    step_idx = 1
    u_prev = dolfinx.fem.Function(u.function_space)
    u_prev.x.array[:] = u.x.array[:]
    
    # Store previous material states for cutback
    prev_states = None
    if context.material_states is not None:
        prev_states = _snapshot_material_states(context.material_states)
    
    # Store solutions for XDMF output at the end
    xdmf_solutions = [] if xdmf else None

    total_physical_time = 0.0
    # failure_attempt = 0
    
    print(f"\n" + "="*70)
    print(f"STARTING LOAD CASE: {load_name}")
    print(f"   Target Load: {target_val:.4f} | Initial da: {da:.2e}")
    if strain_rate is not None:
        print(f"   Constant strain rate: {strain_rate:.2e} (1/time)")
    if material_assignment.has_history_dependence():
        print(f"   History-dependent materials detected - tracking state evolution")
        print(f"   Quadrature points per cell: {context.quad_evaluator.num_quad_points}")
    print("="*70)
    
    while current_a < target_val - 1e-12:
        next_a = min(current_a + da, target_val)
        dt_load = next_a - current_a
        if strain_rate is not None:
            physical_dt = dt_load / strain_rate
        else:
            physical_dt = 0
        context.dt = physical_dt
        context.time = next_a
        
        # Create deformation gradient for this step
        if load_tag == 'built_in':
            Fbar = _make_Fbar(load_name, next_a, dim)
        elif load_tag == 'custom':
            Fbar = load_function(next_a)
        else:
            raise ValueError(f"Unknown load_tag: {load_tag}")
        
        progress = (next_a / target_val) * 100
        print(f"\nStep {step_idx:3d} | Progress: {progress:6.2f}% | Load: {next_a:.6f} | da: {dt_load:.2e}")
        print(f"   Strain increment da: {dt_load:.2e} | Physical dt: {physical_dt:.2e}")
        print(f"   Attempting da: {da:.2e}")

        # --- PRE-STEP HOOK ---
        force_reduction = False
        if _hook_data and _hook_data.get('pre_step'):
            from homicsx.homogenization.driver import PreStepData
            driver = _hook_data.get('driver')
            state = _hook_data.get('state')
            # F_current = Fbar.copy() #make_Fbar(load_name, current_a, dim) if current_a > 0 else np.eye(dim)
            pre_data = PreStepData(
                step_idx=step_idx,
                current_load=current_a,
                target_load=next_a,
                load_increment=dt_load,
                physical_dt=physical_dt,
                # F_macro_current=F_current,
                F_macro_target=Fbar.copy(),
                u_current=u,
                material_states=context.material_states,
                context=context,
                load_name=load_name,
                # custom_metadata=driver._custom_metadata.copy() if driver else {},
                state=state,
            )
            _execute_hooks(_hook_data['pre_step'], pre_data, "Pre-step")
        #     Fbar = pre_data.F_macro_target
            force_reduction = pre_data.force_adaptive_step_reduction
        
        if force_reduction:
            old_da = da
            da = max(da * settings.cutback_factor, da_min)
            print(f"   Hook requested step reduction: {old_da:.2e} -> {da:.2e}")
            continue
        
        try:
            # Restore from previous converged state
            u.x.array[:] = u_prev.x.array[:]
            u.x.scatter_forward()
            
            # Solve current step
            u_sol, converged, iters, residual_norm = _solve_once_with_history(
                problem, u, F_macro, Fbar, context, material_assignment
            )
            
            if converged <= 0:

                # --- STEP FAILURE HOOK ---
                if _hook_data and _hook_data.get('on_step_failure'):
                    from homicsx.homogenization.driver import StepFailureData
                    state = _hook_data.get('state')
                    driver = _hook_data.get('driver')
                    failure_data = StepFailureData(
                        step_idx=step_idx,
                        # attempt_number=failure_attempt + 1,
                        target_load=next_a,
                        attempted_increment=da,
                        error_message=f"Convergence code {converged}",
                        convergence_code=converged,
                        newton_iterations=iters,
                        u_last=u,
                        F_macro_target=Fbar.copy(),
                        material_states=context.material_states,
                        context=context,
                        load_name=load_name,
                        # will_retry=True,
                        # custom_metadata=driver._custom_metadata.copy() if driver else {},
                        state=state,
                    )
                    _execute_hooks(_hook_data['on_step_failure'], failure_data, "Step-failure")

                print(f"   SOLVER FAILED: Reason code {converged} after {iters} iterations.")
                raise RuntimeError(f"Base solve failed with convergence code {converged}")
            
            # --- POST-CONVERGENCE HOOK ---
            if _hook_data and _hook_data.get('post_convergence'):
                from homicsx.homogenization.driver import PostConvergenceData
                state = _hook_data.get('state')
                driver = _hook_data.get('driver')
                post_conv_data = PostConvergenceData(
                    step_idx=step_idx,
                    current_load=next_a,
                    F_macro=Fbar.copy(),
                    u=u_sol,
                    newton_iterations=iters,
                    residual_norm=residual_norm,
                    material_states=context.material_states,
                    context=context,
                    load_name=load_name,
                    # custom_metadata=driver._custom_metadata.copy() if driver else {},
                    state=state,
                )
                _execute_hooks(_hook_data['post_convergence'], post_conv_data, "Post-convergence")

            # Compute average stress and energy
            Pbar, Wbar, Jbar = _compute_average_P_and_energy_with_state(
                domain, u, F_macro, material_assignment, cell_tags, dim,
                context.quad_evaluator, context.material_states, 
                physical_tags, matrix_phase_id=0
            )
            
            # Check physical stability
            if np.isnan(Wbar) or np.any(np.isnan(Pbar)) or Jbar <= 0:
                print(f"   PHYSICAL INSTABILITY: Wbar={Wbar}, Jbar={Jbar:.4f}.")
                raise RuntimeError("Physical instability detected")
            
            # Success - update state
            print(f"   SUCCESS: Converged in {iters} iterations.")
            print(f"   Stats: Wbar={Wbar:.4e} | Jbar={Jbar:.4f} | P_max={np.max(np.abs(Pbar)):.4e}")
            
            # Store solution for XDMF output
            if xdmf_solutions is not None:
                u_copy = dolfinx.fem.Function(u.function_space)
                u_copy.x.array[:] = u.x.array[:]
                xdmf_solutions.append((next_a, u_copy, Fbar.copy()))
            
            current_a = next_a
            total_physical_time += physical_dt
            
            # Store converged state
            u_prev.x.array[:] = u.x.array[:]
            if context.material_states is not None:
                prev_states = _snapshot_material_states(context.material_states)
                state_snapshot = _snapshot_material_states(context.material_states)
                state_history.append((current_a, state_snapshot))
            
            # --- POST-STRESS HOOK (PRIMARY) ---
            if _hook_data and _hook_data.get('post_stress'):
                from homicsx.homogenization.driver import PostStressData
                state = _hook_data.get('state')
                driver = _hook_data.get('driver')
                post_stress_data = PostStressData(
                    step_idx=step_idx,
                    current_load=current_a,
                    F_macro=Fbar.copy(),
                    u=u_sol,
                    P_avg=Pbar.copy(),
                    W_avg=Wbar,
                    J_avg=Jbar,
                    material_states=context.material_states,
                    context=context,
                    load_name=load_name,
                    # custom_metadata=driver._custom_metadata.copy() if driver else {},
                    state=state,
                )
                _execute_hooks(_hook_data['post_stress'], post_stress_data, "Post-stress")
            
            # Compute tangent stiffness if requested
            Ceff = None
            if (step_idx % tangent_every) == 0:
                print(f"   Computing Tangent stiffness...")
                try:
                    Ceff = _compute_Ceff_fd_with_state(
                        problem=problem,
                        u=u,
                        F_macro=F_macro,
                        domain=domain,
                        material_assignment=material_assignment,
                        cell_tags=cell_tags,
                        Fbar=Fbar,
                        dim=dim,
                        quad_evaluator=context.quad_evaluator,
                        context=context,
                        delta=1e-6,
                    )
                    print("   Done.")
                except Exception as e:
                    print(f"   Failed! ({e})")
                    import traceback
                    traceback.print_exc()
                    Ceff = None
            
            # --- POST-TANGENT HOOK ---
            if Ceff is not None and _hook_data and _hook_data.get('post_tangent'):
                from homicsx.homogenization.driver import PostTangentData
                state = _hook_data.get('state')
                driver = _hook_data.get('driver')
                post_tan_data = PostTangentData(
                    step_idx=step_idx,
                    current_load=current_a,
                    F_macro=Fbar.copy(),
                    u=u_sol,
                    P_avg=Pbar.copy(),
                    W_avg=Wbar,
                    J_avg=Jbar,
                    C_tangent=Ceff.copy(),
                    material_states=context.material_states,
                    context=context,
                    load_name=load_name,
                    # tangent_computation_time=tangent_time,
                    # custom_metadata=driver._custom_metadata.copy() if driver else {},
                    state=state,
                )
                _execute_hooks(_hook_data['post_tangent'], post_tan_data, "Post-tangent")

            # Record history
            _record_history(
                history, step_idx, current_a, load_name,
                Fbar, Pbar, Wbar, Jbar, converged, iters, Ceff
            )
            
            step_idx += 1
            
            # Adaptive step size control
            old_da = da
            if iters <= settings.target_iters_min:
                da = min(da * settings.growth_factor, da_max)
                if da > old_da:
                    print(f"   Increasing step size: {old_da:.2e} -> {da:.2e}")
            elif iters >= settings.target_iters_max:
                da = max(da * settings.cutback_factor, da_min)
                if da < old_da:
                    print(f"   Decreasing step size: {old_da:.2e} -> {da:.2e}")
                    
        except RuntimeError as e:
            print(f"   RETRYING: Reducing step size... (Error: {e})")
            da *= settings.cutback_factor
            
            if da < da_min:
                print("\n" + "!"*70)
                print(f"FATAL ERROR: Could not converge at load {next_a:.6f}")
                print(f"   Step size {da:.2e} is below minimum {da_min:.2e}.")
                print("   Suggestion: Check mesh quality or material stability at this strain.")
                print("!"*70 + "\n")
                
                # Write XDMF with collected solutions before exiting
                if xdmf_solutions is not None and len(xdmf_solutions) > 0:
                    _write_xdmf_at_end(domain, xdmf_solutions, F_macro, 
                                       material_assignment, cell_tags, dim, load_name)
   
                return history, state_history if state_history else None
            
            # Restore previous converged state
            u.x.array[:] = u_prev.x.array[:]
            u.x.scatter_forward()
            if prev_states is not None:
                _restore_material_states(context.material_states, prev_states)
            continue
    
    # In run_one_load_case_with_history:
    if xdmf_solutions is not None and len(xdmf_solutions) > 0:
        print(f"\n   Writing XDMF output with {len(xdmf_solutions)} time steps...")
        _write_xdmf_at_end(domain, xdmf_solutions, F_macro, 
                        material_assignment, cell_tags, dim, load_name, physical_tags)
        print("   Done.")
    
    print(f"\nLOAD CASE '{load_name}' COMPLETED SUCCESSFULLY.")
    print(f"   Total steps: {step_idx-1} | Final load: {current_a:.6f}")
    print(f"   Total physical time: {total_physical_time:.4f}")
    print("="*70 + "\n")
    
    return history, state_history if state_history else None


def _run_all_load_cases(
    context: NonlinearFluctuationProblemContext,
    problem: Any,
    domain: dolfinx.mesh.Mesh,
    physical_tags: PhysicalTags,
    material_assignment: MaterialAssignment,
    cell_tags: dolfinx.mesh.MeshTags,
    dim: int,
    tangent_every: int = 1,
    # tangent_delta: float = 1e-6,
    output_prefix: str = "rve",
    max_strain: float = 0.2,
    custom_loads: Optional[Dict[str, Callable]] = None,
    built_in_loading_modes: Optional[List[str]] = None,
    adaptive_settings: Optional[AdaptiveSettings] = None,
    xdmf: bool = False,
    csv: bool = False,
    # physical_dt: float = 1.0,
    strain_rate: Optional[float] = None,
    _hook_data: Optional[Dict] = None,
) -> Dict[str, Tuple[Dict, Optional[List]]]:
    """
    Run all standard load cases for homogenization with material history support.
    
    Parameters
    ----------
    context : NonlinearFluctuationProblemContext
        Problem context containing quad_evaluator and material_states
    problem : Any
        Nonlinear problem solver
    domain : dolfinx.mesh.Mesh
        Computational mesh
    material_assignment : MaterialAssignment
        Material assignment for all phases
    cell_tags : dolfinx.mesh.MeshTags
        Cell tags for phase identification
    dim : int
        Problem dimension (2 or 3)
    tangent_every : int
        Compute tangent stiffness every N steps
    tangent_delta : float
        Perturbation size for finite difference tangent
    output_prefix : str
        Prefix for output files
    max_strain : float
        Maximum strain for load cases
    custom_loads : Optional[Dict[str, Callable]]
        Dictionary of custom load functions
    built_in_loading_modes : Optional[List[str]]
        List of built-in load modes to run (if None, runs all)
    adaptive_settings : Optional[AdaptiveSettings]
        Settings for adaptive stepping
    xdmf : bool
        Whether to write XDMF output files
    csv : bool
        Whether to write CSV output files
        
    Returns
    -------
    Dict[str, Tuple[Dict, Optional[List]]]
        Dictionary mapping load case name to (history_dict, state_history_list)
    """
    if adaptive_settings is None:
        adaptive_settings = AdaptiveSettings()
    
    # Define load cases based on dimension
    if built_in_loading_modes is None:
        if dim == 3:
            load_cases = {
                "uniaxial_x": {'load_value': np.linspace(0, max_strain, 20), 
                              'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "uniaxial_y": {'load_value': np.linspace(0, max_strain, 20), 
                              'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "uniaxial_z": {'load_value': np.linspace(0, max_strain, 20), 
                              'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "biaxial": {'load_value': np.linspace(0, max_strain, 20), 
                           'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "triaxial": {'load_value': np.linspace(0, max_strain, 20), 
                            'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "shear_xy": {'load_value': np.linspace(0, max_strain, 20), 
                            'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "shear_xz": {'load_value': np.linspace(0, max_strain, 20), 
                            'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "shear_yz": {'load_value': np.linspace(0, max_strain, 20), 
                            'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "iso_stretch": {'load_value': np.linspace(0, max_strain, 20), 
                               'load_function': _make_Fbar, 'load_tag': 'built_in'},
            }
        else:  # dim == 2
            load_cases = {
                "uniaxial_x": {'load_value': np.linspace(0, max_strain, 20), 
                              'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "uniaxial_y": {'load_value': np.linspace(0, max_strain, 20), 
                              'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "biaxial": {'load_value': np.linspace(0, max_strain, 20), 
                           'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "shear_xy": {'load_value': np.linspace(0, max_strain, 20), 
                            'load_function': _make_Fbar, 'load_tag': 'built_in'},
                "iso_stretch": {'load_value': np.linspace(0, max_strain, 20), 
                               'load_function': _make_Fbar, 'load_tag': 'built_in'},
            }
    else:
        load_cases = {}
        for mode in built_in_loading_modes:
            if dim == 3:
                valid_modes_3d = ["uniaxial_x", "uniaxial_y", "uniaxial_z", "biaxial", "triaxial", 
                                  "shear_xy", "shear_xz", "shear_yz", "iso_stretch"]
                if mode not in valid_modes_3d:
                    raise ValueError(
                        f'3D modes must be one of {valid_modes_3d}. Received: {mode}.'
                    )
            elif dim == 2:
                valid_modes_2d = ["uniaxial_x", "uniaxial_y", "biaxial", "shear_xy", "iso_stretch"]
                if mode not in valid_modes_2d:
                    raise ValueError(
                        f'2D modes must be one of {valid_modes_2d}. Received: {mode}.'
                    )
            load_cases[mode] = {'load_value': np.linspace(0, max_strain, 20), 
                               'load_function': _make_Fbar, 'load_tag': 'built_in'}
    
    # Add custom load cases
    if custom_loads is not None:
        for custom_mode, custom_func in custom_loads.items():
            load_cases[custom_mode] = {'load_value': np.linspace(0, max_strain, 20), 
                                       'load_function': custom_func, 
                                       'load_tag': 'custom'}
    
    all_histories = {}
    all_state_histories = {}
    
    print("\n" + "="*70)
    print("STARTING HOMOGENIZATION ANALYSIS")
    print(f"   Dimension: {dim}D")
    print(f"   Number of load cases: {len(load_cases)}")
    print(f"   Max strain: {max_strain}")
    print(f"   History-dependent materials: {material_assignment.has_history_dependence()}")
    print("="*70 + "\n")
    
    for load_idx, (load_name, load_config) in enumerate(load_cases.items(), 1):
        print(f"\n{'#'*70}")
        print(f"Load Case {load_idx}/{len(load_cases)}: {load_name}")
        print(f"{'#'*70}")
        
        load_tag = load_config['load_tag']
        load_vals = load_config['load_value']
        load_func = load_config['load_function']

        if _hook_data and 'state' in _hook_data:
            _hook_data['state'].clear_load_case()

        # --- PRE-LOAD-CASE HOOK ---
        if _hook_data and _hook_data.get('pre_load_case'):
            from homicsx.homogenization.driver import PreLoadCaseData
            driver = _hook_data.get('driver')
            state = _hook_data.get('state')
            pre_load_data = PreLoadCaseData(
                load_name=load_name,
                load_tag=load_tag,
                target_load=load_vals[-1],
                dim=dim,
                context=context,
                material_assignment=material_assignment,
                # custom_metadata=driver._custom_metadata.copy() if driver else {},
                state=state,
            )
            _execute_hooks(_hook_data['pre_load_case'], pre_load_data, "Pre-load-case")
        
        # Reset fluctuation field to zero at start of each load case
        u = context.fluctuation_field
        u.x.array[:] = 0.0
        u.x.scatter_forward()
        
        # Reset material states if history-dependent
        if context.material_states is not None:
            print("   Re-initializing material states for new load case...")
            context.material_states = material_assignment.initialize_states(
                domain, cell_tags, context.quad_evaluator
            )
            context.time = 0.0
            context.dt = load_vals[1] - load_vals[0] if len(load_vals) > 1 else 1.0
        
        # Run the load case
        history, state_history = _run_one_load_case_with_history(
            domain=domain,
            physical_tags=physical_tags,
            problem=problem,
            u=u,
            F_macro=context.F_macro,
            material_assignment=material_assignment,
            cell_tags=cell_tags,
            load_tag=load_tag,
            load_name=load_name,
            load_values=load_vals,
            load_function=load_func,
            dim=dim,
            context=context,
            xdmf=xdmf,
            tangent_every=tangent_every,
            adaptive_settings=adaptive_settings,
            strain_rate=strain_rate,
            _hook_data=_hook_data,
        )

        # Save CSV if requested
        if csv:
            csv_name = f"{output_prefix}_{load_name}.csv"
            _save_history_csv(history, csv_name, dim)
            print(f"   Saved CSV: {csv_name}")
            
            # Also save state history if available
            if state_history is not None:
                state_csv_name = f"{output_prefix}_{load_name}_states.csv"
                _save_state_history_csv(state_history, state_csv_name, dim)
                print(f"   Saved state CSV: {state_csv_name}")
        
        # --- POST-LOAD-CASE HOOK ---
        if _hook_data and _hook_data.get('post_load_case'):
            from homicsx.homogenization.driver import PostLoadCaseData
            state = _hook_data.get('state')
            driver = _hook_data.get('driver')
            post_load_data = PostLoadCaseData(
                load_name=load_name,
                history=history,
                state_history=state_history,
                total_steps=len(_get_curve_flexible(history, "P_avg", 0, 0)),
                total_physical_time=context.time if hasattr(context, 'time') else 0.0,
                converged=True,
                context=context,
                # custom_metadata=driver._custom_metadata.copy() if driver else {},
                state=state,
            )
            _execute_hooks(_hook_data['post_load_case'], post_load_data, "Post-load-case")

        all_histories[load_name] = history
        if state_history is not None:
            all_state_histories[load_name] = state_history
    
    print("\n" + "="*70)
    print("HOMOGENIZATION ANALYSIS COMPLETE")
    print(f"   Total load cases completed: {len(all_histories)}")
    print("="*70 + "\n")
    
    return all_histories, all_state_histories if all_state_histories else None


def _execute_hooks(hook_list: List[Callable], data: Any, hook_name: str) -> None:
    """
    Execute a list of hooks safely, catching and reporting errors.
    """
    if not hook_list:
        return
    
    for hook in hook_list:
        try:
            hook(data)
        except Exception as e:
            print(f"    Warning: {hook_name} hook failed: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Post-Processing
# =============================================================================

def _write_xdmf_at_end(
    domain: dolfinx.mesh.Mesh,
    solutions: List[Tuple[float, dolfinx.fem.Function, np.ndarray]],
    F_macro: dolfinx.fem.Constant,
    material_assignment: MaterialAssignment,
    cell_tags: dolfinx.mesh.MeshTags,
    dim: int,
    load_name: str,
    physical_tags: Any = None,
):
    """
    Write all fields to XDMF file at the end of the load case.
    Properly handles multi-phase materials by interpolating per-phase.
    """
    import ufl
    from dolfinx.fem import Function, functionspace, Expression, Constant
    from dolfinx.io import XDMFFile
    
    # Create XDMF file
    xdmf_file = XDMFFile(domain.comm, f"{load_name}_results.xdmf", "w")
    xdmf_file.write_mesh(domain)
    
    # Use DG0 for stress/energy fields (element-wise constant)
    V_tensor = functionspace(domain, ("DG", 0, (dim, dim)))
    V_scalar = functionspace(domain, ("DG", 0))
    V_vector = functionspace(domain, ("CG", 1, (dim,)))
    
    print(f"   Writing {len(solutions)} time steps to XDMF...")
    
    for idx, (time_val, u_sol, Fbar) in enumerate(solutions):
        if idx % max(1, len(solutions)//10) == 0:
            print(f"      Step {idx+1}/{len(solutions)}")
        
        # Create fresh functions for this time step
        pk1_out = Function(V_tensor, name="PK1_Stress")
        vm_out = Function(V_scalar, name="Von_Mises")
        sed_out = Function(V_scalar, name="Strain_Energy_Density")
        J_out = Function(V_scalar, name="Jacobian")
        u_total = Function(V_vector, name="Total_Displacement")
        
        # Save original F_macro and set to current value
        F_macro_original = F_macro.value.copy()
        F_macro.value[...] = Fbar
        
        # 1. Fluctuation displacement
        u_sol.name = "fluctuation"
        xdmf_file.write_function(u_sol, time_val)
        
        # 2. Total displacement
        I_mat = np.eye(dim)
        H_avg = Fbar - I_mat
        x_vec = ufl.SpatialCoordinate(domain)
        H_const = Constant(domain, H_avg)
        
        u_total_expr = Expression(
            ufl.dot(H_const, x_vec) + u_sol, 
            V_vector.element.interpolation_points()
        )
        u_total.interpolate(u_total_expr)
        u_total.name = "total_disp"
        xdmf_file.write_function(u_total, time_val)
        
        # 3. Per-phase fields: PK1 stress, von Mises, energy density, Jacobian
        Ftotal = ufl.variable(F_macro + ufl.grad(u_sol))
        I = ufl.Identity(dim)
        
        for phase_id, material in material_assignment.materials_by_phase.items():
            tag = physical_tags.cell_tag_for_phase(phase_id, matrix_phase_id=0)
            cells = cell_tags.find(tag)
            
            if len(cells) > 0:
                # Material-specific energy and stress
                psi_phase = material.psi_form(Ftotal)
                P_phase = ufl.diff(psi_phase, Ftotal)
                
                # Jacobian
                J = ufl.det(Ftotal)
                
                # von Mises stress (Cauchy stress based)
                sigma = (1/J) * P_phase * Ftotal.T
                s = sigma - (1/dim) * ufl.tr(sigma) * I
                vm_phase = ufl.sqrt(1.5 * ufl.inner(s, s))
                
                # Interpolate only on cells of this phase
                pk1_out.interpolate(
                    Expression(P_phase, V_tensor.element.interpolation_points()), 
                    cells
                )
                vm_out.interpolate(
                    Expression(vm_phase, V_scalar.element.interpolation_points()), 
                    cells
                )
                sed_out.interpolate(
                    Expression(psi_phase, V_scalar.element.interpolation_points()), 
                    cells
                )
                J_out.interpolate(
                    Expression(J, V_scalar.element.interpolation_points()), 
                    cells
                )
        
        # Write per-phase fields
        pk1_out.name = "PK1_stress"
        vm_out.name = "von_Mises"
        sed_out.name = "energy_density"
        J_out.name = "jacobian"
        
        xdmf_file.write_function(pk1_out, time_val)
        xdmf_file.write_function(vm_out, time_val)
        xdmf_file.write_function(sed_out, time_val)
        xdmf_file.write_function(J_out, time_val)
        
        # Restore F_macro for next iteration
        F_macro.value[...] = F_macro_original
    
    xdmf_file.close()
    print(f"   XDMF write complete.")


def _summarize_histories(all_histories: dict, dim: int) -> dict:
    """
    Summarize histories into structured data for plotting.
    
    Handles both old format (dict only) and new format (tuple with state history).
    """
    summary = {}
    
    for load_type, history_data in all_histories.items():
        # Handle both old (dict only) and new (tuple) formats
        if isinstance(history_data, tuple):
            history, state_history = history_data
        else:
            history = history_data
            state_history = None
        
        # Extract load curve - check for possible key names
        if "load_values" in history:
            load_curve = np.array(history["load_values"], dtype=float)
        elif "load_param" in history:
            load_curve = np.array(history["load_param"], dtype=float)
        else:
            # Try to infer from steps
            load_curve = np.array(history.get("steps", []), dtype=float)
        
        # Extract data with flexible key names
        data = {
            "load": load_curve,
            "P11": _get_curve_flexible(history, "P_avg", 0, 0),
            "P22": _get_curve_flexible(history, "P_avg", 1, 1),
            "Wbar": _get_curve_flexible(history, "W_avg"),
            "Jbar": _get_curve_flexible(history, "J_avg"),
        }
        
        if dim >= 2:
            data["P12"] = _get_curve_flexible(history, "P_avg", 0, 1)
            data["P21"] = _get_curve_flexible(history, "P_avg", 1, 0)
        
        if dim == 3:
            data["P33"] = _get_curve_flexible(history, "P_avg", 2, 2)
            data["P13"] = _get_curve_flexible(history, "P_avg", 0, 2)
            data["P23"] = _get_curve_flexible(history, "P_avg", 1, 2)
            data["P31"] = _get_curve_flexible(history, "P_avg", 2, 0)
            data["P32"] = _get_curve_flexible(history, "P_avg", 2, 1)
        
        # Extract tangent components if available
        # Check for different possible key names
        ceff_key = None
        if "Ceff" in history:
            ceff_key = "Ceff"
        elif "C_eff" in history:
            ceff_key = "C_eff"
        
        if ceff_key and history[ceff_key]:
            ceff_array = history[ceff_key]
            if dim == 2:
                data["C11"] = np.array([np.nan if C is None else C[0, 0] for C in history["Ceff"]], dtype=float)
                data["C22"] = np.array([np.nan if C is None else C[3, 3] for C in history["Ceff"]], dtype=float)
                data["C12"] = np.array([np.nan if C is None else C[0, 3] for C in history["Ceff"]], dtype=float)
                data["C21"] = np.array([np.nan if C is None else C[3, 0] for C in history["Ceff"]], dtype=float)
                data["C33"] = np.array([np.nan if C is None else C[2, 2] for C in history["Ceff"]], dtype=float)
            
            elif dim == 3:
                data["C11"] = np.array([np.nan if C is None else C[0, 0] for C in history["Ceff"]], dtype=float)
                data["C22"] = np.array([np.nan if C is None else C[4, 4] for C in history["Ceff"]], dtype=float)
                data["C33"] = np.array([np.nan if C is None else C[8, 8] for C in history["Ceff"]], dtype=float)
                data["C12_12"] = np.array([np.nan if C is None else C[1, 1] for C in history["Ceff"]], dtype=float)
                data["C13_13"] = np.array([np.nan if C is None else C[2, 2] for C in history["Ceff"]], dtype=float)
                data["C23_23"] = np.array([np.nan if C is None else C[5, 5] for C in history["Ceff"]], dtype=float)
                data["C11_22"] = np.array([np.nan if C is None else C[0, 4] for C in history["Ceff"]], dtype=float)
                data["C11_33"] = np.array([np.nan if C is None else C[0, 8] for C in history["Ceff"]], dtype=float)
                data["C22_33"] = np.array([np.nan if C is None else C[4, 8] for C in history["Ceff"]], dtype=float)
        
        # # Add state history metadata if available
        # if state_history is not None:
        #     data["has_state_history"] = True
        #     data["num_state_snapshots"] = len(state_history)
        
        summary[load_type] = data
    
    return summary


def _get_curve_flexible(history: dict, key: str, idx1: int = None, idx2: int = None) -> np.ndarray:
    """
    Extract curve data from history dictionary with flexible key handling.
    """
    # Try different possible key names
    if key == "P_avg":
        possible_keys = ["P_avg", "Pbar", "P"]
    elif key == "W_avg":
        possible_keys = ["W_avg", "Wbar", "W"]
    elif key == "J_avg":
        possible_keys = ["J_avg", "Jbar", "J"]
    else:
        possible_keys = [key]
    
    # Find the actual key in history
    actual_key = None
    for k in possible_keys:
        if k in history:
            actual_key = k
            break
    
    if actual_key is None:
        return np.array([])
    
    values = history[actual_key]
    
    if idx1 is not None and idx2 is not None:
        # Tensor component
        if len(values) > 0 and isinstance(values[0], np.ndarray):
            return np.array([v[idx1, idx2] for v in values], dtype=float)
        else:
            return np.array([])
    else:
        # Scalar
        return np.array(values, dtype=float)
    

def plot_homogenization_summary(
    summary: dict,
    dim: int,
    save: bool = False,
    save_prefix: str = "summary",
    show: bool = True
):
    """Plot comprehensive nonlinear homogenization curves for 2D and 3D based on the provided summary."""

    colors = {
        "uniaxial_x": "tab:blue",
        "uniaxial_y": "tab:orange",
        "uniaxial_z": "tab:cyan",
        "biaxial": "tab:green",
        "triaxial": "tab:brown",
        "shear_xy": "tab:red",
        "shear_xz": "tab:purple",
        "shear_yz": "tab:pink",
        "iso_stretch": "tab:gray",
    }

    # --- 1. Stress Curves ---
    n_stress = 3 if dim == 2 else 6
    fig1, axs1 = plt.subplots(1, n_stress, figsize=(5*n_stress, 4.5))
    fig1.suptitle("Macroscopic Stress Responses (First Piola-Kirchhoff)", fontsize=14, y=1.02)

    stress_comps = ["P11", "P22", "P12"]
    if dim == 3:
        stress_comps = ["P11", "P22", "P33", "P12", "P13", "P23"]

    axs1 = np.atleast_1d(axs1)

    for ax, comp in zip(axs1, stress_comps):
        for lt, data in summary.items():
            if comp in data:
                ax.plot(data["load"], data[comp], marker="o", markersize=4, lw=1.5, 
                        color=colors.get(lt, "black"), label=lt)
        ax.set_title(f"Avg {comp}")
        ax.set_xlabel("Load Parameter")
        ax.set_ylabel("Stress")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig1.tight_layout()
    if save:
        fig1.savefig(f"{save_prefix}_stresses.png", dpi=300, bbox_inches='tight')

    # --- 2. Energy & Jacobian ---
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4.5))
    fig2.suptitle("Energy & Volumetric Response", fontsize=14, y=1.02)

    for lt, data in summary.items():
        c = colors.get(lt, "black")
        axs2[0].plot(data["load"], data["Wbar"], marker="s", markersize=4, lw=1.5, color=c, label=lt)
        axs2[1].plot(data["load"], data["Jbar"], marker="d", markersize=4, lw=1.5, color=c, label=lt)

    axs2[0].set_title("Avg Energy Density (W̄)")
    axs2[1].set_title("Avg Jacobian (J̄)")
    for ax in axs2:
        ax.set_xlabel("Load Parameter")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig2.tight_layout()
    if save:
        fig2.savefig(f"{save_prefix}_energy.png", dpi=300, bbox_inches='tight')

    # --- 3. Tangent Stiffness Components ---
    if dim == 2:
        tangent_comps = ["C11", "C22", "C12"]
    else:
        tangent_comps = ["C11", "C22", "C33", "C12_12", "C13_13", "C23_23", "C11_22"]

    n_tangent = len(tangent_comps)
    n_cols = 4 if n_tangent > 4 else n_tangent
    n_rows = (n_tangent + n_cols - 1) // n_cols

    fig3, axs3 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
    fig3.suptitle("Effective Tangent Moduli (C_eff)", fontsize=14, y=1.02)

    axs3 = np.atleast_1d(axs3).flatten()

    for i, comp in enumerate(tangent_comps):
        ax = axs3[i]
        for lt, data in summary.items():
            if comp in data:
                x = np.array(data["load"])
                y = np.array(data[comp])
                mask = ~np.isnan(y)
                if np.any(mask):
                    ax.plot(x[mask], y[mask], marker="^", markersize=4, lw=1.5, 
                            color=colors.get(lt, "black"), label=lt)
        
        ax.set_title(f"Component: {comp}")
        ax.set_xlabel("Load Parameter")
        ax.set_ylabel("Modulus Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axs3)):
        fig3.delaxes(axs3[j])

    fig3.tight_layout()
    if save:
        fig3.savefig(f"{save_prefix}_tangents.png", dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_each_load_case(summary: dict, dim: int, save: bool = False, save_prefix: str = "load_case", show: bool = True):
    """Plot comprehensive nonlinear homogenization curves per load case for 2D and 3D based on the provided summary."""

    for load_type, data in summary.items():
        c = "tab:blue"
        load = np.array(data["load"])
        
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"Analysis: {load_type}", fontsize=16)
        
        if dim == 2:
            stress_comps = ["P11", "P22", "P12"]
            tangent_pairs = [("C11", "C22"), ("C12", "C21"), ("C33", "C33")]
        else:
            stress_comps = ["P11", "P22", "P33", "P12", "P13", "P23"]
            tangent_pairs = [("C11", "C22", "C33"), ("C12_12", "C13_13", "C23_23"), ("C11_22", "C11_33", "C22_33")]

        for i, comp in enumerate(stress_comps):
            ax = fig.add_subplot(3, len(stress_comps), i + 1)
            y = np.array(data.get(comp, []))
            if len(y) > 0:
                mask = ~np.isnan(y)
                ax.plot(load[mask], y[mask], 'o-', color=c, markersize=3)
            ax.set_title(comp)
            ax.grid(True, alpha=0.3)

        ax_w = fig.add_subplot(3, 3, 4)
        w = np.array(data.get("Wbar", []))
        if len(w) > 0:
            mask = ~np.isnan(w)
            ax_w.plot(load[mask], w[mask], 's-', color='green')
        ax_w.set_title("Energy (Wbar)")
        ax_w.grid(True, alpha=0.3)
        
        ax_j = fig.add_subplot(3, 3, 5)
        j = np.array(data.get("Jbar", []))
        if len(j) > 0:
            mask = ~np.isnan(j)
            ax_j.plot(load[mask], j[mask], 'd-', color='red')
            # ax_j.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax_j.set_title("Jacobian (Jbar)")
        ax_j.grid(True, alpha=0.3)

        for i, comps in enumerate(tangent_pairs):
            ax = fig.add_subplot(3, 3, 7 + i)
            for comp in comps:
                y_t = np.array(data.get(comp, []))
                if len(y_t) > 0:
                    mask = ~np.isnan(y_t)
                    ax.plot(load[mask], y_t[mask], '^-', label=comp, alpha=0.8)
            ax.set_title(f"Stiffness Group {i+1}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save:
            plt.savefig(f"{save_prefix}_{load_type}.png")
        if show: plt.show()
        else: plt.close(fig)


def _create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k


def _create_pyvista_mesh(mesh):
    points = mesh.geometry.x.copy()
    tdim = mesh.topology.dim

    # Extract connectivity (cell-to-vertex mapping)
    # Note: In dolfinx, connectivity is stored in the mesh.topology object.
    connectivity = mesh.topology.connectivity(tdim, 0).array

    # Set number of nodes per cell and corresponding VTK cell type
    if tdim == 2:
        num_nodes_per_cell = 3
        cell_type = pv.CellType.TRIANGLE  # VTK_TRIANGLE (integer value 5)
    elif tdim == 3:
        num_nodes_per_cell = 4
        cell_type = pv.CellType.TETRA  # VTK_TETRA (integer value 10)
    else:
        raise ValueError("Unsupported mesh topology dimension")

    # Determine number of cells
    num_cells = connectivity.shape[0] // num_nodes_per_cell

    # Create a flat cell array in VTK format:
    # For each cell, the data layout is [n, pt0, pt1, ..., pt(n-1)]
    cells = np.hstack(
        [
            np.full((num_cells, 1), num_nodes_per_cell, dtype=np.int64),
            connectivity.reshape(num_cells, num_nodes_per_cell),
        ]
    ).flatten()

    # Create an array for cell types (one type per cell)
    cell_types = np.full(num_cells, cell_type, dtype=np.uint8)

    # Build the PyVista UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    return grid


def _visualize_deformed_geometry(domain, u_field, cell_markers, geometry, factor=1.0):
    """
    u_field: u or u_total
    """
    grid = _create_pyvista_mesh(domain)

    v_array = u_field.x.array.reshape(grid.n_points, domain.topology.dim)
    if domain.topology.dim == 2:
        disp_3d = np.zeros((grid.n_points, 3))
        disp_3d[:, :2] = v_array
        grid.point_data["Displacement"] = disp_3d
    else:
        grid.point_data["Displacement"] = v_array

    deformed_grid = grid.warp_by_vector("Displacement", factor=factor)

    physical_tags = PhysicalTags()
    matrix_phase_id = 0

    unique_phases = {matrix_phase_id}
    for inc in geometry.inclusions:
        unique_phases.add(inc.phase_id)
        if inc.has_interphase:
            unique_phases.add(inc.interphase_phase_id)
    unique_phases = sorted(list(unique_phases))

    phase_to_value = {p_id: float(p_id) for p_id in unique_phases}
    tag_to_phase_value = {}
    for p_id in unique_phases:
        cell_tag = physical_tags.cell_tag_for_phase(p_id, matrix_phase_id)
        tag_to_phase_value[cell_tag] = phase_to_value[p_id]

    val_field = _create_piecewise_constant_field(domain, cell_markers, tag_to_phase_value, name="Phase")
    deformed_grid.cell_data["Phase"] = val_field.x.array

    plotter = pv.Plotter(window_size=[1200, 800], title="Deformed Geometry Visualization")
    plotter.set_background("white")
    cmap = plt.get_cmap("tab10")
    tol = 1e-5

    for i, p_id in enumerate(unique_phases):
        val = phase_to_value[p_id]
        phase_grid = deformed_grid.threshold([val - tol, val + tol], scalars="Phase")

        if phase_grid.n_cells == 0:
            continue

        if p_id == matrix_phase_id:
            color = "lightblue"
            opacity = 0.3
            label = "Matrix"
        elif any(inc.has_interphase and p_id == inc.interphase_phase_id for inc in geometry.inclusions):
            color = "orange"
            opacity = 0.6
            label = f"Interphase (Phase {p_id})"
        else:
            color = cmap(i % 10)
            opacity = 0.8
            label = f"Inclusion (Phase {p_id})"

        plotter.add_mesh(
            phase_grid,
            color=color,
            opacity=opacity,
            show_edges=True,
            line_width=0.5,
            label=label
        )

    plotter.add_mesh(grid, style='wireframe', color='black', opacity=0.05, label="Undeformed")


    plotter.add_legend()
    plotter.view_isometric()
    plotter.add_axes()

    return plotter.show(jupyter_backend="html")


def _save_history_csv(history: Dict, filename: str, dim: int):
    """Save load case history to CSV file."""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['step', 'load_value', 'W_avg', 'J_avg', 'converged', 'iterations']
        
        # Add stress components
        if dim == 2:
            header.extend(['P_xx', 'P_yy', 'P_xy'])
        elif dim == 3:
            header.extend(['P_xx', 'P_yy', 'P_zz', 'P_xy', 'P_yz', 'P_xz'])
            
        # Add tangent stiffness if available
        if history['C_eff'][0] is not None:
            n_comps = 4 if dim == 2 else 9
            for i in range(n_comps):
                for j in range(n_comps):
                    header.append(f'C_{i}{j}')
        
        writer.writerow(header)
        
        # Write data
        for i in range(len(history['steps'])):
            row = [
                history['steps'][i],
                history['load_values'][i],
                history['W_avg'][i],
                history['J_avg'][i],
                history['converged'][i],
                history['iterations'][i],
            ]
            
            # Add stress components
            P = history['P_avg'][i]
            if dim == 2:
                row.extend([P[0, 0], P[1, 1], P[0, 1]])
            elif dim == 3:
                row.extend([P[0, 0], P[1, 1], P[2, 2], P[0, 1], P[1, 2], P[0, 2]])
            
            # Add tangent if available
            if history['C_eff'][i] is not None:
                row.extend(history['C_eff'][i].flatten())
                
            writer.writerow(row)


def _save_state_history_csv(state_history: List[Tuple[float, Dict]], filename: str, dim: int):
    """Save material state evolution to CSV file."""
    import csv
    
    if not state_history:
        return
        
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Get state variable names from first snapshot
        first_snapshot = state_history[0][1]
        state_var_names = []
        for phase_id, phase_states in first_snapshot.items():
            if phase_states:
                first_cell_state = next(iter(phase_states.values()))
                state_var_names = first_cell_state.state_variable_names
                break
        
        # Write header
        header = ['load_value', 'phase_id', 'cell_idx']
        for var_name in state_var_names:
            # Get shape from first state
            for phase_id, phase_states in first_snapshot.items():
                if phase_states:
                    first_cell_state = next(iter(phase_states.values()))
                    var_data = first_cell_state.get_state(var_name)
                    shape = var_data.shape[1:]  # Skip quadrature point dimension
                    if len(shape) == 0:
                        header.append(var_name)
                    elif len(shape) == 1:
                        for i in range(shape[0]):
                            header.append(f'{var_name}_{i}')
                    elif len(shape) == 2:
                        for i in range(shape[0]):
                            for j in range(shape[1]):
                                header.append(f'{var_name}_{i}{j}')
                    break
        
        writer.writerow(header)
        
        # Write data for each load step
        for load_val, snapshot in state_history:
            for phase_id, phase_states in snapshot.items():
                for cell_idx, state in phase_states.items():
                    row = [load_val, phase_id, cell_idx]
                    
                    for var_name in state_var_names:
                        var_data = state.get_state(var_name)
                        # Average over quadrature points
                        avg_data = np.mean(var_data, axis=0)
                        row.extend(avg_data.flatten())
                    
                    writer.writerow(row)


__all__ = [
    # nlhelpers
    "plot_homogenization_summary",
    "plot_each_load_case",
]
