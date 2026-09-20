from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ufl import ln, tr, det

import numpy as np
import ufl
from dolfinx import fem, mesh
import basix
import dolfinx

from .mesh import PhysicalTags

logger = logging.getLogger(__name__)

def _validate_isotropic_elastic_constants(young_modulus: float, poisson_ratio: float) -> None:
    """Reject elastic constants that make the isotropic law singular or nonphysical."""
    if not np.isfinite(young_modulus) or young_modulus <= 0:
        raise ValueError("young_modulus must be finite and greater than zero")
    if not np.isfinite(poisson_ratio) or not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be finite and satisfy -1 < nu < 0.5")


# =============================================================================
# Quadrature Point Utilities
# =============================================================================

class QuadraturePointEvaluator:
    """Evaluate cellwise deformation gradients for material-state updates.

    For first-order simplex elements the deformation gradient is constant in
    each cell, so the DG0 value is repeated at the integration samples. This
    evaluator is not valid for history-dependent Q1 quadrilateral or
    hexahedral elements; the nonlinear driver rejects that combination.
    """
    
    def __init__(self, mesh: dolfinx.mesh.Mesh, degree: int = 4):
        self.mesh = mesh
        self.dim = mesh.topology.dim
        self.degree = degree
        
        # Create quadrature rule for reference
        # Convert dolfinx cell type to basix cell type
        cell_type = mesh.topology.cell_type
        
        # Map dolfinx CellType to basix CellType
        if cell_type == dolfinx.mesh.CellType.triangle:
            basix_cell = basix.CellType.triangle
        elif cell_type == dolfinx.mesh.CellType.quadrilateral:
            basix_cell = basix.CellType.quadrilateral
        elif cell_type == dolfinx.mesh.CellType.tetrahedron:
            basix_cell = basix.CellType.tetrahedron
        elif cell_type == dolfinx.mesh.CellType.hexahedron:
            basix_cell = basix.CellType.hexahedron
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
        
        quadrature_points, weights = basix.make_quadrature(basix_cell, degree)
        self.quad_points = quadrature_points
        self.quad_weights = weights
        self.num_quad_points = len(weights)
        
        # Pre-create quadrature function spaces
        self._create_quadrature_spaces()
        
    def _create_quadrature_spaces(self):
        """Create standard function spaces (not quadrature)."""
        # Use standard CG spaces for interpolation
        self.V_scalar = fem.functionspace(self.mesh, ("Lagrange", 1))
        self.V_vector = fem.functionspace(self.mesh, ("Lagrange", 1, (self.dim,)))
        self.V_tensor = fem.functionspace(
            self.mesh,
            ("Lagrange", 1, (self.dim, self.dim)),
        )

    def compute_deformation_gradient_at_quad_points(
        self,
        u: fem.Function,
        F_macro: fem.Constant,
        cells: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Compute the cellwise total deformation gradient.
        """
        if cells is None:
            cells = np.arange(self.mesh.topology.index_map(self.mesh.topology.dim).size_local)
        
        F_dict = {}
        F_macro_np = F_macro.value.reshape(self.dim, self.dim)
        
        # Project gradient to DG space for evaluation
        V_DG = fem.functionspace(self.mesh, ("DG", 0, (self.dim, self.dim)))
        grad_proj = fem.Function(V_DG)
        
        # Compute gradient expression
        grad_u = ufl.grad(u)
        
        # Project to DG space
        from dolfinx.fem import Expression
        grad_expr = Expression(grad_u, V_DG.element.interpolation_points())
        grad_proj.interpolate(grad_expr)
        
        # Extract values per cell
        for cell_idx in cells:
            # Get cell dof (one dof per cell for DG0)
            cell_dof = V_DG.dofmap.cell_dofs(cell_idx)[0]
            grad_val = grad_proj.x.array[cell_dof * (self.dim * self.dim):(cell_dof + 1) * (self.dim * self.dim)]
            grad_val = grad_val.reshape(self.dim, self.dim)
            
            # Replicate for all quadrature points
            F_cell = np.tile(F_macro_np + grad_val, (self.num_quad_points, 1, 1))
            F_dict[cell_idx] = F_cell
        
        return F_dict
    
    def compute_cell_volumes(self, cells: Optional[np.ndarray] = None) -> Dict[int, float]:
        """Return exact geometric measures for local mesh cells.

        A DG0 test-function integral gives one entry per cell and delegates the
        geometry mapping and Jacobian evaluation to DOLFINx. This is valid for
        nonuniform and curved meshes and avoids relying on private C++ helpers.
        """
        num_local = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        if cells is None:
            cells = np.arange(num_local, dtype=np.int32)
        else:
            cells = np.asarray(cells, dtype=np.int32)

        V_dg0 = fem.functionspace(self.mesh, ("DG", 0))
        test = ufl.TestFunction(V_dg0)
        cell_integrals = fem.assemble_vector(fem.form(test * ufl.dx))

        volumes: Dict[int, float] = {}
        for cell_idx in cells:
            dof = V_dg0.dofmap.cell_dofs(int(cell_idx))[0]
            volumes[int(cell_idx)] = float(cell_integrals.array[dof])
        return volumes


# =============================================================================
# Base Material Classes with Full State Management
# =============================================================================

@dataclass
class MaterialState:
    """Container for material state variables at integration points."""
    
    def __init__(self, num_quad_points: int, state_variable_names: List[str]):
        self.num_quad_points = num_quad_points
        self.state_variable_names = state_variable_names
        self._state_data: Dict[str, np.ndarray] = {}
        
    def initialize_state(self, name: str, shape: Tuple[int, ...], initial_value: float = 0.0):
        """Initialize a state variable with given shape."""
        full_shape = (self.num_quad_points,) + shape
        self._state_data[name] = np.full(full_shape, initial_value, dtype=np.float64)
        
    def get_state(self, name: str) -> np.ndarray:
        return self._state_data[name]
    
    def set_state(self, name: str, values: np.ndarray):
        self._state_data[name] = values.copy()
        
    def copy(self):
        """Deep copy of state."""
        new_state = MaterialState(self.num_quad_points, self.state_variable_names)
        for name, data in self._state_data.items():
            new_state._state_data[name] = data.copy()
        return new_state


class NonlinearMaterialModel(ABC):
    """Abstract base class for nonlinear material models."""
    
    @abstractmethod
    def psi_form(self, F=None, **kwargs) -> ufl.core.expr.Expr:
        """Strain energy density."""
        pass
    
    @abstractmethod
    def evaluate_energy(self, F: np.ndarray, dim: int) -> float:
        """
        Evaluate strain energy density for a given numeric deformation gradient.
        
        This method should be overridden by subclasses to provide efficient
        numeric evaluation that matches the symbolic psi_form.
        """
        pass

    @abstractmethod
    def requires_history(self) -> bool:
        """Whether material requires state variable tracking."""
        pass
    
    @abstractmethod
    def get_state_variable_names(self) -> List[str]:
        """Return list of state variable names."""
        pass
    
    @abstractmethod
    def initialize_state(self, num_quad_points: int) -> MaterialState:
        """Create and initialize material state."""
        pass
    
    @abstractmethod
    def update_state(
        self, 
        state: MaterialState, 
        F_new: np.ndarray,
        dt: float,
        quad_weights: np.ndarray,
        cell_idx: int
    ) -> Tuple[bool, int]:
        """
        Update material state after converged step.
        
        Returns
        -------
        Tuple[bool, int]
            (converged, iterations) for local return mapping
        """
        pass
    
    @abstractmethod
    def get_quadrature_point_stress(
        self,
        state: MaterialState,
        F: np.ndarray,
        quad_point_idx: int
    ) -> np.ndarray:
        """
        Compute first Piola-Kirchhoff stress at a quadrature point.
        
        Parameters
        ----------
        state : MaterialState
            Material state
        F : np.ndarray
            Deformation gradient at quadrature point (dim, dim)
        quad_point_idx : int
            Quadrature point index
            
        Returns
        -------
        np.ndarray
            First Piola-Kirchhoff stress tensor (dim, dim)
        """
        pass   


# =============================================================================
# Hyperelastic Materials
# =============================================================================

class HyperelasticMaterial(NonlinearMaterialModel):
    """Base class for rate-independent hyperelastic materials."""
    
    def requires_history(self) -> bool:
        return False
    
    def get_state_variable_names(self) -> List[str]:
        return []
    
    def initialize_state(self, num_quad_points: int) -> MaterialState:
        return MaterialState(num_quad_points, [])
    
    def update_state(self, state: MaterialState, F_new: np.ndarray, dt: float, 
                     quad_weights: np.ndarray, cell_idx: int) -> Tuple[bool, int]:
        return True, 0
    
    @abstractmethod
    def evaluate_energy(self, F: np.ndarray, dim: int) -> float:
        """Numeric evaluation of strain energy density."""
        pass


@dataclass
class NeoHookeanIsotropic(HyperelasticMaterial):
    """
    Neo-Hookean finite-strain nonlinear-elastic material.

    It is defined by Young's
    modulus and Poisson's ratio and is intended for phases that follow a
    finite-strain constitutive law.

    Attributes
    ----------
    young_modulus:
        Young's modulus E of the material.
    poisson_ratio:
        Poisson's ratio nu of the material.
    metadata:
        Optional free-form dictionary for additional information.
    """
    young_modulus: float
    poisson_ratio: float
    name: str = 'NeoHookeanIsotropic'
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_isotropic_elastic_constants(self.young_modulus, self.poisson_ratio)

    @property
    def mu(self):
        """Shear modulus"""
        E = self.young_modulus
        nu = self.poisson_ratio
        return E / (2.0 * (1.0 + nu))

    @property
    def lmbda(self):
        """First Lame parameter"""
        E = self.young_modulus
        nu = self.poisson_ratio
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def psi_form(self, F):
        """
        Compressible Neo-Hookean strain energy density.

        Parameters
        ----------
        F : UFL tensor
            Deformation gradient.

        Returns
        -------
        UFL expression
            Strain energy density.
        """
        dim = F.ufl_shape[0]
        C = F.T * F
        I1 = tr(C)
        J = det(F)
        return (self.mu / 2) * (I1 - dim - 2 * ln(J)) + (self.lmbda / 2) * (J - 1) ** 2
    
    def evaluate_energy(self, F: np.ndarray, dim: int) -> float:
        """Numeric evaluation matching psi_form."""
        F = np.asarray(F, dtype=float)
        if F.shape != (dim, dim):
            raise ValueError(f"F must have shape ({dim}, {dim}).")
        if not np.all(np.isfinite(F)):
            raise ValueError("Neo-Hookean energy requires a finite deformation gradient.")
        J = np.linalg.det(F)
        if not np.isfinite(J) or J <= 0.0:
            raise ValueError("Neo-Hookean energy requires det(F) > 0.")
        C = F.T @ F
        I1 = np.trace(C)
        return (self.mu / 2) * (I1 - dim - 2 * np.log(J)) + (self.lmbda / 2) * (J - 1) ** 2
    
    def get_quadrature_point_stress(self, state: MaterialState, F: np.ndarray, 
                                    quad_point_idx: int) -> np.ndarray:
        """Compute PK1 stress at quadrature point."""
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] not in (2, 3):
            raise ValueError("F must be a square 2D or 3D deformation gradient.")
        if not np.all(np.isfinite(F)):
            raise ValueError("Neo-Hookean stress requires a finite deformation gradient.")
        J = np.linalg.det(F)
        if not np.isfinite(J) or J <= 0.0:
            raise ValueError("Neo-Hookean stress requires det(F) > 0.")
        Finv = np.linalg.inv(F)
        
        P = self.mu * (F - Finv.T) + self.lmbda * J * (J - 1) * Finv.T
        return P


@dataclass
class LinearElasticIsotropic:
    """Linear elastic material (small strains)."""
    young_modulus: float
    poisson_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_isotropic_elastic_constants(self.young_modulus, self.poisson_ratio)


# =============================================================================
# Viscoelastic Materials with Full Implementation
# =============================================================================

@dataclass
class ViscoelasticGeneralizedMaxwell(NonlinearMaterialModel):
    """
    Finite-strain viscoelasticity using generalized Maxwell model.
    
    State variables: Cv_i (viscous right Cauchy-Green tensor for each branch)
    """
    
    equilibrium_material: NonlinearMaterialModel
    num_branches: int
    shear_moduli: List[float]
    relaxation_times: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.num_branches, int) or self.num_branches < 1:
            raise ValueError("num_branches must be a positive integer")
        if len(self.shear_moduli) != self.num_branches:
            raise ValueError("shear_moduli length must equal num_branches")
        if len(self.relaxation_times) != self.num_branches:
            raise ValueError("relaxation_times length must equal num_branches")
        if not all(np.isfinite(value) and value > 0.0 for value in self.shear_moduli):
            raise ValueError("branch shear moduli must be finite and greater than zero")
        if not all(np.isfinite(value) and value > 0.0 for value in self.relaxation_times):
            raise ValueError("relaxation times must be finite and greater than zero")
        
    def requires_history(self) -> bool:
        return True
    
    def get_state_variable_names(self) -> List[str]:
        return [f"Cv_{i}" for i in range(self.num_branches)]
    
    def initialize_state(self, num_quad_points: int) -> MaterialState:
        state = MaterialState(num_quad_points, self.get_state_variable_names())
        for i in range(self.num_branches):
            state.initialize_state(f"Cv_{i}", shape=(3, 3), initial_value=0.0)
            Cv_i = state.get_state(f"Cv_{i}")
            for q in range(num_quad_points):
                Cv_i[q, 0, 0] = 1.0
                Cv_i[q, 1, 1] = 1.0
                Cv_i[q, 2, 2] = 1.0
        return state
    
    def psi_form(self, F=None, **kwargs):
        """
        Total strain energy = equilibrium + non-equilibrium branches.
        For variational form, only equilibrium part contributes to potential.
        Non-equilibrium stress is handled via internal variables.
        """
        return self.equilibrium_material.psi_form(F=F, **kwargs)
    
    def evaluate_energy(self, F: np.ndarray, dim: int) -> float:
        """
        Evaluate the equilibrium energy when no material state is available.

        Use :meth:`get_quadrature_point_energy` when a state is available to
        include recoverable energy in the Maxwell springs.
        """
        return self.equilibrium_material.evaluate_energy(F, dim)

    def get_quadrature_point_energy(
        self,
        state: MaterialState,
        F: np.ndarray,
        quad_point_idx: int,
    ) -> float:
        """Return equilibrium plus recoverable Maxwell-branch energy."""
        dim = F.shape[0]
        C = F.T @ F
        energy = self.equilibrium_material.evaluate_energy(F, dim)
        for branch, shear_modulus in enumerate(self.shear_moduli):
            Cv = state.get_state(f"Cv_{branch}")[
                quad_point_idx, :dim, :dim
            ]
            elastic_metric = C @ np.linalg.inv(Cv)
            energy += 0.5 * shear_modulus * (
                np.trace(elastic_metric)
                - dim
                - np.log(np.linalg.det(elastic_metric))
            )
        return float(energy)
    
    def update_state(
        self, 
        state: MaterialState, 
        F_new: np.ndarray,
        dt: float,
        quad_weights: np.ndarray,
        cell_idx: int
    ) -> Tuple[bool, int]:
        """
        Update viscous deformation using exponential map.
        """
        dim = F_new.shape[-1]
        
        for i in range(self.num_branches):
            tau = self.relaxation_times[i]
            Cv = state.get_state(f"Cv_{i}")
            
            for q in range(state.num_quad_points):
                F_q = F_new[q, :dim, :dim]
                C_q = F_q.T @ F_q
                
                # Exponential update
                alpha = np.exp(-dt / tau)
                Cv[q, :dim, :dim] = alpha * Cv[q, :dim, :dim] + (1 - alpha) * C_q
                
        return True, 0
    
    def get_quadrature_point_stress(
        self,
        state: MaterialState,
        F: np.ndarray,
        quad_point_idx: int
    ) -> np.ndarray:
        """
        Compute total PK1 stress: P = P_eq + Σ P_neq_i
        """
        dim = F.shape[0]
        
        # Equilibrium stress
        P = self.equilibrium_material.get_quadrature_point_stress(
            state, F, quad_point_idx
        )
        
        # Non-equilibrium stresses
        C = F.T @ F
        for i in range(self.num_branches):
            mu_i = self.shear_moduli[i]
            Cv_i = state.get_state(f"Cv_{i}")[quad_point_idx, :dim, :dim]
            
            Cv_inv = np.linalg.inv(Cv_i)
            
            # Non-equilibrium PK2 stress: S_neq = 2 ∂Ψ/∂C = μ (Cv^{-1} - C^{-1})
            Cinv = np.linalg.inv(C)
            S_neq = mu_i * (Cv_inv - Cinv)
            
            # Convert to PK1: P = F S
            P += F @ S_neq

        return P

@dataclass
class MaterialAssignment:
    """Map phase ids to material models with state management."""
    
    materials_by_phase: Dict[int, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_history_dependence(self) -> bool:
        return any(mat.requires_history() for mat in self.materials_by_phase.values())
    
    def initialize_states(
        self, 
        mesh: mesh.Mesh, 
        cell_tags: mesh.MeshTags,
        quad_evaluator: QuadraturePointEvaluator,
        physical_tags: PhysicalTags = None,
        matrix_phase_id: int = 0,
    ) -> Dict[int, Dict[int, MaterialState]]:
        """
        Initialize material states for all phases with history dependence.
        """
        if physical_tags is None:
            physical_tags = PhysicalTags()
        
        states = {}
        
        for phase_id, material in self.materials_by_phase.items():
            if material.requires_history():
                # Get the physical tag for this phase
                phase_tag = physical_tags.cell_tag_for_phase(phase_id, matrix_phase_id)
                
                # Get cells with this physical tag
                phase_cells = np.where(cell_tags.values == phase_tag)[0]
                
                if len(phase_cells) > 0:
                    phase_states = {}
                    num_quad_points = quad_evaluator.num_quad_points
                    
                    for cell_idx in phase_cells:
                        phase_states[cell_idx] = material.initialize_state(num_quad_points)
                    
                    states[phase_id] = phase_states
                    print(f"  Initialized states for phase {phase_id} (tag {phase_tag}): {len(phase_cells)} cells")
                        
        return states
    
    def update_all_states(
        self,
        states: Dict[int, Dict[int, MaterialState]],
        F_by_cell: Dict[int, np.ndarray],
        dt: float,
        quad_weights: np.ndarray
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
            material = self.materials_by_phase[phase_id]
            
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

__all__ = [
    # materials
    "LinearElasticIsotropic",
    "NonlinearMaterialModel",
    "NeoHookeanIsotropic",
    "MaterialAssignment",
]
