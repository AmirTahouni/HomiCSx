from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ufl import ln, tr, det

import numpy as np
import ufl
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Tuple, List, Callable
from abc import ABC, abstractmethod
import basix
import dolfinx

from .mesh import PhysicalTags

# helpers
def _flatten_tensor(P: np.ndarray) -> np.ndarray:
    """
    Flatten a tensor to vector in STANDARD Voigt notation.
    
    For 2D: [11, 22, 12]  (3 components, not 4!)
    For 3D: [11, 22, 33, 12, 23, 13]  (6 components)
    
    Note: This assumes symmetry (P12 = P21), which is true for PK1 stress
    in the reference configuration for hyperelastic materials.
    """
    dim = P.shape[0]
    if dim == 2:
        # Standard Voigt for 2D: [11, 22, 12]
        # We use 12 (not 21) assuming symmetry
        return np.array([P[0, 0], P[1, 1], P[0, 1]])
    elif dim == 3:
        # Standard Voigt for 3D: [11, 22, 33, 12, 23, 13]
        return np.array([
            P[0, 0], P[1, 1], P[2, 2],
            P[0, 1], P[1, 2], P[0, 2]
        ])
    else:
        return P.flatten()


def _voigt_to_tensor_perturbation(voigt_idx: int, dim: int, eps: float) -> np.ndarray:
    """
    Create a tensor perturbation corresponding to a Voigt component.
    
    Voigt ordering for 2D: [11, 22, 12]
    Voigt ordering for 3D: [11, 22, 33, 12, 23, 13]
    """
    dF = np.zeros((dim, dim))
    
    if dim == 2:
        if voigt_idx == 0:    # 11
            dF[0, 0] = eps
        elif voigt_idx == 1:  # 22
            dF[1, 1] = eps
        elif voigt_idx == 2:  # 12 (and 21 for symmetry)
            dF[0, 1] = eps
            dF[1, 0] = eps  # Symmetric perturbation
    elif dim == 3:
        if voigt_idx == 0:    # 11
            dF[0, 0] = eps
        elif voigt_idx == 1:  # 22
            dF[1, 1] = eps
        elif voigt_idx == 2:  # 33
            dF[2, 2] = eps
        elif voigt_idx == 3:  # 12
            dF[0, 1] = eps
            dF[1, 0] = eps
        elif voigt_idx == 4:  # 23
            dF[1, 2] = eps
            dF[2, 1] = eps
        elif voigt_idx == 5:  # 13
            dF[0, 2] = eps
            dF[2, 0] = eps
    
    return dF


def _voigt_to_tensor_perturbation(voigt_idx: int, dim: int, eps: float) -> np.ndarray:
    """
    Create a tensor perturbation corresponding to a Voigt component.
    
    Voigt ordering for 2D: [11, 22, 12]
    Voigt ordering for 3D: [11, 22, 33, 12, 23, 13]
    
    Note: Perturbations are symmetric (e.g., perturbing 12 also perturbs 21).
    """
    dF = np.zeros((dim, dim))
    
    if dim == 2:
        if voigt_idx == 0:    # 11
            dF[0, 0] = eps
        elif voigt_idx == 1:  # 22
            dF[1, 1] = eps
        elif voigt_idx == 2:  # 12 (and 21 for symmetry)
            dF[0, 1] = eps
            dF[1, 0] = eps  # Symmetric perturbation
    elif dim == 3:
        if voigt_idx == 0:    # 11
            dF[0, 0] = eps
        elif voigt_idx == 1:  # 22
            dF[1, 1] = eps
        elif voigt_idx == 2:  # 33
            dF[2, 2] = eps
        elif voigt_idx == 3:  # 12
            dF[0, 1] = eps
            dF[1, 0] = eps
        elif voigt_idx == 4:  # 23
            dF[1, 2] = eps
            dF[2, 1] = eps
        elif voigt_idx == 5:  # 13
            dF[0, 2] = eps
            dF[2, 0] = eps
    
    return dF

# =============================================================================
# Quadrature Point Utilities
# =============================================================================

class QuadraturePointEvaluator:
    """
    Evaluate fields at quadrature points for state variable updates.
    Uses projection to quadrature space for reliable evaluation.
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
        self.V_scalar = fem.functionspace(self.mesh, ("CG", 1))
        self.V_vector = fem.functionspace(self.mesh, ("CG", 1, (self.dim,)))
        self.V_tensor = fem.functionspace(self.mesh, ("CG", 1, (self.dim, self.dim)))

    def compute_deformation_gradient_at_quad_points(
        self,
        u: fem.Function,
        F_macro: fem.Constant,
        cells: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Compute total deformation gradient at cell centers as approximation.
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
        """Compute volume for each cell."""
        from dolfinx.cpp.mesh import cell_volume
        
        if cells is None:
            cells = np.arange(self.mesh.topology.index_map(self.mesh.topology.dim).size_local)
            
        volumes = {}
        for cell_idx in cells:
            cell = dolfinx.mesh.Cell(self.mesh, cell_idx)
            volumes[cell_idx] = cell_volume(self.mesh, cell)
            
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
        J = np.linalg.det(F)
        C = F.T @ F
        I1 = np.trace(C)
        return (self.mu / 2) * (I1 - dim - 2 * np.log(J)) + (self.lmbda / 2) * (J - 1) ** 2
    
    def get_quadrature_point_stress(self, state: MaterialState, F: np.ndarray, 
                                    quad_point_idx: int) -> np.ndarray:
        """Compute PK1 stress at quadrature point."""
        dim = F.shape[0]
        J = np.linalg.det(F)
        Finv = np.linalg.inv(F)
        
        P = self.mu * (F - Finv.T) + self.lmbda * J * (J - 1) * Finv.T
        return P


@dataclass
class LinearElasticIsotropic:
    """Linear elastic material (small strains)."""
    young_modulus: float
    poisson_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Viscoelastic Materials with Full Implementation
# =============================================================================

@dataclass
class ViscoelasticGeneralizedMaxwell(NonlinearMaterialModel):
    """
    Finite-strain viscoelasticity using generalized Maxwell model.
    
    State variables: Cv_i (inverse viscous right Cauchy-Green for each branch)
    """
    
    equilibrium_material: NonlinearMaterialModel
    num_branches: int
    shear_moduli: List[float]
    relaxation_times: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        assert len(self.shear_moduli) == self.num_branches
        assert len(self.relaxation_times) == self.num_branches
        
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
        Numeric evaluation of strain energy density.
        For viscoelasticity, only the equilibrium part contributes to stored energy.
        The non-equilibrium branches are dissipative.
        """
        return self.equilibrium_material.evaluate_energy(F, dim)
    
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
        FinvT = np.linalg.inv(F).T
        
        for i in range(self.num_branches):
            mu_i = self.shear_moduli[i]
            Cv_i = state.get_state(f"Cv_{i}")[quad_point_idx, :dim, :dim]
            
            # Elastic right Cauchy-Green for this branch: Ce = F^T F Cv^{-1}
            Cv_inv = np.linalg.inv(Cv_i)
            Ce = C @ Cv_inv
            
            # Non-equilibrium PK2 stress: S_neq = 2 ∂Ψ/∂C = μ (Cv^{-1} - C^{-1})
            Cinv = np.linalg.inv(C)
            S_neq = mu_i * (Cv_inv - Cinv)
            
            # Convert to PK1: P = F S
            P += F @ S_neq

        # # Debug: print stress contributions
        # if quad_point_idx == 0:
        #     P_eq = self.equilibrium_material.get_quadrature_point_stress(state, F, quad_point_idx)
        #     # P_total = self.get_quadrature_point_stress(state, F, quad_point_idx)
        #     # print(f"P_eq[0,0]={P_eq[0,0]:.4f}, P_total[0,0]={P_total[0,0]:.4f}")
        #     print(f"    [DEBUG qp=0] P_eq[0,0]={P_eq[0,0]:.4f}, P_total[0,0]={P[0,0]:.4f}, diff={P[0,0] - P_eq[0,0]:.4f}")
            
        return P
    
    def _update_state_for_perturbation(
        self,
        state: MaterialState,
        F: np.ndarray,
        F_perturbed: np.ndarray,
        dt: float,
        quad_point_idx: int
    ):
        """Update state for a perturbation (used in tangent computation)."""
        dim = F.shape[0]
        C = F.T @ F
        C_pert = F_perturbed.T @ F_perturbed
        
        for branch in range(self.num_branches):
            tau = self.relaxation_times[branch]
            alpha = np.exp(-dt / tau)
            
            Cv = state.get_state(f"Cv_{branch}")[quad_point_idx, :dim, :dim]
            Cv_pert = alpha * Cv + (1 - alpha) * C_pert
            
            state.get_state(f"Cv_{branch}")[quad_point_idx, :dim, :dim] = Cv_pert

        def get_algorithmic_tangent(
            self,
            state: MaterialState,
            F: np.ndarray,
            quad_point_idx: int,
            dt: float
        ) -> np.ndarray:
            """
            Compute viscoelastic tangent using central difference with state update.
            """
            dim = F.shape[0]
            voigt_size = 3 if dim == 2 else 6
            eps = 1e-5  # Optimal step size for finite difference
            
            tangent = np.zeros((voigt_size, voigt_size))
            
            import copy
            
            for i in range(voigt_size):
                dF = _voigt_to_tensor_perturbation(i, dim, eps)
                
                Fp = F + dF
                Fm = F - dF
                
                # Deep copy states for perturbation
                state_p = copy.deepcopy(state)
                state_m = copy.deepcopy(state)
                
                # Update states with perturbed F
                self._update_state_for_perturbation(state_p, F, Fp, dt, quad_point_idx)
                self._update_state_for_perturbation(state_m, F, Fm, dt, quad_point_idx)
                
                # Compute perturbed stresses
                Pp = self.get_quadrature_point_stress(state_p, Fp, quad_point_idx)
                Pm = self.get_quadrature_point_stress(state_m, Fm, quad_point_idx)
                
                Pp_voigt = _flatten_tensor(Pp)
                Pm_voigt = _flatten_tensor(Pm)
                
                tangent[:, i] = (Pp_voigt - Pm_voigt) / (2 * eps)
            
            # Enforce symmetry
            tangent = 0.5 * (tangent + tangent.T)
            
            return tangent

# =============================================================================
# Plastic Materials with Return Mapping
# =============================================================================

@dataclass
class J2Plasticity(NonlinearMaterialModel):
    """
    Finite-strain J2 plasticity with isotropic hardening and return mapping.
    """
    
    elastic_material: NonlinearMaterialModel
    yield_stress: float
    hardening_modulus: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Material parameters
        self.mu = self.elastic_material.mu
        self.kappa = self.elastic_material.bulk_modulus
        
        # Algorithmic parameters
        self.max_local_iter = 50
        self.local_tol = 1e-10
        
    def requires_history(self) -> bool:
        return True
    
    def get_state_variable_names(self) -> List[str]:
        return ["Fp", "alpha"]
    
    def initialize_state(self, num_quad_points: int) -> MaterialState:
        state = MaterialState(num_quad_points, self.get_state_variable_names())
        
        # Initialize Fp as identity
        state.initialize_state("Fp", shape=(3, 3), initial_value=0.0)
        Fp = state.get_state("Fp")
        for q in range(num_quad_points):
            Fp[q, 0, 0] = 1.0
            Fp[q, 1, 1] = 1.0
            Fp[q, 2, 2] = 1.0
            
        # Initialize equivalent plastic strain
        state.initialize_state("alpha", shape=(1,), initial_value=0.0)
        
        return state
    
    def psi_form(self, F=None, **kwargs):
        """Elastic stored energy."""
        return self.elastic_material.psi_form(F=F, **kwargs)
    
    def evaluate_energy(self, F: np.ndarray, dim: int) -> float:
        """
        Numeric evaluation of stored elastic energy.
        Uses the elastic part of the deformation: Fe = F · Fp^{-1}
        """
        # This would require access to the state to get Fp
        # For now, just use total F (approximation for elastic steps)
        return self.elastic_material.evaluate_energy(F, dim)
    
    def _compute_elastic_trial(
        self, 
        F: np.ndarray, 
        Fp_old: np.ndarray, 
        alpha_old: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute elastic trial state.
        
        Returns
        -------
        Fe_tr : np.ndarray
            Trial elastic deformation gradient
        be_tr : np.ndarray
            Trial elastic left Cauchy-Green tensor
        tau_tr : np.ndarray
            Trial Kirchhoff stress
        f_tr : float
            Trial yield function value
        """
        dim = F.shape[0]
        
        # Trial elastic deformation gradient
        Fp_inv = np.linalg.inv(Fp_old)
        Fe_tr = F @ Fp_inv
        
        # Trial elastic left Cauchy-Green tensor
        be_tr = Fe_tr @ Fe_tr.T
        
        # Trial Kirchhoff stress (Neo-Hookean)
        J = np.linalg.det(F)
        dev_be = be_tr - (1/3) * np.trace(be_tr) * np.eye(dim)
        tau_tr = self.kappa * np.log(J) * np.eye(dim) + self.mu * dev_be
        
        # Deviatoric stress
        s_tr = tau_tr - (1/3) * np.trace(tau_tr) * np.eye(dim)
        
        # Yield function: f = ||s|| - sqrt(2/3) * (σ_y + H·α)
        s_norm = np.sqrt(np.sum(s_tr * s_tr))
        yield_stress_current = self.yield_stress + self.hardening_modulus * alpha_old
        f_tr = s_norm - np.sqrt(2.0/3.0) * yield_stress_current
        
        return Fe_tr, be_tr, tau_tr, f_tr
    
    def _return_mapping(
        self,
        F: np.ndarray,
        Fp_old: np.ndarray,
        alpha_old: float
    ) -> Tuple[np.ndarray, float, bool, int]:
        """
        Perform return mapping for J2 plasticity.
        
        Returns
        -------
        Fp_new : np.ndarray
            Updated plastic deformation gradient
        alpha_new : float
            Updated equivalent plastic strain
        converged : bool
            Whether return mapping converged
        iterations : int
            Number of iterations
        """
        dim = F.shape[0]
        
        # Trial state
        Fe_tr, be_tr, tau_tr, f_tr = self._compute_elastic_trial(F, Fp_old, alpha_old)
        
        if f_tr <= self.local_tol:
            # Elastic step
            return Fp_old, alpha_old, True, 0
        
        # Plastic step - return mapping
        s_tr = tau_tr - (1/3) * np.trace(tau_tr) * np.eye(dim)
        s_norm_tr = np.sqrt(np.sum(s_tr * s_tr))
        n_tr = s_tr / s_norm_tr  # Flow direction
        
        # Initial guess for plastic multiplier
        dgamma = f_tr / (2.0 * self.mu + (2.0/3.0) * self.hardening_modulus)
        
        converged = False
        iterations = 0
        
        for iteration in range(self.max_local_iter):
            # Update state
            alpha_new = alpha_old + np.sqrt(2.0/3.0) * dgamma
            
            # Return mapping residual
            yield_stress_new = self.yield_stress + self.hardening_modulus * alpha_new
            s_norm = s_norm_tr - 2.0 * self.mu * dgamma
            
            if s_norm < 0:
                s_norm = 0.0
                
            residual = s_norm - np.sqrt(2.0/3.0) * yield_stress_new
            
            # Check convergence
            if abs(residual) < self.local_tol:
                converged = True
                iterations = iteration + 1
                break
            
            # Newton update
            dresidual_ddgamma = -2.0 * self.mu - (2.0/3.0) * self.hardening_modulus
            dgamma -= residual / dresidual_ddgamma
            
            # Ensure positive plastic multiplier
            dgamma = max(dgamma, 0.0)
        
        if converged:
            # Update plastic deformation gradient
            # F = Fe · Fp  =>  Fp = Fe^{-1} · F
            # For radial return: Fe = exp(Δγ n) · Fe_tr
            N = n_tr  # Flow direction in current configuration
            
            # Exponential map (simplified for small increments)
            exp_dgamma_N = np.eye(dim) + dgamma * N
            Fe_new = exp_dgamma_N @ Fe_tr
            
            # Update Fp
            Fp_new = np.linalg.solve(Fe_new, F)
            
            return Fp_new, alpha_new, True, iterations
        else:
            return Fp_old, alpha_old, False, iterations
    
    def update_state(
        self,
        state: MaterialState,
        F_new: np.ndarray,
        dt: float,
        quad_weights: np.ndarray,
        cell_idx: int
    ) -> Tuple[bool, int]:
        """
        Update plastic state using return mapping for all quadrature points.
        """
        dim = F_new.shape[-1]
        Fp = state.get_state("Fp")
        alpha = state.get_state("alpha")
        
        all_converged = True
        total_iterations = 0
        
        for q in range(state.num_quad_points):
            F_q = F_new[q, :dim, :dim]
            Fp_old = Fp[q, :dim, :dim]
            alpha_old = alpha[q, 0]
            
            Fp_new, alpha_new, converged, iterations = self._return_mapping(
                F_q, Fp_old, alpha_old
            )
            
            if converged:
                Fp[q, :dim, :dim] = Fp_new
                alpha[q, 0] = alpha_new
                total_iterations += iterations
            else:
                all_converged = False
                print(f"    Warning: Return mapping failed at cell {cell_idx}, qp {q}")
                
        return all_converged, total_iterations
    
    def get_quadrature_point_stress(
        self,
        state: MaterialState,
        F: np.ndarray,
        quad_point_idx: int
    ) -> np.ndarray:
        """
        Compute PK1 stress at quadrature point considering plastic deformation.
        """
        dim = F.shape[0]
        
        # Get plastic state
        Fp = state.get_state("Fp")[quad_point_idx, :dim, :dim]
        
        # Elastic deformation gradient
        Fe = F @ np.linalg.inv(Fp)
        
        # Compute stress using elastic material
        # For hyperelastic material, stress depends on Fe
        P = self.elastic_material.get_quadrature_point_stress(
            state, Fe, quad_point_idx
        )
        
        return P


# =============================================================================
# Material Assignment with State Management
# =============================================================================

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
