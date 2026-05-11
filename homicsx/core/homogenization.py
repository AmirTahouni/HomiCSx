from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import dolfinx

@dataclass
class LinearHomogenizationResult:
    """
    Result of a linear homogenization run.

    Attributes
    ----------
    C_hom: np.ndarray
        Homogenized stiffness matrix in Voigt notation.
    load_cases: list[np.ndarray]
        List of macroscopic strain tensors used in the solve loop.
    average_stresses: list[np.ndarray]
        List of volume-averaged stress vectors, one per load case.
    fluctuation_fields: list[any]
        Solved periodic fluctuation fields, one per load case.
    metadata: dict[str, any]
        Optional bookkeeping dictionary.
    """
    C_hom: np.ndarray
    load_cases: list[np.ndarray]
    average_stresses: list[np.ndarray]
    fluctuation_fields: list[Any]
    metadata: dict[str, Any]


@dataclass
class NonlinearHomogenizationResult:
    """
    Result of a nonlinear homogenization run.
    
    Attributes
    ----------
    histories : Dict[str, Dict]
        History dictionaries for each load case.
        Keys are load case names, values are dicts with 'step', 'load_param', 'load_type', 'Fbar', 'Pbar', 'Wbar', 'Jbar', 'converged', 'iters', and 'Ceff'.
    state_histories : Dict[str, List] or None
        Material state histories for each load case.
    summary : Dict
        Summary statistics across all load cases.
    metadata : Dict
        Bookkeeping dictionary.
    """
    histories: Dict[str, Dict]
    state_histories: Optional[Dict[str, List]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveSettings:
    """
    Configuration for adaptive time-stepping in nonlinear solvers.

    This class defines the parameters used to dynamically adjust the increment 
    size based on the convergence behavior of the Newton-Raphson iterations.

    Attributes
    ----------
    initial_step_ratio : float, default 0.025
        The fraction of the total simulation time for the first increment.
    min_step : float, default 1e-6
        The absolute minimum allowed step size before the solver terminates.
    max_step_ratio : float, default 0.2
        The maximum allowed step size as a fraction of total time. 
        Prevents the solver from taking too large jumps that miss physics.
    target_iters_min : int, default 4
        Lower bound for Newton iterations. If converged in fewer steps, 
        the next increment will be increased.
    target_iters_max : int, default 8
        Upper bound for Newton iterations. If convergence takes more steps, 
        the next increment will be reduced.
    growth_factor : float, default 1.5
        Multiplier used to increase the step size when convergence is fast.
    cutback_factor : float, default 0.5
        Multiplier used to reduce the step size when convergence is slow 
        or fails (cut-back).
    """
    initial_step_ratio: float = 0.025
    min_step: float = 1e-6
    max_step_ratio: float = 0.2
    target_iters_min: int = 4
    target_iters_max: int = 8
    growth_factor: float = 1.5
    cutback_factor: float = 0.5


@dataclass
class PreStepData:
    """
    Data passed to pre-step hooks before each load increment solve.
    
    Access to pre-step passed data.
    """
    step_idx: int
    current_load: float
    target_load: float
    load_increment: float
    physical_dt: float
    F_macro_target: np.ndarray
    u_current: dolfinx.fem.Function
    material_states: Optional[Dict]
    context: Any
    load_name: str
    state: 'SimulationState'
    force_adaptive_step_reduction: bool = False


@dataclass
class PostConvergenceData:
    """
    Data passed to post-convergence hooks after successful Newton solve.
    
    Called before stress/energy computation.
    """
    step_idx: int
    current_load: float
    F_macro: np.ndarray
    u: dolfinx.fem.Function
    newton_iterations: int
    residual_norm: float
    material_states: Optional[Dict]
    context: Any
    load_name: str
    state: 'SimulationState'


@dataclass
class PostStressData:
    """
    Data passed to post-stress hooks after computing homogenized stress/energy.
    
    This is the PRIMARY extension point for most customizations.
    """
    step_idx: int
    current_load: float
    F_macro: np.ndarray
    u: dolfinx.fem.Function
    P_avg: np.ndarray
    W_avg: float
    J_avg: float
    material_states: Optional[Dict]
    context: Any
    load_name: str
    state: 'SimulationState'
    @property
    def domain(self):
        return self.u.function_space.mesh


@dataclass
class PostTangentData:
    """
    Data passed to post-tangent hooks after computing consistent tangent.
    
    Only called on steps where tangent_every triggers computation.
    """
    step_idx: int
    current_load: float
    F_macro: np.ndarray
    u: dolfinx.fem.Function
    P_avg: np.ndarray
    W_avg: float
    J_avg: float
    C_tangent: np.ndarray
    material_states: Optional[Dict]
    context: Any
    load_name: str
    state: 'SimulationState'


@dataclass
class PreLoadCaseData:
    """
    Data passed before starting a new load case.
    """
    load_name: str
    load_tag: str
    target_load: float
    dim: int
    context: Any
    material_assignment: Any
    state: 'SimulationState'


@dataclass
class PostLoadCaseData:
    """
    Data passed after completing a load case.
    """
    load_name: str
    history: Dict
    state_history: Optional[List]
    total_steps: int
    total_physical_time: float
    converged: bool
    context: Any
    state: 'SimulationState'


@dataclass
class StepFailureData:
    """
    Data passed when a load step fails to converge.
    """
    step_idx: int
    target_load: float
    attempted_increment: float
    error_message: str
    convergence_code: int
    newton_iterations: int
    u_last: dolfinx.fem.Function
    F_macro_target: np.ndarray
    material_states: Optional[Dict]
    context: Any
    load_name: str
    state: 'SimulationState'


@dataclass
class SimulationState:
    """
    Shared mutable state container that persists across all hooks.
    
    This container is passed by reference to all hook data objects,
    so modifications in one hook are visible to all subsequent hooks.
    
    Examples
    --------
    >>> state = SimulationState()
    >>> state.set('damage_threshold', 0.15)
    >>> state.set('current_damage', 0.0)
    >>> 
    >>> # In hook 1
    >>> state.set('current_damage', 0.3)
    >>> 
    >>> # In hook 2 (called later)
    >>> damage = state.get('current_damage')  # Returns 0.3
    """
    
    # Persistent data that survives across all hooks and load cases
    _persistent: Dict[str, Any] = field(default_factory=dict)
    
    # Per-load-case data (cleared automatically on new load case)
    _per_load_case: Dict[str, Any] = field(default_factory=dict)
    
    # Per-step data (accumulates history)
    _history: Dict[str, List[Any]] = field(default_factory=dict)
    
    def set(self, key: str, value: Any, scope: str = 'load_case') -> None:
        """
        Store a value in the shared state.
        
        Parameters
        ----------
        key : str
            Key to store under
        value : Any
            Value to store
        scope : str
            'persistent' - survives entire simulation
            'load_case' - cleared when new load case starts
            'step' - only valid for current step (use with caution)
        """
        if scope == 'persistent':
            self._persistent[key] = value
        elif scope == 'load_case':
            self._per_load_case[key] = value
        else:
            # Step-scoped values just stored in load_case with _step_ prefix
            self._per_load_case[f"_step_{key}"] = value
    
    def get(self, key: str, default: Any = None, scope: str = 'load_case') -> Any:
        """
        Retrieve a value from the shared state.
        
        Checks per-load-case first, then persistent.
        """
        if scope == 'persistent':
            return self._persistent.get(key, default)
        else:
            # Check load_case first, then persistent as fallback
            if key in self._per_load_case:
                return self._per_load_case[key]
            return self._persistent.get(key, default)
    
    def append_history(self, key: str, value: Any) -> None:
        """Append a value to a history list."""
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(value)
    
    def get_history(self, key: str) -> List[Any]:
        """Get a history list."""
        return self._history.get(key, [])
    
    def clear_load_case(self) -> None:
        """Clear per-load-case data (called before new load case)."""
        self._per_load_case.clear()
        # History persists across load cases unless explicitly cleared
    
    def clear_history(self, key: Optional[str] = None) -> None:
        """Clear history data."""
        if key is None:
            self._history.clear()
        elif key in self._history:
            del self._history[key]
    
    def update(self, other: Dict[str, Any], scope: str = 'load_case') -> None:
        """Update multiple values at once."""
        target = self._persistent if scope == 'persistent' else self._per_load_case
        target.update(other)
    
    def snapshot(self) -> Dict[str, Any]:
        """Return a combined snapshot of all state."""
        return {
            'persistent': self._persistent.copy(),
            'load_case': self._per_load_case.copy(),
            'history': {k: v.copy() for k, v in self._history.items()}
        }

__all__ = [
    "LinearHomogenizationResult",
    "NonlinearHomogenizationResult",
    "AdaptiveSettings",
    "SimulationState",
    "PreLoadCaseData",
    "PreStepData",
    "PostConvergenceData",
    "PostStressData",
    "PostTangentData",
    "PostLoadCaseData",
    "StepFailureData",
]  