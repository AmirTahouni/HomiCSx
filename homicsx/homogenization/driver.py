from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
import dolfinx
import numpy as np

from homicsx.core.homogenization import (
    LinearHomogenizationResult,
    NonlinearHomogenizationResult,
    AdaptiveSettings,
    PreStepData,
    PreLoadCaseData,
    PostConvergenceData,
    PostStressData,
    PostTangentData,
    PostLoadCaseData,
    StepFailureData,
    SimulationState,
)
from homicsx.fem.fluctuation import build_nonlinear_periodic_fluctuation_problem_with_quadrature
from homicsx.homogenization.linear import _solve_linear_homogenization


class LinearHomogenizationDriver:
    """
    Thin wrapper for API symmetry.
    """
    
    def __init__(
        self,
        mesh_obj,
        cell_tags,
        facet_tags,
        assignment,
        settings,
        physical_tags,
        domain_size,
        matrix_phase_id=0,
        mode='complete'
    ):
        self.mesh_obj = mesh_obj
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.assignment = assignment
        self.settings = settings
        self.physical_tags = physical_tags
        self.domain_size = domain_size
        self.matrix_phase_id = matrix_phase_id
        self.mode = mode
    
    def run(self, mode="complete", petsc_options=None) -> LinearHomogenizationResult:
        """Execute linear homogenization. No hooks, no state."""
        return _solve_linear_homogenization(
            mesh_obj=self.mesh_obj,
            cell_tags=self.cell_tags,
            facet_tags=self.facet_tags,
            assignment=self.assignment,
            settings=self.settings,
            physical_tags=self.physical_tags,
            domain_size=self.domain_size,
            matrix_phase_id=self.matrix_phase_id,
            mode=self.mode,
            petsc_options=self.settings.petsc_options if self.settings.petsc_options else None,
        )


class NonlinearHomogenizationDriver:
    """
    Driver for nonlinear homogenization with extensibility hooks.
    
    This class wraps the core homogenization pipeline and provides callback
    hooks for advanced users who need to inject custom behavior.
    
    Examples
    --------
    >>> # Basic usage
    >>> driver = NonlinearHomogenizationDriver(
    ...     mesh_obj=mesh, cell_tags=cell_tags, facet_tags=facet_tags,
    ...     assignment=assignment, settings=settings, physical_tags=physical_tags,
    ...     domain_size=domain_size
    ... )
    >>> summary, state_histories = driver.run(max_strain=0.3)
    
    >>> # With hooks
    >>> def my_hook(data: PostStressData):
    ...     # Update damage variable
    ...     for phase_states in data.material_states.values():
    ...         for state in phase_states.values():
    ...             state.state_vars['damage'] = compute_damage(state)
    
    >>> driver.add_post_stress_hook(my_hook)
    >>> summary, state_histories = driver.run(max_strain=0.3)
    """
    
    def __init__(
        self,
        mesh_obj: dolfinx.mesh.Mesh,
        cell_tags: dolfinx.mesh.MeshTags,
        facet_tags: dolfinx.mesh.MeshTags,
        assignment: Any,  # MaterialAssignment
        settings: Any,    # ProblemSettings
        physical_tags: Any,
        domain_size: tuple,
        matrix_phase_id: int = 0,
        quad_degree: int = 4,
        enable_hooks: bool = True,
    ):
        """
        Initialize the driver.
        
        Parameters
        ----------
        mesh_obj : dolfinx.mesh.Mesh
            The computational RVE mesh
        cell_tags : dolfinx.mesh.MeshTags
            Cell tags for phase identification
        facet_tags : dolfinx.mesh.MeshTags
            Facet tags for boundary condition application
        assignment : MaterialAssignment
            Material assignment for all phases
        settings : ProblemSettings
            Problem settings including dimension, element degree, etc.
        physical_tags : PhysicalTags
            Physical tags for boundary identification
        domain_size : tuple
            Size of the RVE domain (Lx, Ly, [Lz])
        matrix_phase_id : int
            Phase ID for the matrix material (default: 0)
        quad_degree : int
            Quadrature degree for integration (default: 4)
        enable_hooks : bool
            Enable/disable hook system globally (default: True)
        """
        # Build the problem context
        self.problem, self.context = build_nonlinear_periodic_fluctuation_problem_with_quadrature(
            mesh_obj=mesh_obj,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            assignment=assignment,
            settings=settings,
            physical_tags=physical_tags,
            domain_size=domain_size,
            matrix_phase_id=matrix_phase_id,
            petsc_options=settings.petsc_options,
            quad_degree=quad_degree,
        )
        
        # Store essential data
        self.mesh_obj = mesh_obj
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.physical_tags = physical_tags
        self.assignment = assignment
        self.settings = settings
        self.domain_size = domain_size
        self.matrix_phase_id = matrix_phase_id
        self.quad_degree = quad_degree
        self.dim = self.context.metadata['dim']
        
        # Hook management
        self.enable_hooks = enable_hooks
        self._pre_load_case_hooks: List[Callable[[PreLoadCaseData], None]] = []
        self._post_load_case_hooks: List[Callable[[PostLoadCaseData], None]] = []
        self._pre_step_hooks: List[Callable[[PreStepData], None]] = []
        self._post_convergence_hooks: List[Callable[[PostConvergenceData], None]] = []
        self._post_stress_hooks: List[Callable[[PostStressData], None]] = []
        self._post_tangent_hooks: List[Callable[[PostTangentData], None]] = []
        self._step_failure_hooks: List[Callable[[StepFailureData], None]] = []
        
        # Shared state container
        self.state = SimulationState()
    
    # -------------------------------------------------------------------------
    # Public Hook Registration API
    # -------------------------------------------------------------------------
    def set_parameter(self, key: str, value: Any, scope: str = 'load_case') -> None:
        """Set a simulation parameter."""
        self.state.set(key, value, scope)
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a simulation parameter."""
        return self.state.get(key, default)

    def add_pre_load_case_hook(self, callback: Callable[[PreLoadCaseData], None]) -> None:
        """Register a hook called before starting a new load case."""
        self._pre_load_case_hooks.append(callback)
    
    def add_post_load_case_hook(self, callback: Callable[[PostLoadCaseData], None]) -> None:
        """Register a hook called after completing a load case."""
        self._post_load_case_hooks.append(callback)
    
    def add_pre_step_hook(self, callback: Callable[[PreStepData], None]) -> None:
        """
        Register a hook called before solving each load increment.
        
        Use this to modify target deformation gradient, skip tangent
        computation, or force adaptive step reduction.
        """
        self._pre_step_hooks.append(callback)
    
    def add_post_convergence_hook(self, callback: Callable[[PostConvergenceData], None]) -> None:
        """
        Register a hook called immediately after Newton convergence.
        
        Called before stress/energy computation.
        """
        self._post_convergence_hooks.append(callback)
    
    def add_post_stress_hook(self, callback: Callable[[PostStressData], None]) -> None:
        """
        Register a hook called after computing homogenized stress and energy.
        
        This is the PRIMARY extension point for most customizations:
        - Update damage variables
        - Export field data for ML training
        - Implement non-local models
        """
        self._post_stress_hooks.append(callback)
    
    def add_post_tangent_hook(self, callback: Callable[[PostTangentData], None]) -> None:
        """
        Register a hook called after computing consistent tangent stiffness.
        
        Called only on steps where tangent is computed.
        """
        self._post_tangent_hooks.append(callback)
    
    def add_step_failure_hook(self, callback: Callable[[StepFailureData], None]) -> None:
        """Register a hook called when a load step fails to converge."""
        self._step_failure_hooks.append(callback)
    
    def clear_hooks(self) -> None:
        """Clear all registered hooks."""
        self._pre_load_case_hooks.clear()
        self._post_load_case_hooks.clear()
        self._pre_step_hooks.clear()
        self._post_convergence_hooks.clear()
        self._post_stress_hooks.clear()
        self._post_tangent_hooks.clear()
        self._step_failure_hooks.clear()
    
    def set_custom_metadata(self, key: str, value: Any) -> None:
        """
        Set custom metadata that will be passed to all hook data objects.
        
        Parameters
        ----------
        key : str
            Metadata key
        value : any
            Metadata value
        """
        self._custom_metadata[key] = value
    
    def get_custom_metadata(self) -> Dict[str, Any]:
        """Get a copy of the custom metadata dictionary."""
        return self._custom_metadata.copy()
    
    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    
    def run(
        self,
        tangent_every: int = 1,
        output_prefix: str = "rve",
        max_strain: float = 0.2,
        custom_loads: Optional[Dict[str, Callable]] = None,
        from_built_in_loads: Optional[List[str]] = None,
        adaptive_settings: Optional[AdaptiveSettings] = None,
        xdmf_opt: bool = False,
        csv_opt: bool = False,
        strain_rate: Optional[float] = None,
        plot_summary: bool = True,
        plot_individual: bool = False,
        save_plots: bool = False,
    ) -> NonlinearHomogenizationResult:
        """
        Run homogenization analysis with all registered hooks.
        
        Parameters
        ----------
        tangent_every : int
            Compute tangent stiffness every N steps (default: 1)
        output_prefix : str
            Prefix for output files (default: "rve")
        max_strain : float
            Maximum strain magnitude (default: 0.2)
        custom_loads : dict, optional
            Custom load functions mapping name to callable
        from_built_in_loads : list, optional
            List of built-in load modes to run
        adaptive_settings : AdaptiveSettings, optional
            Parameters for adaptive time-stepping
        xdmf_opt : bool
            Save displacement fields for ParaView (default: False)
        csv_opt : bool
            Save history to CSV files (default: False)
        strain_rate : float, optional
            Strain rate for rate-dependent materials
        plot_summary : bool
            Generate summary plots (default: True)
        plot_individual : bool
            Generate individual plots for each load case (default: False)
        save_plots: bool
            Option to save the generated plots (default: False)
            
        Returns
        -------
        summary : dict
            Summary of homogenization results
        state_histories : dict or None
            State histories for history-dependent materials
        """
        from homicsx.homogenization.nlhelpers import _run_all_load_cases
        from homicsx.homogenization.nlhelpers import (
            _summarize_histories, plot_homogenization_summary, plot_each_load_case
        )
        
        self.state.clear_load_case()
        # Prepare hook data for the core pipeline
        hook_data = None
        if self.enable_hooks:
            hook_data = {
                'pre_load_case': self._pre_load_case_hooks,
                'post_load_case': self._post_load_case_hooks,
                'pre_step': self._pre_step_hooks,
                'post_convergence': self._post_convergence_hooks,
                'post_stress': self._post_stress_hooks,
                'post_tangent': self._post_tangent_hooks,
                'on_step_failure': self._step_failure_hooks,
                'driver': self,
                'state': self.state,
            }
        
        # Run the analysis
        result = _run_all_load_cases(
            context=self.context,
            problem=self.problem,
            domain=self.mesh_obj,
            physical_tags=self.physical_tags,
            material_assignment=self.assignment,
            cell_tags=self.cell_tags,
            dim=self.dim,
            tangent_every=tangent_every,
            output_prefix=output_prefix,
            max_strain=max_strain,
            custom_loads=custom_loads,
            built_in_loading_modes=from_built_in_loads,
            adaptive_settings=adaptive_settings,
            xdmf=xdmf_opt,
            csv=csv_opt,
            strain_rate=strain_rate,
            _hook_data=hook_data,
        )
        
        # Handle return format
        if isinstance(result, tuple):
            all_histories, all_state_histories = result
        else:
            all_histories = result
            all_state_histories = None
        
        # Summarize and plot
        summary = _summarize_histories(all_histories, self.dim)
        
        if plot_summary:
            plot_homogenization_summary(
                summary, self.dim, save=save_plots,
                save_prefix=f"{output_prefix}_summary", show=True
            )
        
        if plot_individual:
            plot_each_load_case(
                summary, self.dim, save=save_plots,
                save_prefix=f"{output_prefix}_individual", show=True
            )
        
        output = NonlinearHomogenizationResult(
            histories=all_histories,
            state_histories=all_state_histories,
            summary=summary,
            metadata={}
        )
        
        return output