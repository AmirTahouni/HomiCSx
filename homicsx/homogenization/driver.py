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
    Lightweight driver for linear elastic homogenization.

    This class provides a thin wrapper around the core linear homogenization
    solver. It exists for API symmetry with
    :class:`~homicsx.homogenization.driver.NonlinearHomogenizationDriver` and
    follows the same ``driver = Driver(...); result = driver.run()`` pattern.

    Unlike the nonlinear driver, this class is **stateless**—it performs no
    incremental loading, maintains no material history, and supports no hooks.
    For advanced features use `NonlinearHomogenizationDriver`.

    Parameters
    ----------
    mesh_obj : dolfinx.mesh.Mesh
        The computational RVE mesh.
    cell_tags : dolfinx.mesh.MeshTags
        Cell tags identifying material phases.
    facet_tags : dolfinx.mesh.MeshTags
        Facet tags identifying boundary surfaces.
    assignment : MaterialAssignment
        Linear elastic material assignment for each phase.
    settings : ProblemSettings
        Simulation settings including dimension, element family, degree,
        and PETSc solver options.
    physical_tags : PhysicalTags
        Convention for physical tag IDs on cells and facets.
    domain_size : tuple of float
        RVE domain dimensions (Lx, Ly) in 2D or (Lx, Ly, Lz) in 3D.
    matrix_phase_id : int, default 0
        Phase ID corresponding to the matrix material.
    mode : str, default "complete"
        Homogenization mode:

        - ``"complete"`` : Solve all independent load cases to populate the
          full stiffness matrix.
        - ``"partial"`` : Assume macro-isotropy; solve only the unique
          components and symmetrize the result.

    Examples
    --------
    >>> from homicsx import LinearHomogenizationDriver
    >>> 
    >>> driver = LinearHomogenizationDriver(
    ...     mesh_obj=mesh,
    ...     cell_tags=ct,
    ...     facet_tags=ft,
    ...     assignment=assignment,
    ...     settings=settings,
    ...     physical_tags=physical_tags,
    ...     domain_size=(1.0, 1.0),
    ... )
    >>> result = driver.run()
    >>> print(result.C_hom)

    See Also
    --------
    NonlinearHomogenizationDriver :
        Nonlinear driver.
    HomogenizationResult :
        Result container with stiffness matrix and engineering constants.
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

    This class orchestrates the complete nonlinear homogenization workflow:
    incremental loading with adaptive step size control, material state
    tracking, consistent tangent computation, and configurable post-processing.
    It is the primary entry point for large-deformation, rate-dependent, and
    damage-based RVE analysis.

    The driver's defining feature is its **hook system**—user-defined callback
    functions that are invoked at specific points during the simulation.
    Hooks enable advanced customizations without modifying the core solver:

    - **Damage models** : update material properties based on local strain/stress
    - **Non-local models** : compute gradient-enhanced quantities using the
      full displacement field
    - **ML data export** : extract field data at quadrature points
    - **Adaptive strategies** : modify load step size based on solution quality
    - **Debugging** : log convergence statistics, save intermediate states

    Hooks receive typed dataclass objects (e.g.,
    :class:`~homicsx.homogenization.driver.PostStressData`) that provide
    access to the converged displacement field, homogenized stress, material
    states, and a shared simulation state container.

    Parameters
    ----------
    mesh_obj : dolfinx.mesh.Mesh
        The computational RVE mesh.
    cell_tags : dolfinx.mesh.MeshTags
        Cell tags identifying material phases.
    facet_tags : dolfinx.mesh.MeshTags
        Facet tags identifying boundary surfaces for periodic boundary
        condition application.
    assignment : MaterialAssignment
        Material assignment for all phases. Supports hyperelastic,
        viscoelastic, and custom nonlinear materials.
    settings : ProblemSettings
        Problem settings including dimension, element family, degree,
        kinematic formulation, and PETSc solver options.
    physical_tags : PhysicalTags
        Convention for physical tag IDs on cells and facets.
    domain_size : tuple of float
        RVE domain dimensions ``(Lx, Ly)`` in 2D or ``(Lx, Ly, Lz)`` in 3D.
    matrix_phase_id : int, default 0
        Phase ID corresponding to the matrix material.
    quad_degree : int, default 4
        Quadrature rule degree for numerical integration.
    enable_hooks : bool, default True
        Global switch to enable or disable the hook system. Disabling
        hooks provides a small performance gain when no customization is
        needed.

    Attributes
    ----------
    state : SimulationState
        Shared mutable state container that persists across all hooks.
        Use :meth:`set_parameter` and :meth:`get_parameter` to store and
        retrieve data that must be shared between hooks.

    Notes
    -----
    - **Variational Formulation** : Uses a total potential energy
      formulation with periodic boundary conditions enforced via
      ``dolfinx_mpc.MultiPointConstraint``.
    - **Solver** : Newton-Raphson with configurable PETSc options (SNES).
      Adaptive load stepping automatically adjusts the increment size based
      on convergence behavior.
    - **Material Support** : Hyperelastic, viscoelastic (generalized
      Maxwell), and user-defined nonlinear materials via the abstract
      :class:`NonlinearMaterialModel` base class.
    - **State Management** : Material state variables are tracked at
      quadrature points and deep-copied during tangent computation to
      preserve consistency.

    See Also
    --------
    LinearHomogenizationDriver :
        Lightweight driver for linear elastic homogenization.
    NonlinearHomogenizationResult :
        Result container with stress-strain histories and summaries.

    Examples
    --------
    **Basic usage (no hooks):**

    >>> driver = NonlinearHomogenizationDriver(
    ...     mesh_obj=mesh,
    ...     cell_tags=ct,
    ...     facet_tags=ft,
    ...     assignment=material_assignment,
    ...     settings=settings,
    ...     physical_tags=physical_tags,
    ...     domain_size=(1.0, 1.0),
    ... )
    >>> result = driver.run(max_strain=0.3)

    **With damage hook:**

    >>> def my_damage_hook(data: PostStressData):
    ...     # Compute equivalent strain and update material stiffness
    ...     damage = compute_damage(data.P_avg, data.W_avg)
    ...     data.state.set('current_damage', damage)
    ...     # Modify material properties via data.material_states
    ...
    >>> driver = NonlinearHomogenizationDriver(...)
    >>> driver.add_post_stress_hook(my_damage_hook)
    >>> driver.set_parameter('damage_threshold', 0.15)
    >>> result = driver.run(max_strain=0.5)

    **With adaptive step control:**

    >>> def adaptive_step_hook(data: PreStepData):
    ...     damage = data.state.get('current_damage', 0.0)
    ...     if damage > 0.3:
    ...         data.force_adaptive_step_reduction = True
    ...
    >>> driver.add_pre_step_hook(adaptive_step_hook)
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
        """
        Register a hook that is called once before each load case begins.

        This hook fires after the driver has reset material states for the
        new load case but before any load steps are executed. It is ideal
        for per-load-case initialization tasks.

        Parameters
        ----------
        callback : Callable[[PreLoadCaseData], None]
            A function that accepts a single argument of type
            :class:`PreLoadCaseData`.

        PreLoadCaseData Fields
        ----------------------
        load_name : str
            Name of the load case about to start (e.g., ``"uniaxial_x"``).
        load_tag : str
            Either ``"built_in"`` or ``"custom"``.
        target_load : float
            The final load value for this case.
        dim : int
            Spatial dimension (2 or 3).
        context : NonlinearFluctuationProblemContext
            The problem context (quadrature evaluator, material states, etc.).
        material_assignment : MaterialAssignment
            The material assignment for all phases.
        state : SimulationState
            Shared mutable state container. Values written here (with
            ``load_case`` scope) persist across all hooks within the same
            load case.

        Use Cases
        ---------
        - Create a per-load-case tracking file for logging.
        - Initialize lists or accumulators for statistics.
        - Pre-allocate arrays for field data export.

        Example
        -------
        >>> def init_tracking(data: PreLoadCaseData):
        ...     filename = f"tracking_{data.load_name}.txt"
        ...     with open(filename, 'w') as f:
        ...         f.write("step,load,W_avg,J_avg\\n")
        ...     data.state.set('tracking_file', filename, scope='load_case')
        ...
        >>> driver.add_pre_load_case_hook(init_tracking)
        """
        self._pre_load_case_hooks.append(callback)
    
    def add_post_load_case_hook(self, callback: Callable[[PostLoadCaseData], None]) -> None:
        """
        Register a hook that is called once after each load case completes.

        This hook fires after all load steps for a load case have been
        processed (or after a fatal failure). It receives the complete
        history for the load case and is ideal for finalization tasks.

        Parameters
        ----------
        callback : Callable[[PostLoadCaseData], None]
            A function that accepts a single argument of type
            :class:`PostLoadCaseData`.

        PostLoadCaseData Fields
        -----------------------
        load_name : str
            Name of the completed load case.
        history : dict
            Dictionary containing arrays of load, stress, energy, volume
            ratio, and tangent stiffness over all steps. Keys include
            ``"load"``, ``"P"``, ``"W"``, ``"J"``, and ``"Ceff"``.
        state_history : list or None
            List of material state snapshots if history-dependent materials
            were used.
        total_steps : int
            Number of successfully completed steps.
        total_physical_time : float
            Accumulated physical time (relevant for rate-dependent materials).
        converged : bool
            ``True`` if the load case completed successfully, ``False`` if
            it terminated early due to convergence failure.
        context : NonlinearFluctuationProblemContext
            The problem context.
        state : SimulationState
            Shared mutable state container.

        Use Cases
        ---------
        - Generate per-load-case summary reports.
        - Save final material state for checkpointing.
        - Plot stress-strain curves with custom formatting.
        - Compute aggregate quantities (e.g., total dissipated energy).

        Example
        -------
        >>> def final_summary(data: PostLoadCaseData):
        ...     if data.converged:
        ...         final_W = data.history['W'][-1]
        ...         print(f"{data.load_name}: final energy = {final_W:.4e}")
        ...     else:
        ...         print(f"{data.load_name}: FAILED to converge")
        ...
        >>> driver.add_post_load_case_hook(final_summary)
        """
        self._post_load_case_hooks.append(callback)
    
    def add_pre_step_hook(self, callback: Callable[[PreStepData], None]) -> None:
        """
        Register a hook that is called before each load increment is solved.

        This hook provides an opportunity to inspect the upcoming load step
        and **modify its behavior** through mutable fields on the data object.
        It is the primary mechanism for implementing custom adaptive stepping
        strategies.

        Parameters
        ----------
        callback : Callable[[PreStepData], None]
            A function that accepts a single argument of type
            :class:`PreStepData`.

        PreStepData Fields
        ------------------
        step_idx : int
            Current load step number (1-indexed).
        current_load : float
            The load value at the beginning of this step.
        target_load : float
            The load value to be reached after this step.
        load_increment : float
            The size of this load increment (``target_load - current_load``).
        physical_dt : float
            Physical time increment (nonzero only when ``strain_rate`` is set).
        F_macro_target : np.ndarray (mutable)
            The target macroscopic deformation gradient for this step.
            **Modify this field** to change the loading direction or
            magnitude.
        u_current : dolfinx.fem.Function
            The displacement field at the beginning of this step.
        material_states : dict or None
            Material state variables from the previous converged step.
        context : NonlinearFluctuationProblemContext
            The problem context.
        load_name : str
            Name of the current load case.
        state : SimulationState
            Shared mutable state container.
        skip_tangent : bool (mutable, default False)
            Set to ``True`` to skip tangent stiffness computation for this
            step (saves computational cost when the tangent is not needed).
        force_adaptive_step_reduction : bool (mutable, default False)
            Set to ``True`` to discard the current step size and retry with
            a smaller increment. Useful for capturing rapid changes in
            material response (e.g., damage onset, snap-through).

        Use Cases
        ---------
        - **Damage-aware stepping** : Reduce step size when damage begins to
        accumulate.
        - **Softening detection** : Force smaller steps when material
        stiffness decreases.
        - **Target modification** : Adjust the deformation gradient mid-step
        for custom loading protocols.
        - **Performance optimization** : Skip tangent computation for steps
        where it is not required.

        Example
        -------
        >>> def damage_aware_stepping(data: PreStepData):
        ...     damage = data.state.get('current_damage', 0.0)
        ...     if damage > 0.3:
        ...         # Slow down in the damage regime
        ...         data.force_adaptive_step_reduction = True
        ...     if damage > 0.8:
        ...         # Near failure: skip expensive tangent computation
        ...         data.skip_tangent = True
        ...
        >>> driver.add_pre_step_hook(damage_aware_stepping)

        Notes
        -----
        - When ``force_adaptive_step_reduction`` is set to ``True``, the
        driver discards the current increment, applies the cutback factor,
        and retries. No solve is attempted for the original increment.
        - Modifying ``F_macro_target`` changes only the current step; it
        does not affect the overall loading path.
        - Hook execution order follows registration order. If multiple
        hooks modify the same fields, the last modification wins.
        """
        self._pre_step_hooks.append(callback)
    
    def add_post_convergence_hook(self, callback: Callable[[PostConvergenceData], None]) -> None:
        """
        Register a hook that is called immediately after the Newton-Raphson
        solver converges but **before** stress and energy are computed.

        This hook provides the earliest access to the converged displacement
        field, allowing pre-processing before stress evaluation.

        Parameters
        ----------
        callback : Callable[[PostConvergenceData], None]
            A function that accepts a single argument of type
            :class:`PostConvergenceData`.

        PostConvergenceData Fields
        --------------------------
        step_idx : int
            Current load step number.
        current_load : float
            The load value after convergence.
        F_macro : np.ndarray
            The macroscopic deformation gradient.
        u : dolfinx.fem.Function (mutable)
            The converged displacement field. Modifications to this field
            will affect the subsequent stress calculation.
        newton_iterations : int
            Number of Newton iterations required for convergence.
        residual_norm : float
            Final L2 norm of the residual vector.
        material_states : dict or None (mutable)
            Material state variables. Modifications here persist to the
            stress computation.
        context : NonlinearFluctuationProblemContext
            The problem context.
        load_name : str
            Name of the current load case.
        state : SimulationState
            Shared mutable state container.

        Use Cases
        ---------
        - **Solution quality checks** : Warn or log when iteration counts
        are unusually high.
        - **Field pre-processing** : Compute gradient fields or projections
        needed for non-local constitutive models before stress evaluation.
        - **State initialization** : Set up temporary state variables that
        the stress computation will consume.
        - **Convergence monitoring** : Track residual norm history for
        solver diagnostics.

        Example
        -------
        >>> def check_convergence_quality(data: PostConvergenceData):
        ...     if data.newton_iterations > 10:
        ...         print(f"Warning: Step {data.step_idx} took "
        ...               f"{data.newton_iterations} iterations")
        ...     if data.residual_norm > 1e-6:
        ...         print(f"Warning: Large residual {data.residual_norm:.2e}")
        ...
        >>> driver.add_post_convergence_hook(check_convergence_quality)

        Notes
        -----
        - This hook runs **before** :meth:`add_post_stress_hook`. State
        modifications made here are visible to stress computation and all
        subsequent hooks.
        - The displacement field ``u`` is provided as a full
        ``dolfinx.fem.Function``, enabling operations like gradient
        projection and interpolation.
        """
        self._post_convergence_hooks.append(callback)
    
    def add_post_stress_hook(self, callback: Callable[[PostStressData], None]) -> None:
        """
        Register a hook that is called after homogenized stress and energy
        have been computed for the current load step.

        **This is the primary extension point for most customizations.**
        It provides the richest set of data: the converged displacement
        field, the homogenized stress and energy, the macroscopic
        deformation gradient, and full access to material state variables.

        Parameters
        ----------
        callback : Callable[[PostStressData], None]
            A function that accepts a single argument of type
            :class:`PostStressData`.

        PostStressData Fields
        ---------------------
        step_idx : int
            Current load step number.
        current_load : float
            The load value after convergence.
        F_macro : np.ndarray
            The macroscopic deformation gradient.
        u : dolfinx.fem.Function
            The converged displacement field (full FE function). Use the
            :attr:`domain` property to access the underlying mesh.
        P_avg : np.ndarray
            Volume-averaged first Piola-Kirchhoff stress tensor
            (shape: ``(dim, dim)``).
        W_avg : float
            Volume-averaged strain energy density.
        J_avg : float
            Volume-averaged Jacobian (determinant of deformation gradient).
        material_states : dict or None (mutable)
            Material state variables for all phases and cells. Modifying
            ``state.state_vars['damage']`` on individual state objects
            implements damage accumulation that persists to subsequent steps.
        context : NonlinearFluctuationProblemContext
            The problem context, including the quadrature evaluator for
            computing local deformation gradients.
        load_name : str
            Name of the current load case.
        state : SimulationState
            Shared mutable state container for inter-hook communication.

        Use Cases
        ---------
        - **Damage models** : Compute equivalent strain at quadrature
        points and update damage variables in ``material_states``.
        - **Machine learning data export** : Extract full-field deformation
        gradients and stresses for training constitutive models.
        - **Non-local models** : Use ``u`` to compute spatial gradients of
        field variables and solve auxiliary PDEs (e.g., Helmholtz filter
        for gradient damage).
        - **Material parameter calibration** : Compare computed stresses
        against experimental data.
        - **Real-time monitoring** : Log or visualize stress-strain
        evolution during long simulations.

        Example
        -------
        >>> def damage_update(data: PostStressData):
        ...     # Get local deformation gradient at quadrature points
        ...     F_local = data.context.quad_evaluator \\
        ...         .compute_deformation_gradient_at_quad_points(
        ...             data.u, data.context.F_macro
        ...         )
        ...     # Compute equivalent strain and update damage
        ...     for phase_states in data.material_states.values():
        ...         for state in phase_states.values():
        ...             eq_strain = compute_equiv_strain(state, F_local)
        ...             state.state_vars['damage'] = max(
        ...                 state.state_vars.get('damage', 0.0),
        ...                 damage_law(eq_strain)
        ...             )
        ...
        >>> driver.add_post_stress_hook(damage_update)

        Notes
        -----
        - This hook fires after :meth:`add_post_convergence_hook` and
        before :meth:`add_post_tangent_hook`.
        - Modifications to ``material_states`` **must** respect the state
        variable naming conventions of the active material models.
        - The :class:`~homicsx.homogenization.quadrature.QuadraturePointEvaluator`
        accessible via ``context.quad_evaluator`` provides methods for
        computing deformation gradients at quadrature points, enabling
        local constitutive updates.
        """
        self._post_stress_hooks.append(callback)
    
    def add_post_tangent_hook(self, callback: Callable[[PostTangentData], None]) -> None:
        """
        Register a hook that is called after the consistent tangent stiffness
        tensor has been computed.

        This hook fires only on load steps where tangent computation is
        performed (controlled by the ``tangent_every`` parameter of
        :meth:`run`). It provides access to the full tangent matrix for
        stability analysis and multi-scale coupling.

        Parameters
        ----------
        callback : Callable[[PostTangentData], None]
            A function that accepts a single argument of type
            :class:`PostTangentData`.

        PostTangentData Fields
        ----------------------
        step_idx : int
            Current load step number.
        current_load : float
            The load value after convergence.
        F_macro : np.ndarray
            The macroscopic deformation gradient.
        u : dolfinx.fem.Function
            The converged displacement field.
        P_avg : np.ndarray
            Volume-averaged first Piola-Kirchhoff stress tensor.
        W_avg : float
            Volume-averaged strain energy density.
        J_avg : float
            Volume-averaged Jacobian.
        C_tangent : np.ndarray
            The homogenized consistent tangent stiffness tensor in Voigt
            notation (shape: ``(dim*dim, dim*dim)``).
        material_states : dict or None
            Material state variables.
        context : NonlinearFluctuationProblemContext
            The problem context.
        load_name : str
            Name of the current load case.
        state : SimulationState
            Shared mutable state container.

        Use Cases
        ---------
        - **Material stability analysis** : Compute eigenvalues of the
        tangent matrix; negative eigenvalues indicate loss of ellipticity
        (strain localization).
        - **Acoustic tensor check** : Evaluate the strong ellipticity
        condition for detecting shear band formation.
        - **Multi-scale FE² coupling** : Pass the consistent tangent to a
        macro-scale finite element simulation.
        - **Stiffness degradation tracking** : Monitor the evolution of
        specific tangent components during damage progression.
        - **Bifurcation detection** : Identify critical loads where the
        tangent becomes singular.

        Example
        -------
        >>> def stability_check(data: PostTangentData):
        ...     C = data.C_tangent
        ...     # Extract Voigt matrix for 2D
        ...     C_voigt = np.array([
        ...         [C[0,0], C[0,1], C[0,3]],
        ...         [C[1,0], C[1,1], C[1,3]],
        ...         [C[3,0], C[3,1], C[3,3]],
        ...     ])
        ...     eigenvalues = np.linalg.eigvalsh(C_voigt)
        ...     if np.any(eigenvalues < 0):
        ...         print(f"INSTABILITY at load {data.current_load:.4f}!")
        ...         data.state.set('material_unstable', True)
        ...
        >>> driver.add_post_tangent_hook(stability_check)

        Notes
        -----
        - The tangent is computed via finite difference perturbation of the
        macroscopic deformation gradient, using state-aware solves to
        preserve material history.
        - 2D problems produce a 4*4 tangent (plane strain components).
        Convert to 3*3 Voigt notation for eigenvalue analysis:
        indices [0, 1, 3] correspond to (11, 22, 12).
        - 3D problems produce a 9*9 tangent. Convert to 6*6 Voigt notation.
        """
        self._post_tangent_hooks.append(callback)
    
    def add_step_failure_hook(self, callback: Callable[[StepFailureData], None]) -> None:
        """
        Register a hook that is called when a load step fails to converge.

        This hook provides diagnostic information about the failure and an
        opportunity to save the non-converged state for post-mortem analysis.

        Parameters
        ----------
        callback : Callable[[StepFailureData], None]
            A function that accepts a single argument of type
            :class:`StepFailureData`.

        StepFailureData Fields
        ----------------------
        step_idx : int
            The load step number that failed.
        target_load : float
            The load value that was attempted.
        attempted_increment : float
            The size of the load increment that failed.
        error_message : str
            Description of the failure (e.g., convergence code, physical
            instability).
        convergence_code : int
            PETSc SNES convergence reason code (positive = converged,
            negative = diverged).
        newton_iterations : int
            Number of Newton iterations attempted before failure.
        u_last : dolfinx.fem.Function
            The displacement field at the last Newton iteration (not
            converged).
        F_macro_target : np.ndarray
            The target deformation gradient for the failed step.
        material_states : dict or None
            Material state variables at the last Newton iteration.
        context : NonlinearFluctuationProblemContext
            The problem context.
        load_name : str
            Name of the current load case.
        state : SimulationState
            Shared mutable state container.

        Use Cases
        ---------
        - **Failure logging** : Record load, increment size, and error
        details for debugging.
        - **State snapshot** : Save the non-converged displacement field
        and material states for offline analysis.
        - **Custom recovery** : Implement problem-specific fallback
        strategies (e.g., switching to a more robust solver).
        - **Early termination** : Decide whether to continue with smaller
        steps or abort the load case.

        Example
        -------
        >>> def failure_handler(data: StepFailureData):
        ...     # Log failure details
        ...     with open('failures.log', 'a') as f:
        ...         f.write(f"{data.load_name}, step {data.step_idx}, "
        ...                 f"load={data.target_load:.4f}, "
        ...                 f"code={data.convergence_code}\\n")
        ...     # Save non-converged state for debugging
        ...     np.save(f"failure_{data.step_idx}_u.npy",
        ...             data.u_last.x.array)
        ...
        >>> driver.add_step_failure_hook(failure_handler)

        Notes
        -----
        - After this hook fires, the driver automatically reduces the step
        size and retries (unless the minimum step size has been reached).
        - Multiple failures at the same load level will cause this hook to
        fire multiple times.
        - If the minimum step size is reached, a fatal failure occurs and
        the load case terminates. The :meth:`add_post_load_case_hook`
        will still fire with ``converged=False``.
        """
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
    
    # def set_custom_metadata(self, key: str, value: Any) -> None:
    #     """
    #     Set custom metadata that will be passed to all hook data objects.
        
    #     Parameters
    #     ----------
    #     key : str
    #         Metadata key
    #     value : any
    #         Metadata value
    #     """
    #     self._custom_metadata[key] = value
    
    # def get_custom_metadata(self) -> Dict[str, Any]:
    #     """Get a copy of the custom metadata dictionary."""
    #     return self._custom_metadata.copy()
    
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
    





