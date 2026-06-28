from __future__ import annotations

from typing import Any

from homicsx.core.geometry import GeometryInput
from homicsx.core.mesh import PhysicalTags, MeshSettings
from homicsx.core.material import MaterialAssignment
from homicsx.core.fem import ProblemSettings
from homicsx.core.stochastic import EnsembleStudyResult

from homicsx.geometry.universal_generator import (
    particulate_geometry_generator
)

from homicsx.mesh.gmsh_builder import (
    generate_mesh,
)

from homicsx.homogenization.linear import (
    _solve_linear_homogenization,
)


def perform_ensemble_study(
        ensemble_size: int,
        geometry_input: GeometryInput,
        mesh_settings: MeshSettings,
        Physical_tags: PhysicalTags,
        assignment: MaterialAssignment,
        fem_settings: ProblemSettings,
        matrix_phase_id: int = 0,
        homogenization_solver: str = "linear",
        homogenization_mode: str = "complete",
    ) -> EnsembleStudyResult:
    """
    Executes a statistical ensemble study by generating and solving multiple RVEs.

    This function automates the entire pipeline: from random geometry generation 
    and meshing to finite element solving and homogenization. It collects results 
    from multiple realizations to allow for statistical characterization of 
    effective material properties.

    Parameters
    ----------
    ensemble_size : int
        Number of random realizations to generate and solve.
    geometry_input : GeometryInput
        Parameters for the RSA geometry generator (dim, volume fraction, etc.).
    mesh_settings : MeshSettings
        Configuration for the mesh generator (refinement, element types).
    Physical_tags : PhysicalTags
        Identifiers for different regions (phases) within the mesh.
    assignment : MaterialAssignment
        Mapping of physical tags to material constitutive laws and properties.
    fem_settings : ProblemSettings
        Solver settings, including tolerances and boundary condition types.
    matrix_phase_id : int, default 0
        The physical tag ID representing the matrix phase.
    homogenization_solver : str, default "linear"
        The type of solver to use. Currently only "linear" is implemented.
    homogenization_mode : str, default "complete"
        Defines the scope of homogenization (e.g., full stiffness tensor vs. scalar).

    Returns
    -------
    EnsembleStudyResult
        A container holding the list of all individual results and associated 
        metadata for post-processing and statistical averaging.

    Raises
    ------
    NotImplementedError
        If a solver type other than "linear" is requested.

    Example
    -------
    >>> study = perform_ensemble_study(ensemble_size=10, geometry_input=my_input, ...)
    >>> print(f"Average Stiffness: {study.print_summary()}")
    >>> study.visualize_moduli_histogram(num_bins=10)
    """
    hom_result_list = []
    
    if homogenization_solver=="linear":
        for i in range(ensemble_size):
            geometry = particulate_geometry_generator(
                geometry_input
            )

            mesh, ct, ft = generate_mesh(
                geometry=geometry,
                mesh_settings=mesh_settings,
            )

            result = _solve_linear_homogenization(
                mesh_obj=mesh,
                cell_tags=ct,
                facet_tags=ft,
                assignment=assignment,
                settings=fem_settings,
                physical_tags=Physical_tags,
                domain_size=geometry_input.domain_size,
                matrix_phase_id=matrix_phase_id,
                mode=homogenization_mode,
            )

            hom_result_list.append(result)
            # print(f'study {i} of ensemble done')

    else:
        raise NotImplementedError(f'currently, only linear ensemble study is supported. recived {homogenization_solver}.')

    result_container = EnsembleStudyResult(
        result_list=hom_result_list,  
        metadata={
            "volume_fraction": geometry_input.volume_fraction,
        },
    )

    return result_container


__all__ = [
    # ensemble
    "perform_ensemble_study",
]
    

    

