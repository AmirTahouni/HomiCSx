from __future__ import annotations

from typing import Any

from homicsx.core.geometry import GeometryInput
from homicsx.core.mesh import PhysicalTags, MeshSettings
from homicsx.core.material import MaterialAssignment
from homicsx.core.fem import ProblemSettings
from homicsx.core.stochastic import EnsembleStudyResult

from homicsx.geometry.universal_generator import (
    patriculate_geometry_generator
)

from homicsx.mesh.gmsh_builder import (
    generate_mesh,
)

from homicsx.homogenization.linear import (
    _solve_linear_homogenization,
)

from homicsx.utils.moduli import (
    extract_effective_moduli_3d,
    extract_effective_moduli_2d_plane_strain,
)

import matplotlib.pyplot as plt
import numpy as np


def sweep_volume_fraction_linear (
        sweep_list: list[float],
        homogenization_mode: str,
        geometry_settings: GeometryInput,
        mesh_settings: MeshSettings,
        physical_tags: PhysicalTags,
        material_assignment: MaterialAssignment,
        fem_settings: ProblemSettings,
) -> EnsembleStudyResult:
    """
    Performs a parametric sweep over different volume fractions to study 
    the evolution of effective elastic properties.

    For each volume fraction in the input list, this function generates a 
    random microstructure, solves the linear elasticity homogenization problem, 
    and extracts effective moduli (e.g., Young's Modulus). It concludes by 
    plotting the normalized stiffness (E_hom / E_matrix) against the volume fraction.

    Parameters
    ----------
    sweep_list : list[float]
        A list of volume fractions (between 0 and 1) to be analyzed.
    homogenization_mode : str
        The scope of computation: 'partial' or 'complete'.
    geometry_settings : GeometryInput
        Base configuration for geometry generation (updated per sweep step).
    mesh_settings : MeshSettings
        Parameters for the meshing engine.
    physical_tags : PhysicalTags
        Database of material and boundary tags.
    material_assignment : MaterialAssignment
        Definition of phase properties (Matrix, Inclusions).
    fem_settings : ProblemSettings
        Solver parameters including 2D formulation types (e.g., plane strain).

    Returns
    -------
    EnsembleStudyResult
        A container containing the detailed results for each volume fraction 
        and metadata such as the stiffness contrast ratio.

    Raises
    ------
    ValueError
        If an invalid homogenization_mode is provided.
    NotImplementedError
        If 2D plane stress formulation is requested.

    Notes
    -----
    The function assumes the matrix phase is at index 0 in `material_assignment` 
    to compute the normalization ratio for the final plot.
    """
    if homogenization_mode not in ['partial', 'complete']:
        raise ValueError(f'homogenization_mode must be either "partial" or "complete". recieved {homogenization_mode}.')

    result_list = []
    C_hom_list = []
    E_hom_list= []
    for vf in sweep_list:
        geometry_settings.volume_fraction = vf

        geom = patriculate_geometry_generator(
            geometry_settings,
        )

        msh, ct, ft = generate_mesh(
            geometry=geom,
            mesh_settings=mesh_settings,
        )

        result = _solve_linear_homogenization(
            domain_size=geometry_settings.domain_size,
            mesh_obj=msh,
            cell_tags=ct,
            facet_tags=ft,
            physical_tags=physical_tags,
            matrix_phase_id=0,
            assignment=material_assignment,
            settings=fem_settings,
            mode=homogenization_mode,
        )
        
        result_list.append(result)
        C_hom = result.C_hom
        C_hom_list.append(C_hom)
        if geometry_settings.dim == 3:
            kappa_hom, mu_hom, E_hom, nu_hom = extract_effective_moduli_3d(C_hom)
        elif geometry_settings.dim == 2 and fem_settings.two_dimensional_formulation == "plane_strain":
            kappa_hom, mu_hom, E_hom, nu_hom = extract_effective_moduli_2d_plane_strain(C_hom)
        elif geometry_settings.dim == 2 and fem_settings.two_dimensional_formulation == "plane_stress":
            raise NotImplementedError('2D plane stress homogenization is not implemented yet.')

        E_hom_list.append(E_hom)
        # print(f'case vf={vf}: done')
    
    E_matrix = material_assignment.materials_by_phase[0].young_modulus

    E_hom_over_E_mat_list = [E_h / E_matrix for E_h in E_hom_list]

    plt.scatter(sweep_list, E_hom_over_E_mat_list, marker='o')
    plt.title('Volume fraction sweep results')
    plt.xlabel('volume fraction')
    plt.ylabel('E_homogenized / E_matrix')
    plt.grid(visible=True, which='both', axis='both')
    plt.show()

    output_study_result = EnsembleStudyResult(
        result_list=result_list,
        metadata={
            "sweep_list": sweep_list,
            # "stiffness_fraction": material_assignment.materials_by_phase[1].young_modulus/material_assignment.materials_by_phase[0].young_modulus
        }
    )

    return output_study_result


def sweep_stiffness_contrast_linear (
        sweep_list: list[float],
        homogenization_mode: str,
        geometry_settings: GeometryInput,
        mesh_settings: MeshSettings,
        physical_tags: PhysicalTags,
        material_assignment: MaterialAssignment,
        fem_settings: ProblemSettings,
) -> EnsembleStudyResult:
    """
    Analyzes the effect of material property contrast on the effective Young's modulus.

    This function performs a sweep over different stiffness ratios (E_particle / E_matrix). 
    In each step, it fixes the matrix modulus to 1.0 and scales the particle modulus 
    according to the contrast value. This is particularly useful for validating 
    the numerical model against analytical bounds.

    Parameters
    ----------
    sweep_list : list[float]
        A list of ratios representing (E_inclusion / E_matrix).
    homogenization_mode : str
        The scope of computation: 'partial' or 'complete'.
    geometry_settings : GeometryInput
        Configuration for the RVE geometry.
    mesh_settings : MeshSettings
        Parameters for mesh generation.
    physical_tags : PhysicalTags
        Mapping of IDs to physical regions.
    material_assignment : MaterialAssignment
        The material database (updated in each iteration with new moduli).
    fem_settings : ProblemSettings
        Solver settings and 2D/3D formulation options.

    Returns
    -------
    EnsembleStudyResult
        A container holding the results for each contrast level and 
        associated simulation metadata.

    Notes
    -----
    The function automatically normalizes the results by setting E_matrix = 1.0, 
    making the output E_homogenized inherently representative of the 
    stiffness enhancement factor.
    """
    if homogenization_mode not in ['partial', 'complete']:
        raise ValueError(f'homogenization_mode must be either "partial" or "complete". recieved {homogenization_mode}.')

    C_hom_list = []
    E_hom_list= []
    result_list = []
    for contrast in sweep_list:
        geom = patriculate_geometry_generator(
            geometry_settings,
        )

        msh, ct, ft = generate_mesh(
            geometry=geom,
            mesh_settings=mesh_settings,
        )

        material_assignment.materials_by_phase[0].young_modulus = 1
        material_assignment.materials_by_phase[1].young_modulus = contrast

        result = _solve_linear_homogenization(
            domain_size=geometry_settings.domain_size,
            mesh_obj=msh,
            cell_tags=ct,
            facet_tags=ft,
            physical_tags=physical_tags,
            matrix_phase_id=0,
            assignment=material_assignment,
            settings=fem_settings,
            mode=homogenization_mode,
        )
        
        result_list.append(result)
        C_hom = result.C_hom
        C_hom_list.append(C_hom)
        if geometry_settings.dim == 3:
            kappa_hom, mu_hom, E_hom, nu_hom = extract_effective_moduli_3d(C_hom)
        elif geometry_settings.dim == 2 and fem_settings.two_dimensional_formulation == "plane_strain":
            kappa_hom, mu_hom, E_hom, nu_hom = extract_effective_moduli_2d_plane_strain(C_hom)
        elif geometry_settings.dim == 2 and fem_settings.two_dimensional_formulation == "plane_stress":
            raise NotImplementedError('2D plane stress homogenization is not implemented yet.')

        E_hom_list.append(E_hom)
        # print(f'case contrast={contrast}: done')

    plt.scatter(sweep_list, E_hom_list, marker='o')
    plt.title('Young modulus contrast sweep results')
    plt.xlabel('E_particle / E_matrix')
    plt.ylabel('E_homogenized / E_matrix')
    plt.grid(visible=True, which='both', axis='both')
    plt.show()

    output_study_result = EnsembleStudyResult(
        result_list=result_list,
        metadata={
            "sweep_list": sweep_list,
            # "volume_fraction": geometry_settings.volume_fraction,
        }
    )

    return output_study_result
    
    
__all__ = [
    # sweep
    "sweep_volume_fraction_linear",
    "sweep_stiffness_contrast_linear",
]










