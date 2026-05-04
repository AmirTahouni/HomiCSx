from __future__ import annotations

from typing import Any

from homicsx.core.material import MaterialAssignment, LinearElasticIsotropic
from .laws import _material_family, _validate_material


def _validate_material_assignment(assignment: MaterialAssignment) -> None:
    """
    Validate a phase-to-material assignment.

    Parameters
    ----------
    assignment:
        MaterialAssignment object to validate.

    Raises
    ------
    ValueError
        If the assignment is empty.
    TypeError
        If a phase id is not an integer or a material is unsupported.
    """
    if not assignment.materials_by_phase:
        raise ValueError("MaterialAssignment cannot be empty.")

    for phase_id, material in assignment.materials_by_phase.items():
        if not isinstance(phase_id, int):
            raise TypeError("All phase ids in MaterialAssignment must be integers.")
        _validate_material(material)


def get_material_for_phase(assignment: MaterialAssignment, phase_id: int) -> object:
    """
    Return the material assigned to a given phase id.

    Parameters
    ----------
    assignment:
        MaterialAssignment object.
    phase_id:
        Phase id to query.

    Returns
    -------
    object
        Concrete material dataclass instance.

    Raises
    ------
    KeyError
        If the phase id is not present in the assignment.
    """
    return assignment.materials_by_phase[int(phase_id)]


def _collect_material_families(assignment: MaterialAssignment) -> dict[int, str]:
    """
    Return the constitutive family used by each phase.

    Parameters
    ----------
    assignment:
        MaterialAssignment object.

    Returns
    -------
    dict[int, str]
        Mapping:
            phase_id -> material family name

    Notes
    -----
    The family names are currently:
        - "linear_elastic"
        - "hyperelastic"
    """
    return {
        int(phase_id): _material_family(material)
        for phase_id, material in assignment.materials_by_phase.items()
    }


def _problem_is_nonlinear(assignment: MaterialAssignment) -> bool:
    """
    Determine whether the global problem must be treated as nonlinear.

    Parameters
    ----------
    assignment:
        MaterialAssignment object.

    Returns
    -------
    bool
        True if at least one phase uses a nonlinear constitutive family.

    Notes
    -----
    Current rule:
    - if all phases are linear elastic -> linear problem
    - if any phase is hyperelastic -> nonlinear problem
    """
    # families = collect_material_families(assignment)
    # return any(family != "linear_elastic" for family in families.values())
    return any(type(mat) != LinearElasticIsotropic for mat in assignment.materials_by_phase.values())


def _build_phase_cell_tag_map(
    assignment: MaterialAssignment,
    physical_tags: Any,
    matrix_phase_id: int = 0,
) -> dict[int, int]:
    """
    Convert assignment phase ids into physical cell tags.

    Parameters
    ----------
    assignment:
        MaterialAssignment object.
    physical_tags:
        PhysicalTags object from the mesh layer.
    matrix_phase_id:
        Phase id used for the matrix.

    Returns
    -------
    dict[int, int]
        Mapping:
            phase_id -> physical cell tag

    Notes
    -----
    This helper uses the mesh-module convention:

    - matrix phase -> physical_tags.matrix
    - inclusion/material phase k -> physical_tags.phase_tag_offset + k
    """
    return {
        int(phase_id): int(physical_tags.cell_tag_for_phase(phase_id, matrix_phase_id))
        for phase_id in assignment.materials_by_phase.keys()
    }


def _build_cell_tag_material_map(
    assignment: MaterialAssignment,
    physical_tags: Any,
    matrix_phase_id: int = 0,
) -> dict[int, object]:
    """
    Build a mapping from physical cell tag to material object.

    Parameters
    ----------
    assignment:
        MaterialAssignment object.
    physical_tags:
        PhysicalTags object from the mesh layer.
    matrix_phase_id:
        Phase id used for the matrix.

    Returns
    -------
    dict[int, object]
        Mapping:
            physical cell tag -> material object

    Notes
    -----
    This helper is useful in the FEM assembly layer, where the actual integration
    happens phase-by-phase over subdomain cell tags rather than over phase ids.
    """
    phase_cell_tag_map = _build_phase_cell_tag_map(
        assignment=assignment,
        physical_tags=physical_tags,
        matrix_phase_id=matrix_phase_id,
    )

    return {
        int(cell_tag): assignment.materials_by_phase[int(phase_id)]
        for phase_id, cell_tag in phase_cell_tag_map.items()
    }


__all__ = [
    # assignment
    # "_validate_material_assignment",
    "get_material_for_phase",
    # "collect_material_families",
    # "_problem_is_nonlinear",
    # "_build_phase_cell_tag_map",
    # "_build_cell_tag_material_map",
]