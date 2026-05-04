from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from dolfinx import fem

from homicsx.core.material import (
    LinearElasticIsotropic,
    MaterialAssignment,
    NeoHookeanIsotropic,
)
from .assignment import _build_phase_cell_tag_map, get_material_for_phase
from .laws import _bulk_from_young_poisson, _lame_from_young_poisson


@dataclass
class _LinearElasticCoefficients:
    """
    Cellwise DG0 coefficient fields for isotropic linear elasticity.

    Attributes
    ----------
    young_modulus:
        Piecewise-constant Young's modulus field.
    poisson_ratio:
        Piecewise-constant Poisson's ratio field.
    lambda_:
        Piecewise-constant first Lamé parameter field.
    mu:
        Piecewise-constant shear modulus field.

    Notes
    -----
    These fields are defined on the full mesh. Cells belonging to phases that are
    not included in the linear-elastic subset receive zero values.
    """

    young_modulus: fem.Function
    poisson_ratio: fem.Function
    lambda_: fem.Function
    mu: fem.Function


@dataclass
class _HyperelasticCoefficients:
    """
    Cellwise DG0 coefficient fields for hyperelastic materials.

    Attributes
    ----------
    young_modulus:
        Piecewise-constant Young's modulus field.
    poisson_ratio:
        Piecewise-constant Poisson's ratio field.
    lambda_:
        Piecewise-constant first Lamé parameter field.
    mu:
        Piecewise-constant shear modulus field.
    bulk_modulus:
        Piecewise-constant bulk modulus field.

    Notes
    -----
    The current hyperelastic implementation supports Neo-Hookean materials, but the
    container is named more generally so the public API can stay stable if more
    hyperelastic submodels are added later.
    """

    young_modulus: fem.Function
    poisson_ratio: fem.Function
    lambda_: fem.Function
    mu: fem.Function
    bulk_modulus: fem.Function


def _create_piecewise_constant_field(
    mesh,
    cell_tags,
    values_by_cell_tag: dict[int, float],
    name: str,
) -> fem.Function:
    """
    Create a DG0 scalar field from physical cell-tag values.

    Parameters
    ----------
    mesh:
        DOLFINx mesh.
    cell_tags:
        Cell MeshTags object.
    values_by_cell_tag:
        Mapping from physical cell tag to scalar value.
    name:
        Name assigned to the created function.

    Returns
    -------
    fem.Function
        DG0 scalar function with cellwise values.

    Notes
    -----
    This helper follows the same general idea as your earlier material assignment
    workflow, where material parameters such as E and nu are stored as piecewise-
    constant fields over tagged cells.
    """
    V0 = fem.functionspace(mesh, ("DG", 0))
    field = fem.Function(V0, name=name)
    field.x.array[:] = 0.0

    for cell_tag, value in values_by_cell_tag.items():
        cells = cell_tags.find(int(cell_tag))
        field.x.array[cells] = float(value)

    field.x.scatter_forward()
    return field


def _values_by_tag_for_linear_materials(
    assignment: MaterialAssignment,
    physical_tags: Any,
    matrix_phase_id: int,
    phases: Iterable[int] | None = None,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """
    Build scalar value dictionaries keyed by physical cell tag for linear-elastic phases.

    Returns
    -------
    tuple
        Four dictionaries:
            - E by cell tag
            - nu by cell tag
            - lambda by cell tag
            - mu by cell tag
    """
    phase_cell_tag_map = _build_phase_cell_tag_map(
        assignment=assignment,
        physical_tags=physical_tags,
        matrix_phase_id=matrix_phase_id,
    )

    selected_phases = (
        set(int(phase_id) for phase_id in phases)
        if phases is not None
        else set(phase_cell_tag_map.keys())
    )

    E_by_tag: dict[int, float] = {}
    nu_by_tag: dict[int, float] = {}
    lambda_by_tag: dict[int, float] = {}
    mu_by_tag: dict[int, float] = {}

    for phase_id, cell_tag in phase_cell_tag_map.items():
        if phase_id not in selected_phases:
            continue

        material = get_material_for_phase(assignment, phase_id)
        if not isinstance(material, LinearElasticIsotropic):
            continue

        lmbda, mu = _lame_from_young_poisson(
            material.young_modulus,
            material.poisson_ratio,
        )

        E_by_tag[cell_tag] = material.young_modulus
        nu_by_tag[cell_tag] = material.poisson_ratio
        lambda_by_tag[cell_tag] = lmbda
        mu_by_tag[cell_tag] = mu

    return E_by_tag, nu_by_tag, lambda_by_tag, mu_by_tag


def _values_by_tag_for_hyperelastic_materials(
    assignment: MaterialAssignment,
    physical_tags: Any,
    matrix_phase_id: int,
    phases: Iterable[int] | None = None,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
]:
    """
    Build scalar value dictionaries keyed by physical cell tag for hyperelastic phases.

    Returns
    -------
    tuple
        Five dictionaries:
            - E by cell tag
            - nu by cell tag
            - lambda by cell tag
            - mu by cell tag
            - bulk modulus by cell tag
    """
    phase_cell_tag_map = _build_phase_cell_tag_map(
        assignment=assignment,
        physical_tags=physical_tags,
        matrix_phase_id=matrix_phase_id,
    )

    selected_phases = (
        set(int(phase_id) for phase_id in phases)
        if phases is not None
        else set(phase_cell_tag_map.keys())
    )

    E_by_tag: dict[int, float] = {}
    nu_by_tag: dict[int, float] = {}
    lambda_by_tag: dict[int, float] = {}
    mu_by_tag: dict[int, float] = {}
    bulk_by_tag: dict[int, float] = {}

    for phase_id, cell_tag in phase_cell_tag_map.items():
        if phase_id not in selected_phases:
            continue

        material = get_material_for_phase(assignment, phase_id)
        if not isinstance(material, NeoHookeanIsotropic):
            continue

        lmbda, mu = _lame_from_young_poisson(
            material.young_modulus,
            material.poisson_ratio,
        )
        bulk = _bulk_from_young_poisson(
            material.young_modulus,
            material.poisson_ratio,
        )

        E_by_tag[cell_tag] = material.young_modulus
        nu_by_tag[cell_tag] = material.poisson_ratio
        lambda_by_tag[cell_tag] = lmbda
        mu_by_tag[cell_tag] = mu
        bulk_by_tag[cell_tag] = bulk

    return E_by_tag, nu_by_tag, lambda_by_tag, mu_by_tag, bulk_by_tag


def _build_linear_elastic_coefficients(
    mesh,
    cell_tags,
    assignment: MaterialAssignment,
    physical_tags: Any,
    matrix_phase_id: int = 0,
    phases: Iterable[int] | None = None,
) -> _LinearElasticCoefficients:
    """
    Build DG0 material fields for the linear-elastic phases in an assignment.

    Parameters
    ----------
    mesh:
        DOLFINx mesh.
    cell_tags:
        Cell MeshTags object.
    assignment:
        Material assignment.
    physical_tags:
        PhysicalTags object from the mesh module.
    matrix_phase_id:
        Phase id used for the matrix.
    phases:
        Optional subset of phase ids to include. If omitted, all linear-elastic
        phases in the assignment are included.

    Returns
    -------
    LinearElasticCoefficients
        Bundle of DG0 coefficient fields.
    """
    E_by_tag, nu_by_tag, lambda_by_tag, mu_by_tag = _values_by_tag_for_linear_materials(
        assignment=assignment,
        physical_tags=physical_tags,
        matrix_phase_id=matrix_phase_id,
        phases=phases,
    )

    return _LinearElasticCoefficients(
        young_modulus=_create_piecewise_constant_field(
            mesh, cell_tags, E_by_tag, name="YoungModulus_linear"
        ),
        poisson_ratio=_create_piecewise_constant_field(
            mesh, cell_tags, nu_by_tag, name="PoissonRatio_linear"
        ),
        lambda_=_create_piecewise_constant_field(
            mesh, cell_tags, lambda_by_tag, name="Lambda_linear"
        ),
        mu=_create_piecewise_constant_field(
            mesh, cell_tags, mu_by_tag, name="Mu_linear"
        ),
    )


def _build_hyperelastic_coefficients(
    mesh,
    cell_tags,
    assignment: MaterialAssignment,
    physical_tags: Any,
    matrix_phase_id: int = 0,
    phases: Iterable[int] | None = None,
) -> _HyperelasticCoefficients:
    """
    Build DG0 material fields for the hyperelastic phases in an assignment.

    Parameters
    ----------
    mesh:
        DOLFINx mesh.
    cell_tags:
        Cell MeshTags object.
    assignment:
        Material assignment.
    physical_tags:
        PhysicalTags object from the mesh module.
    matrix_phase_id:
        Phase id used for the matrix.
    phases:
        Optional subset of phase ids to include. If omitted, all hyperelastic
        phases in the assignment are included.

    Returns
    -------
    HyperelasticCoefficients
        Bundle of DG0 coefficient fields.

    Notes
    -----
    The current implementation supports Neo-Hookean materials, but the function
    is intentionally named more generally so that additional hyperelastic models
    can later be added without changing the public API.
    """
    E_by_tag, nu_by_tag, lambda_by_tag, mu_by_tag, bulk_by_tag = (
        _values_by_tag_for_hyperelastic_materials(
            assignment=assignment,
            physical_tags=physical_tags,
            matrix_phase_id=matrix_phase_id,
            phases=phases,
        )
    )

    return _HyperelasticCoefficients(
        young_modulus=_create_piecewise_constant_field(
            mesh, cell_tags, E_by_tag, name="YoungModulus_hyper"
        ),
        poisson_ratio=_create_piecewise_constant_field(
            mesh, cell_tags, nu_by_tag, name="PoissonRatio_hyper"
        ),
        lambda_=_create_piecewise_constant_field(
            mesh, cell_tags, lambda_by_tag, name="Lambda_hyper"
        ),
        mu=_create_piecewise_constant_field(
            mesh, cell_tags, mu_by_tag, name="Mu_hyper"
        ),
        bulk_modulus=_create_piecewise_constant_field(
            mesh, cell_tags, bulk_by_tag, name="BulkModulus_hyper"
        ),
    )

__all__ = [
]