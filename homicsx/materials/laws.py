from __future__ import annotations

from homicsx.core.material import LinearElasticIsotropic, NeoHookeanIsotropic


# -----------------------------------------------------------------------------#
# Basic parameter validation
# -----------------------------------------------------------------------------#
def _validate_young_poisson(young_modulus: float, poisson_ratio: float) -> None:
    """
    Validate isotropic material parameters defined by (E, nu).

    Parameters
    ----------
    young_modulus:
        Young's modulus E. Must be strictly positive.
    poisson_ratio:
        Poisson's ratio nu. Must satisfy -1 < nu < 0.5 for compressible models.

    Raises
    ------
    ValueError
        If parameters are outside valid range.
    """
    if young_modulus <= 0.0:
        raise ValueError("Young's modulus must be > 0.")

    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("Poisson's ratio must satisfy -1 < nu < 0.5.")


# -----------------------------------------------------------------------------#
# Parameter conversions (kept similar to your existing code)
# -----------------------------------------------------------------------------#
def _lame_from_young_poisson(
    young_modulus: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """
    Compute Lamé parameters (lambda, mu) from (E, nu).

    This follows the same formulas used in your current FEM code.

    Returns
    -------
    (lambda, mu)
    """
    _validate_young_poisson(young_modulus, poisson_ratio)

    lmbda = young_modulus * poisson_ratio / (1.0 + poisson_ratio) / (1.0 - 2.0 * poisson_ratio)
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))

    return lmbda, mu


def _bulk_from_young_poisson(
    young_modulus: float,
    poisson_ratio: float,
) -> float:
    """
    Compute bulk modulus K from (E, nu).
    """
    _validate_young_poisson(young_modulus, poisson_ratio)
    return young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))


# -----------------------------------------------------------------------------#
# Material inspection helpers
# -----------------------------------------------------------------------------#
def _material_family(material: object) -> str:
    """
    Identify the constitutive family of a material.

    Returns
    -------
    str
        One of:
            - "linear_elastic"
            - "hyperelastic"
    """
    if isinstance(material, LinearElasticIsotropic):
        return "linear_elastic"

    if isinstance(material, NeoHookeanIsotropic):
        return "hyperelastic"

    print(f"WARNING: nsupported/custom material type: {type(material)}")


def _validate_material(material: object) -> None:
    """
    Validate a material instance.

    Parameters
    ----------
    material:
        Material dataclass instance.

    Raises
    ------
    TypeError
        If unsupported material type.
    ValueError
        If parameters are invalid.
    """
    if isinstance(material, LinearElasticIsotropic):
        _validate_young_poisson(material.young_modulus, material.poisson_ratio)
        return

    if isinstance(material, NeoHookeanIsotropic):
        _validate_young_poisson(material.young_modulus, material.poisson_ratio)
        return
    
__all__ = [
]    