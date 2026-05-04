from __future__ import annotations

import ufl

from .kinematics import (
    small_strain,
    volumetric_strain,
    deviatoric_strain,
    deformation_gradient,
    _embed_2d_tensor_in_3d,
)


def _linear_isotropic_stress(u, lambda_, mu):
    """
    Small-strain isotropic linear-elastic stress tensor.

    Parameters
    ----------
    u:
        Displacement field.
    lambda_:
        First Lamé parameter.
    mu:
        Shear modulus.

    Returns
    -------
    ufl.Expr
        Stress tensor.
    """
    eps = small_strain(u)
    dim = eps.ufl_shape[0]
    return lambda_ * ufl.tr(eps) * ufl.Identity(dim) + 2.0 * mu * eps


def _linear_elastic_energy_density(u, lambda_, mu):
    """
    Small-strain linear-elastic strain energy density.

    Parameters
    ----------
    u:
        Displacement field.
    lambda_:
        First Lamé parameter.
    mu:
        Shear modulus.

    Returns
    -------
    ufl.Expr
        Scalar energy density.
    """
    eps = small_strain(u)
    return 0.5 * lambda_ * ufl.tr(eps) ** 2 + mu * ufl.inner(eps, eps)


def _linear_total_strain(v, Eps):
    """
    Total small-strain tensor for homogenization problems.

    Parameters
    ----------
    v:
        Periodic fluctuation displacement field.
    Eps:
        Constant macroscopic strain tensor.

    Returns
    -------
    ufl.Expr
        Total strain tensor:
            Eps + sym(grad(v))
    """
    return Eps + small_strain(v)


def _linear_homogenization_stress(v, Eps, lambda_, mu):
    """
    Linear-elastic stress tensor based on total strain in a homogenization problem.

    Parameters
    ----------
    v:
        Periodic fluctuation displacement field.
    Eps:
        Constant macroscopic strain tensor.
    lambda_:
        First Lamé parameter.
    mu:
        Shear modulus.

    Returns
    -------
    ufl.Expr
        Stress tensor computed from:
            Eps + sym(grad(v))
    """
    eps_total = _linear_total_strain(v, Eps)
    dim = eps_total.ufl_shape[0]
    return lambda_ * ufl.tr(eps_total) * ufl.Identity(dim) + 2.0 * mu * eps_total


__all__ = [
    # # constitutive
    # "_linear_isotropic_stress",
    # "_linear_elastic_energy_density",
]


