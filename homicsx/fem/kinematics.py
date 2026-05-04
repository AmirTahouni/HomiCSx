from __future__ import annotations

import ufl


def small_strain(u):
    """
    Return the infinitesimal strain tensor sym(grad(u)).

    Parameters
    ----------
    u:
        Displacement field.

    Returns
    -------
    ufl.Expr
        Symmetric small-strain tensor.
    """
    return ufl.sym(ufl.grad(u))


def volumetric_strain(u):
    """
    Return the volumetric small-strain measure div(u).

    Parameters
    ----------
    u:
        Displacement field.

    Returns
    -------
    ufl.Expr
        Scalar volumetric strain.
    """
    return ufl.div(u)


def deviatoric_strain(u):
    """
    Return the deviatoric part of the small-strain tensor.

    Parameters
    ----------
    u:
        Displacement field.

    Returns
    -------
    ufl.Expr
        Deviatoric small-strain tensor.

    Notes
    -----
    This helper is useful for mixed and nearly incompressible formulations,
    where volumetric and deviatoric parts are often treated separately.
    """
    eps = small_strain(u)
    dim = eps.ufl_shape[0]
    return eps - (ufl.tr(eps) / dim) * ufl.Identity(dim)


def deformation_gradient(u, dim: int):
    """
    Return the deformation gradient F = I + grad(u).

    Parameters
    ----------
    u:
        Displacement field.
    dim:
        Geometric dimension, either 2 or 3.

    Returns
    -------
    ufl.Expr
        Deformation gradient tensor.
    """
    return ufl.Identity(dim) + ufl.grad(u)


def _embed_2d_tensor_in_3d(A2):
    """
    Embed a 2x2 tensor into a 3x3 tensor for plane-strain-style use.

    Parameters
    ----------
    A2:
        2x2 UFL tensor expression.

    Returns
    -------
    ufl.Expr
        3x3 tensor with the in-plane block from A2 and an out-of-plane
        unit entry in the third normal component.

    Notes
    -----
    This helper is mainly used for 2D hyperelastic formulations, where we
    want a simple plane-strain interpretation by embedding the in-plane
    deformation gradient into 3D with F33 = 1.
    """
    return ufl.as_matrix(
        [
            [A2[0, 0], A2[0, 1], 0.0],
            [A2[1, 0], A2[1, 1], 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def right_cauchy_green(F):
    """
    Return the right Cauchy-Green tensor C = F.T * F.

    Parameters
    ----------
    F:
        Deformation gradient tensor.

    Returns
    -------
    ufl.Expr
        Right Cauchy-Green tensor.
    """
    return F.T * F


def green_lagrange_strain(F):
    """
    Return the Green-Lagrange strain tensor E = 0.5 * (C - I).

    Parameters
    ----------
    F:
        Deformation gradient tensor.

    Returns
    -------
    ufl.Expr
        Green-Lagrange strain tensor.
    """
    C = right_cauchy_green(F)
    return 0.5 * (C - ufl.Identity(C.ufl_shape[0]))


__all__ = [
    # kinematics
    "small_strain",
    "volumetric_strain",
    "deviatoric_strain",
    "deformation_gradient",
    "right_cauchy_green",
    "green_lagrange_strain",
]
