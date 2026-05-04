from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import fem
import dolfinx_mpc
import petsc4py.PETSc as PETSc

# ======================================================================================
# Single-field helpers
# ======================================================================================
def build_anchor_bc(
    V,
    dim: int,
    anchor_point: tuple[float, ...] | None = None,
    atol: float = 1e-12,
):
    """
    Build a single-point Dirichlet boundary condition that fixes all displacement
    components at one point.

    Parameters
    ----------
    V:
        Vector-valued displacement function space.
    dim:
        Geometric dimension of the problem.
    anchor_point:
        Coordinates of the point to pin. If omitted, the origin is used.
    atol:
        Absolute tolerance for point matching.

    Returns
    -------
    fem.DirichletBC
        Dirichlet boundary condition fixing all displacement components at the
        chosen anchor point.

    Notes
    -----
    This plays the same role as the point constraint in your current FEM code,
    where the origin is fixed before applying periodic constraints.
    """
    if anchor_point is None:
        anchor_point = tuple(0.0 for _ in range(dim))

    def at_anchor(x):
        checks = [np.isclose(x[i], anchor_point[i], atol=atol) for i in range(dim)]
        return np.logical_and.reduce(checks)

    anchor_dofs = fem.locate_dofs_geometrical(V, at_anchor)
    zero_value = np.zeros(dim, dtype=np.float64)

    return fem.dirichletbc(zero_value, anchor_dofs, V)


def build_periodic_mpc(
    mesh,
    facet_tags,
    V,
    domain_size: tuple[float, ...],
    physical_tags: Any,
    bcs: list[Any] | None = None,
):
    """
    Build periodic multi-point constraints using the package boundary-tag convention.

    Parameters
    ----------
    mesh:
        DOLFINx mesh.
    facet_tags:
        Facet MeshTags object.
    V:
        Vector-valued displacement function space.
    domain_size:
        Domain lengths in each coordinate direction. Must have length equal to
        the mesh dimension.
    physical_tags:
        PhysicalTags object defining the boundary-tag convention.
    bcs:
        Optional list of existing Dirichlet BCs that should be respected while
        constructing the periodic constraints.

    Returns
    -------
    dolfinx_mpc.MultiPointConstraint
        Finalized periodic MPC object.

    Notes
    -----
    Boundary pairing follows the fixed tagging logic:

    2D
        right  -> left
        top    -> bottom

    3D
        right  -> left
        top    -> bottom
        far    -> near

    This mirrors the relation-map logic from your earlier code, while using the
    boundary tags defined by `PhysicalTags`.
    """
    if bcs is None:
        bcs = []

    dim = mesh.topology.dim
    if len(domain_size) != dim:
        raise ValueError(
            f"domain_size must have length {dim}, got {len(domain_size)}."
        )

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    if dim == 2:
        Lx, Ly = domain_size

        def periodic_relation_right_left(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0] - Lx
            out_x[1] = x[1]
            return out_x

        def periodic_relation_top_bottom(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0]
            out_x[1] = x[1] - Ly
            return out_x

        mpc.create_periodic_constraint_topological(
            V,
            facet_tags,
            int(physical_tags.right),
            periodic_relation_right_left,
            bcs,
        )
        mpc.create_periodic_constraint_topological(
            V,
            facet_tags,
            int(physical_tags.top),
            periodic_relation_top_bottom,
            bcs,
        )

    elif dim == 3:
        Lx, Ly, Lz = domain_size

        def periodic_relation_right_left(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0] - Lx
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x

        def periodic_relation_top_bottom(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0]
            out_x[1] = x[1] - Ly
            out_x[2] = x[2]
            return out_x

        def periodic_relation_far_near(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0]
            out_x[1] = x[1]
            out_x[2] = x[2] - Lz
            return out_x

        mpc.create_periodic_constraint_topological(
            V,
            facet_tags,
            int(physical_tags.right),
            periodic_relation_right_left,
            bcs,
        )
        mpc.create_periodic_constraint_topological(
            V,
            facet_tags,
            int(physical_tags.top),
            periodic_relation_top_bottom,
            bcs,
        )
        mpc.create_periodic_constraint_topological(
            V,
            facet_tags,
            int(physical_tags.far),
            periodic_relation_far_near,
            bcs,
        )

    else:
        raise ValueError("Periodic MPC is currently implemented only for dim = 2 or 3.")

    mpc.finalize()
    return mpc


def build_anchor_and_periodic_constraints(
    mesh,
    facet_tags,
    V,
    domain_size: tuple[float, ...],
    physical_tags: Any,
    anchor_point: tuple[float, ...] | None = None,
    atol: float = 1e-12,
) -> tuple[list, dolfinx_mpc.MultiPointConstraint]:
    """
    Convenience helper that builds both the anchor BC and the periodic MPC.

    Parameters
    ----------
    mesh:
        DOLFINx mesh.
    facet_tags:
        Facet MeshTags object.
    V:
        Vector-valued displacement function space.
    domain_size:
        Domain lengths in each coordinate direction.
    physical_tags:
        PhysicalTags object defining the boundary-tag convention.
    anchor_point:
        Coordinates of the point to pin. If omitted, the origin is used.
    atol:
        Absolute tolerance for locating the anchor point.

    Returns
    -------
    tuple[list, dolfinx_mpc.MultiPointConstraint]
        Pair consisting of:
            - list of Dirichlet BCs
            - periodic MPC object

    Notes
    -----
    This helper matches the most common RVE workflow:
    - pin one point
    - apply periodic constraints on opposing boundaries
    """
    dim = mesh.topology.dim
    anchor_bc = build_anchor_bc(
        V=V,
        dim=dim,
        anchor_point=anchor_point,
        atol=atol,
    )
    bcs = [anchor_bc]

    mpc = build_periodic_mpc(
        mesh=mesh,
        facet_tags=facet_tags,
        V=V,
        domain_size=domain_size,
        physical_tags=physical_tags,
        bcs=bcs,
    )

    return bcs, mpc


# ======================================================================================
# Nonlinear helpers
# ======================================================================================
def build_constraints_nonlinear(
    mesh,
    facet_tags,
    V: fem.FunctionSpace,
    domain_size: tuple[float, ...],
    physical_tags: Any,
    atol: float = 1e-12,
) -> tuple[list, dolfinx_mpc.MultiPointConstraint]:
    """
    Constructs periodic constraints and anchor Dirichlet BCs for nonlinear RVE analysis.

    This function sets up Multi-Point Constraints (MPC) for periodic boundary 
    conditions and applies Dirichlet boundary conditions to the corners (anchor points) 
    to prevent rigid body translations during the nonlinear solve.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The computational mesh.
    facet_tags : dolfinx.mesh.MeshTags
        Tags identifying the boundaries (facets) of the mesh.
    V : dolfinx.fem.FunctionSpace
        The function space for the displacement field.
    domain_size : tuple of float
        Dimensions of the RVE (used for periodic mapping).
    physical_tags : Any
        Physical labels from the mesh (e.g., from Gmsh).
    atol : float, default 1e-12
        Absolute tolerance for geometric point location.

    Returns
    -------
    tuple[list, dolfinx_mpc.MultiPointConstraint]
        - bcs: A list of Dirichlet boundary conditions (corner anchors).
        - mpc: The initialized and finalized MultiPointConstraint object.

    Notes
    -----
    - Anchor Strategy: In 2D, all 4 corners are fixed to zero. In 3D, 
      8 corners are anchored using a helper function `build_anchor_bc`, In contrast
      to the linear anchoring strategy where only the node at origin is anchored.
      This strategy has shown to be more effective for nonlinear convergence.
    - Face-only MPC: Periodic relations are applied strictly to the 
      internal parts of the faces, excluding edges and corners to avoid 
      over-constraining the degrees of freedom (DOFs).
    """
    dim = mesh.geometry.dim
    
    if dim==2:
        def bot_left(x):
            return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

        def top_right(x):
            return np.isclose(x[0], 1.0) & np.isclose(x[1], 1.0)

        def top_left(x):
            return np.isclose(x[0], 0.0) & np.isclose(x[1], 1.0)

        def bot_right(x):
            return np.isclose(x[0], 1.0) & np.isclose(x[1], 0.0)

        # Fix displacement at all corners
        dofs_bl = fem.locate_dofs_geometrical(V, bot_left)
        dofs_tr = fem.locate_dofs_geometrical(V, top_right)
        dofs_tl = fem.locate_dofs_geometrical(V, top_left)
        dofs_br = fem.locate_dofs_geometrical(V, bot_right)
        u_zero = fem.Constant(mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
        bcs = [fem.dirichletbc(u_zero, dofs_bl, V),
        fem.dirichletbc(u_zero, dofs_tr, V),
        fem.dirichletbc(u_zero, dofs_tl, V),
        fem.dirichletbc(u_zero, dofs_br, V)]
        # bcs = [fem.dirichletbc(u_zero, dofs_bl, V),]

        # Initialize MultiPointConstraint
        # Link Left <-> Right and Bottom <-> Top
        def periodic_relation_left_right(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - 1.0
            out_x[1] = x[1]
            return out_x

        def periodic_relation_bottom_top(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1] - 1.0
            return out_x

        def right_boundary_locator(x):
            on_right = np.isclose(x[0], 1.0)
            on_top = np.isclose(x[1], 1.0)
            on_left = np.isclose(x[0], 0.0)
            on_bottom = np.isclose(x[1], 0.0)
            return on_right & ~on_top & ~on_bottom

        def top_boundary_locator(x):
            on_right = np.isclose(x[0], 1.0)
            on_top = np.isclose(x[1], 1.0)
            on_left = np.isclose(x[0], 0.0)
            on_bottom = np.isclose(x[1], 0.0)
            return on_top & ~on_left & ~on_right

        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_geometrical(V, right_boundary_locator, periodic_relation_left_right, bcs)
        mpc.create_periodic_constraint_geometrical(V, top_boundary_locator, periodic_relation_bottom_top, bcs)
        mpc.finalize()
    
    elif dim==3:
        corners = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1)
        ]

        bcs = []
        for anchor_point in corners:
            anchor_bc = build_anchor_bc(
                V=V,
                dim=dim,
                anchor_point=anchor_point,
                atol=atol,
            )
            bcs.append(anchor_bc)
        
        def periodic_relation_left_right(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - 1.0
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x
        
        def periodic_relation_bottom_top(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1] - 1.0
            out_x[2] = x[2]
            return out_x
        
        def periodic_relation_near_far(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1]
            out_x[2] = x[2] - 1.0
            return out_x
        
        def right_boundary_locator(x):
            on_right = np.isclose(x[0], 1.0)
            on_left = np.isclose(x[0], 0.0)
            on_top = np.isclose(x[1], 1.0)
            on_bottom = np.isclose(x[1], 0.0)
            on_far = np.isclose(x[2], 1.0)
            on_near = np.isclose(x[2], 0.0)

            return on_right & ~on_top & ~on_bottom & ~on_far & ~on_near

        def top_boundary_locator(x):
            on_right = np.isclose(x[0], 1.0)
            on_left = np.isclose(x[0], 0.0)
            on_top = np.isclose(x[1], 1.0)
            on_bottom = np.isclose(x[1], 0.0)
            on_far = np.isclose(x[2], 1.0)
            on_near = np.isclose(x[2], 0.0)

            return on_top & ~on_right & ~on_left & ~on_far & ~on_near
        
        def far_boundary_locator(x):
            on_right = np.isclose(x[0], 1.0)
            on_left = np.isclose(x[0], 0.0)
            on_top = np.isclose(x[1], 1.0)
            on_bottom = np.isclose(x[1], 0.0)
            on_far = np.isclose(x[2], 1.0)
            on_near = np.isclose(x[2], 0.0)

            return on_far & ~on_right & ~on_left & ~on_top & ~on_bottom
        
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_geometrical(V, right_boundary_locator, periodic_relation_left_right, bcs)
        mpc.create_periodic_constraint_geometrical(V, top_boundary_locator, periodic_relation_bottom_top, bcs)
        mpc.create_periodic_constraint_geometrical(V, far_boundary_locator, periodic_relation_near_far, bcs)
        mpc.finalize()

    return bcs, mpc


# ======================================================================================
# Mixed / subspace helpers
# ======================================================================================
def build_anchor_bc_on_subspace(
    W_sub,
    W_sub_collapsed,
    dim: int,
    anchor_point: tuple[float, ...] | None = None,
    atol: float = 1e-12,
):
    if anchor_point is None:
        anchor_point = tuple(0.0 for _ in range(dim))

    def at_anchor(x):
        checks = [np.isclose(x[i], anchor_point[i], atol=atol) for i in range(dim)]
        return np.logical_and.reduce(checks)

    anchor_dofs = fem.locate_dofs_geometrical(W_sub_collapsed, at_anchor)
    zero_value = np.zeros(dim, dtype=np.float64)

    return fem.dirichletbc(zero_value, anchor_dofs, W_sub_collapsed)


def build_pressure_pin_bc_on_subspace(
    W_sub,
    W_sub_collapsed,
    pressure_pin_point: tuple[float, ...] | None = None,
    atol: float = 1e-12,
):
    """
    Build a single-point Dirichlet condition on a scalar subspace.
    """
    if pressure_pin_point is None:
        # use origin by default
        gdim = W_sub.mesh.geometry.dim
        pressure_pin_point = tuple(0.0 for _ in range(gdim))

    def at_pressure_point(x):
        checks = [np.isclose(x[i], pressure_pin_point[i], atol=atol) for i in range(len(pressure_pin_point))]
        return np.logical_and.reduce(checks)

    pressure_dofs = fem.locate_dofs_geometrical(W_sub_collapsed, at_pressure_point)
    return fem.dirichletbc(0.0, pressure_dofs, W_sub_collapsed)


def build_periodic_mpc_on_subspace(
    mesh,
    facet_tags,
    W,
    constrained_subspace,
    domain_size: tuple[float, ...],
    physical_tags: Any,
    bcs: list[Any] | None = None,
):
    if bcs is None:
        bcs = []

    dim = mesh.topology.dim
    if len(domain_size) != dim:
        raise ValueError(
            f"domain_size must have length {dim}, got {len(domain_size)}."
        )

    mpc = dolfinx_mpc.MultiPointConstraint(W)

    if dim == 2:
        Lx, Ly = domain_size

        def periodic_relation_right_left(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0] - Lx
            out_x[1] = x[1]
            return out_x

        def periodic_relation_top_bottom(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0]
            out_x[1] = x[1] - Ly
            return out_x

        mpc.create_periodic_constraint_topological(
            constrained_subspace, facet_tags, int(physical_tags.right), periodic_relation_right_left, bcs
        )
        mpc.create_periodic_constraint_topological(
            constrained_subspace, facet_tags, int(physical_tags.top), periodic_relation_top_bottom, bcs
        )

    elif dim == 3:
        Lx, Ly, Lz = domain_size

        def periodic_relation_right_left(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0] - Lx
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x

        def periodic_relation_top_bottom(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0]
            out_x[1] = x[1] - Ly
            out_x[2] = x[2]
            return out_x

        def periodic_relation_far_near(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[0]
            out_x[1] = x[1]
            out_x[2] = x[2] - Lz
            return out_x

        mpc.create_periodic_constraint_topological(
            constrained_subspace, facet_tags, int(physical_tags.right), periodic_relation_right_left, bcs
        )
        mpc.create_periodic_constraint_topological(
            constrained_subspace, facet_tags, int(physical_tags.top), periodic_relation_top_bottom, bcs
        )
        mpc.create_periodic_constraint_topological(
            constrained_subspace, facet_tags, int(physical_tags.far), periodic_relation_far_near, bcs
        )

    else:
        raise ValueError("Periodic subspace MPC is currently implemented only for dim = 2 or 3.")

    mpc.finalize()
    return mpc


def build_mixed_anchor_and_periodic_constraints(
    mesh,
    facet_tags,
    W,
    displacement_subspace,
    displacement_subspace_collapsed,
    domain_size: tuple[float, ...],
    physical_tags: Any,
    anchor_point: tuple[float, ...] | None = None,
    atol: float = 1e-12,
):
    dim = mesh.topology.dim

    anchor_bc = build_anchor_bc_on_subspace(
        W_sub=displacement_subspace,
        W_sub_collapsed=displacement_subspace_collapsed,
        dim=dim,
        anchor_point=anchor_point,
        atol=atol,
    )

    bcs = [anchor_bc]

    mpc = build_periodic_mpc_on_subspace(
        mesh=mesh,
        facet_tags=facet_tags,
        W=W,
        constrained_subspace=displacement_subspace,
        domain_size=domain_size,
        physical_tags=physical_tags,
        bcs=bcs,
    )

    return bcs, mpc


# ======================================================================================
# Zero-mean pressure helpers
# ======================================================================================
def zero_mean_scalar_constraint_forms(r, q, pressure, dx_measure):
    """
    Build the additional forms needed to enforce a zero-mean scalar field
    using a Lagrange multiplier.

    Parameters
    ----------
    r:
        Trial scalar multiplier.
    q:
        Test scalar multiplier.
    pressure:
        Pressure field variable.
    dx_measure:
        Domain integration measure, typically `ufl.dx(domain=mesh)`.

    Returns
    -------
    tuple
        Pair of UFL contributions:
            - coupling contribution for the pressure test equation
            - zero-mean contribution for the multiplier test equation

    Notes
    -----
    In a mixed/hybrid formulation with pressure as an independent field, the
    pressure is usually determined only up to an additive constant. The proper
    way to remove that null mode is to enforce:

        Integral(p dx) = 0

    through an additional scalar Lagrange multiplier, rather than pinning one
    pressure DOF at an arbitrary point.
    """
    pressure_coupling = pressure * q * dx_measure
    zero_mean_equation = pressure * r * dx_measure
    return pressure_coupling, zero_mean_equation


__all__ = [
   # constraints
    "build_anchor_bc",
    "build_periodic_mpc",
    "build_anchor_and_periodic_constraints",
    "build_constraints_nonlinear", 
]
