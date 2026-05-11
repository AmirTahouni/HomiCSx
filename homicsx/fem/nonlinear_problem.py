from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial
import functools

from petsc4py import PETSc

import dolfinx.fem.petsc
import ufl
from dolfinx import fem as _fem
from dolfinx.fem.petsc import create_vector

import dolfinx_mpc
from dolfinx_mpc.cpp import mpc as _cpp_mpc

from dolfinx_mpc import (
    MultiPointConstraint,
    assemble_matrix,
    assemble_matrix_nest,
    create_matrix_nest,
    assemble_vector,
    assemble_vector_nest,
    create_vector_nest,
    apply_lifting,
)


def assemble_jacobian_mpc(
    u: Sequence[_fem.Function] | _fem.Function,
    jacobian: _fem.Form | Sequence[Sequence[_fem.Form]],
    preconditioner: _fem.Form | Sequence[Sequence[_fem.Form]] | None,
    bcs: Iterable[_fem.DirichletBC],
    mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
    _snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    J: PETSc.Mat,  # type: ignore
    P: PETSc.Mat,  # type: ignore
):
    """
    Assemble the Jacobian matrix and pre-conditioner.

    A function conforming to the interface expected by SNES.setJacobian can
    be created by fixing the first four arguments:

        functools.partial(assemble_jacobian, u, jacobian, pre-conditioner,
                          bcs)

    Args:
        u: Function tied to the solution vector within the residual and
            jacobian
        jacobian: Form of the Jacobian
        pre-conditioner: Form of the pre-conditioner
        bcs: List of Dirichlet boundary conditions
        mpc: The multi point constraint or a sequence of multi point
        _snes: The solver instance
        x: The vector containing the point to evaluate at
        J: Matrix to assemble the Jacobian into
        P: Matrix to assemble the pre-conditioner into
    """
    x.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    # dolfinx.fem.petsc.assign(x, u)
    u.x.petsc_vec.array = x.array_r

    if isinstance(u, Sequence):
        assert isinstance(mpc, Sequence)
        for i in range(len(u)):
            mpc[i].homogenize(u[i])
            mpc[i].backsubstitution(u[i])
    else:
        assert isinstance(u, _fem.Function)
        assert isinstance(mpc, MultiPointConstraint)
        mpc.homogenize(u)
        mpc.backsubstitution(u)

    J.zeroEntries()
    if J.getType() == "nest":
        dolfinx_mpc.assemble_matrix_nest(J, jacobian, mpc, bcs, diagval=1.0)
    else:
        dolfinx_mpc.assemble_matrix(jacobian, mpc, bcs, A=J, diagval=1.0)
    J.assemble()

    if preconditioner is not None:
        P.zeroEntries()
        if P.getType() == "nest":
            dolfinx_mpc.assemble_matrix_nest(P, preconditioner, mpc, bcs, diagval=1.0)
        else:
            dolfinx_mpc.assemble_matrix(preconditioner, mpc, bcs, A=P, diagval=1.0)
        P.assemble()


def assemble_residual_mpc(
    u: _fem.Function | Sequence[_fem.Function],
    residual: _fem.Form | Sequence[_fem.Form],
    jacobian: _fem.Form | Sequence[Sequence[_fem.Form]],
    bcs: Sequence[_fem.DirichletBC],
    mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
    _snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    F: PETSc.Vec,  # type: ignore
):
    """
    Assemble the residual into the vector F.

    A function conforming to the interface expected by SNES.setResidual can
    be created by fixing the first four arguments:

        functools.partial(assemble_residual, u, jacobian, preconditioner,
                          bcs)

    Args:
        u: Function(s) tied to the solution vector within the residual and
            Jacobian.
        residual: Form of the residual. It can be a sequence of forms.
        jacobian: Form of the Jacobian. It can be a nested sequence of
            forms.
        bcs: List of Dirichlet boundary conditions.
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        F: Vector to assemble the residual into.
    """
    def function_to_vec(u, x):
        x.array[:] = u.x.array


    def vec_to_function(x, u):
        u.x.array[:] = x.array
        u.x.scatter_forward()

    x.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    # dolfinx.fem.petsc.assign(x, u)
    # vec_to_function(x, u)
    u.x.petsc_vec.array = x.array_r

    if isinstance(u, Sequence):
        for i in range(len(u)):
            mpc[i].homogenize(u[i])
            mpc[i].backsubstitution(u[i])
    else:
        mpc.homogenize(u)
        mpc.backsubstitution(u)

    # F.localForm().set(0.0)
    with F.localForm() as F_local:
            F_local.set(0.0)

    if x.getType() == "nest":
        dolfinx_mpc.assemble_vector_nest(F, residual, mpc)
    else:
        dolfinx_mpc.assemble_vector(residual, mpc, F)

    if x.getType() == "nest":
        bcs1 = dolfinx.fem.bcs.bcs_by_block(
            dolfinx.fem.forms.extract_function_spaces([[jacobian]], 1), bcs
        )
        dolfinx.fem.petsc._assign_block_data(residual, x)
        dolfinx_mpc.apply_lifting(F, jacobian, bcs=bcs1, constraint=mpc, x0=x, scale=-1.0)
        F.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        bcs0 = dolfinx.fem.bcs.bcs_by_block(
            dolfinx.fem.forms.extract_function_spaces(residual), bcs
        )
        dolfinx.fem.petsc.set_bc(F, bcs0, x0=x, alpha=-1.0)
    else:
        dolfinx_mpc.apply_lifting(F, [jacobian], bcs=[bcs], constraint=mpc, x0=[x], scale=-1.0)
        F.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, bcs, x0=x, alpha=-1.0)

    F.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)


class NonlinearProblemMPC(dolfinx.fem.petsc.NonlinearProblem):
    """
    Class for solving nonlinear problems with SNES.

    Solves problems of the form
    ::math:F_i(u, v) = 0, i=0,...N\\ \\forall v \\in V where
    ::math:u=(u_0,...,u_N), v=(v_0,...,v_N) using PETSc SNES as the
    non-linear solver.

    Args
    ----
    F: 
        UFL form(s) of residual :math:F_i.
    u: 
        Function used to define the residual and Jacobian.
    bcs: 
        Dirichlet boundary conditions.
    J: 
        UFL form(s) representing the Jacobian
        :math:J_ij = dF_i/du_j.
    P: 
        UFL form(s) representing the preconditioner.
    kind: 
        The PETSc matrix type(s) for the Jacobian and
        preconditioner (`MatType`).
        See :func:dolfinx.fem.petsc.create_matrix for more
        information.
    form_compiler_options: 
        Options used in FFCx compilation of all
        forms. Run `ffcx --help` at the command line to see all
        available options.
    jit_options: 
        Options used in CFFI JIT compilation of C code
        generated by FFCx. See `python/dolfinx/jit.py` for all
        available options. Takes priority over all other option
        values.
    petsc_options: 
        Options to pass to the PETSc SNES object.
    entity_maps: 
        If any trial functions, test functions, or
        coefficients in the form are not defined over the same mesh
        as the integration domain, `entity_maps` must be
        supplied. For each key (a mesh, different to the
        integration domain mesh) a map should be provided relating
        the entities in the integration domain mesh to the entities
        in the key mesh e.g. for a key-value pair `(msh, emap)`
        in `entity_maps, emap[i] is the entity in msh`
        corresponding to entity `i` in the integration domain
        mesh.
    """
    def __init__(
        self,
        F: ufl.form.Form | Sequence[ufl.form.Form],
        u: _fem.Function | Sequence[_fem.Function],
        mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
        bcs: Sequence[_fem.DirichletBC] | None = None,
        J: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        P: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        kind: str | Sequence[Sequence[str]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        petsc_options: dict | None = None,
        entity_maps: Sequence[dolfinx.mesh.EntityMap] | None = None,
    ):
        self._F = _fem.form(
            F,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self._u = u
        self.mpc = mpc
        self.bcs = [] if bcs is None else bcs
        self.preconditioner = P

        if J is None:
            if isinstance(F, Sequence):
                dus = [
                    ufl.TrialFunction(Fi.arguments()[0].ufl_function_space())
                    for Fi in F
                ]
                J = _fem.forms.derivative_block(F, u, dus)
            else:
                du = ufl.TrialFunction(F.arguments()[0].ufl_function_space())
                J = ufl.derivative(F, u, du)

        self._J = _fem.form(
            J,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        if P is not None:
            self._preconditioner = _fem.form(
                P,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
        else:
            self._preconditioner = None

        if kind == "nest" or isinstance(kind, Sequence):
            assert isinstance(mpc, Sequence)
            self._A = dolfinx_mpc.create_matrix_nest(self._J, mpc)
            kind = "nest"
        elif kind is None:
            assert isinstance(mpc, MultiPointConstraint)
            self._A = _cpp_mpc.create_matrix(self._J._cpp_object, mpc._cpp_object)
        else:
            raise ValueError(f"Unsupported kind for matrix: {kind}")

        kind = "nest" if self._A.getType() == "nest" else kind

        if kind == "nest":
            assert isinstance(mpc, Sequence)
            assert isinstance(self._F, Sequence)
            self._b = dolfinx_mpc.create_vector_nest(self._F, mpc)
            self._x = dolfinx_mpc.create_vector_nest(self._F, mpc)
        else:
            assert isinstance(mpc, MultiPointConstraint)
            # self._b = _fem.petsc.create_vector(
            #     [(mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs)]
            # )
            self._b = _fem.petsc.create_vector(self._F)
            # self._x = _fem.petsc.create_vector(
            #     [(mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs)]
            # )
            self._x = _fem.petsc.create_vector(self._F)

        if self._preconditioner is not None:
            if kind == "nest":
                assert isinstance(self._preconditioner, Sequence)
                assert isinstance(self.mpc, Sequence)
                self._P_mat = dolfinx_mpc.create_matrix_nest(self._preconditioner, self.mpc)
            else:
                assert isinstance(self._preconditioner, _fem.Form)
                self._P_mat = _cpp_mpc.create_matrix(self._preconditioner._cpp_object, kind=kind)
        else:
            self._P_mat = None

        self._snes = PETSc.SNES().create(comm=self._A.comm)
        self._snes.setJacobian(
            partial(
                assemble_jacobian_mpc,
                self._u,
                self._J,
                self._preconditioner,
                self.bcs,
                self.mpc,
            ),
            self._A,
            self._P_mat,
        )
        self._snes.setFunction(
            partial(

                assemble_residual_mpc,
                self._u,
                self._F,
                self._J,
                self.bcs,
                self.mpc,
            ),
            self._b,
        )

        problem_prefix = f"dolfinx_nonlinearproblem_{id(self)}"
        self._snes.setOptionsPrefix(problem_prefix)
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)

        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
            self._snes.setFromOptions()
            for k in petsc_options.keys():
                del opts[k]

        opts.prefixPop()

    def solve(self) -> tuple[_fem.Function | Sequence[_fem.Function], int, int]:
        def function_to_vec(u, x):
            x.array[:] = u.x.array


        def vec_to_function(x, u):
            u.x.array[:] = x.array
            u.x.scatter_forward()

        # function_to_vec(self._u, self._x)
        self._x.array_w = self._u.x.petsc_vec.array
        self._snes.solve(None, self._x)
        self._u.x.petsc_vec.array = self._x.array_r
        # vec_to_function(self._x, self._u)
        
        # _fem.petsc.assign(self._u, self._x)
        # self._snes.solve(None, self._x)
        # _fem.petsc.assign(self._x, self._u)

        if isinstance(self.mpc, Sequence):
            assert isinstance(self._u, Sequence)
            for i in range(len(self._u)):
                self.mpc[i].homogenize(self._u[i])
                self.mpc[i].backsubstitution(self._u[i])
        else:
            assert isinstance(self._u, _fem.Function)
            self.mpc.homogenize(self._u)
            self.mpc.backsubstitution(self._u)

        converged_reason = self._snes.getConvergedReason()
        return self._u, converged_reason, self._snes.getIterationNumber()


__all__ = [
    # nonlinear_problem
    "assemble_jacobian_mpc",
    "assemble_residual_mpc",
    "NonlinearProblemMPC",
]


