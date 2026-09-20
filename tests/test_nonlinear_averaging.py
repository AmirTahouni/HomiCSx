import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from homicsx.core.material import MaterialAssignment, QuadraturePointEvaluator
from homicsx.core.mesh import PhysicalTags
from homicsx.homogenization.nlhelpers import (
    _compute_average_P_and_energy_with_state,
)


class _PrescribedHistoryMaterial:
    def requires_history(self):
        return True

    def get_quadrature_point_stress(self, state, F, quad_point_idx):
        return F[0, 0] * np.eye(2)

    def get_quadrature_point_energy(self, state, F, quad_point_idx):
        return F[0, 0] ** 2


def _nonuniform_two_triangle_mesh():
    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 2.0]],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    coordinate_element = basix.ufl.element(
        "Lagrange", "triangle", 1, shape=(2,)
    )
    return mesh.create_mesh(
        MPI.COMM_SELF, cells, points, ufl.Mesh(coordinate_element)
    )


def test_cell_volumes_and_history_averages_use_geometric_measure(monkeypatch):
    domain = _nonuniform_two_triangle_mesh()
    evaluator = QuadraturePointEvaluator(domain, degree=2)
    volumes = evaluator.compute_cell_volumes()
    np.testing.assert_allclose([volumes[0], volumes[1]], [1.0, 2.0])

    deformation_gradients = {
        0: np.tile(np.diag([2.0, 1.0]), (evaluator.num_quad_points, 1, 1)),
        1: np.tile(np.diag([5.0, 1.0]), (evaluator.num_quad_points, 1, 1)),
    }
    monkeypatch.setattr(
        evaluator,
        "compute_deformation_gradient_at_quad_points",
        lambda displacement, macro_gradient: deformation_gradients,
    )

    tdim = domain.topology.dim
    cell_tags = mesh.meshtags(
        domain,
        tdim,
        np.arange(2, dtype=np.int32),
        np.full(2, PhysicalTags().matrix, dtype=np.int32),
    )
    displacement_space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(displacement_space)
    macro_gradient = fem.Constant(domain, np.eye(2))
    assignment = MaterialAssignment(
        materials_by_phase={0: _PrescribedHistoryMaterial()}
    )

    stress, energy, jacobian = _compute_average_P_and_energy_with_state(
        domain=domain,
        u=displacement,
        F_macro=macro_gradient,
        material_assignment=assignment,
        cell_tags=cell_tags,
        dim=2,
        quad_evaluator=evaluator,
        material_states={0: {0: object(), 1: object()}},
    )

    # Areas are 1 and 2, so the exact measure-weighted averages are 4 and 18.
    np.testing.assert_allclose(stress, 4.0 * np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(energy, 18.0, atol=1.0e-12)
    np.testing.assert_allclose(jacobian, 4.0, atol=1.0e-12)
