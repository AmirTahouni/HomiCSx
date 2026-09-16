from types import SimpleNamespace

import numpy as np

from homicsx.fem import constraints


class _RecordingMPC:
    def __init__(self, _function_space):
        self.constraints = []
        self.finalized = False

    def create_periodic_constraint_geometrical(
        self, _function_space, locator, relation, _bcs
    ):
        self.constraints.append((locator, relation))

    def finalize(self):
        self.finalized = True


def test_nonlinear_3d_constraints_use_each_domain_length(monkeypatch):
    lengths = (2.0, 3.0, 4.0)
    mesh = SimpleNamespace(geometry=SimpleNamespace(dim=3))
    monkeypatch.setattr(
        constraints, "build_anchor_bc", lambda **kwargs: kwargs["anchor_point"]
    )
    monkeypatch.setattr(
        constraints.dolfinx_mpc, "MultiPointConstraint", _RecordingMPC
    )

    bcs, mpc = constraints.build_constraints_nonlinear(
        mesh=mesh,
        facet_tags=None,
        V=object(),
        domain_size=lengths,
        physical_tags=None,
    )

    assert set(bcs) == {
        (x, y, z)
        for x in (0.0, lengths[0])
        for y in (0.0, lengths[1])
        for z in (0.0, lengths[2])
    }
    assert mpc.finalized
    assert len(mpc.constraints) == 3

    sample_points = (
        np.array([[lengths[0]], [1.5], [2.0]]),
        np.array([[1.0], [lengths[1]], [2.0]]),
        np.array([[1.0], [1.5], [lengths[2]]]),
    )
    expected_master_points = (
        np.array([[0.0], [1.5], [2.0]]),
        np.array([[1.0], [0.0], [2.0]]),
        np.array([[1.0], [1.5], [0.0]]),
    )
    for (locator, relation), slave, master in zip(
        mpc.constraints, sample_points, expected_master_points
    ):
        assert locator(slave).item()
        np.testing.assert_allclose(relation(slave), master)


def test_nonlinear_constraints_reject_invalid_domain_sizes():
    mesh = SimpleNamespace(geometry=SimpleNamespace(dim=3))
    for domain_size in ((1.0, 1.0), (1.0, -1.0, 1.0), (1.0, np.nan, 1.0)):
        try:
            constraints.build_constraints_nonlinear(
                mesh=mesh,
                facet_tags=None,
                V=object(),
                domain_size=domain_size,
                physical_tags=None,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for domain_size={domain_size}")
