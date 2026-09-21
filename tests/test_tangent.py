from types import SimpleNamespace

import numpy as np
import pytest

from homicsx.core.material import MaterialState
from homicsx.homogenization import nlhelpers


class _Array:
    def __init__(self):
        self.array = np.array([0.25, -0.5])

    def scatter_forward(self):
        pass


def test_central_tangent_uses_backward_fallback_when_forward_fails(monkeypatch):
    u = SimpleNamespace(x=_Array())
    F_macro = SimpleNamespace(value=np.eye(2))
    context = SimpleNamespace(material_states=None)
    assignment = SimpleNamespace(has_history_dependence=lambda: False)
    base = np.eye(2)

    def fake_solve(problem, u_arg, F_macro_arg, F_perturbed, *args, **kwargs):
        F_macro_arg.value[...] = F_perturbed
        forward_first_column = F_perturbed[0, 0] > base[0, 0]
        converged = -1 if forward_first_column else 1
        return u_arg, converged, 2, 0.0

    def fake_average(domain, u_arg, F_macro_arg, *args, **kwargs):
        return 2.0 * F_macro_arg.value.copy(), 0.0, 1.0

    monkeypatch.setattr(nlhelpers, "_solve_once_with_history", fake_solve)
    monkeypatch.setattr(
        nlhelpers, "_compute_average_P_and_energy_with_state", fake_average
    )

    tangent = nlhelpers._compute_Ceff_fd_with_state(
        problem=object(),
        u=u,
        F_macro=F_macro,
        domain=object(),
        material_assignment=assignment,
        cell_tags=object(),
        Fbar=base,
        dim=2,
        quad_evaluator=object(),
        context=context,
        delta=1.0e-6,
    )

    np.testing.assert_allclose(tangent, 2.0 * np.eye(4), rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(F_macro.value, base)
    np.testing.assert_allclose(u.x.array, [0.25, -0.5])


def _scalar_state(value):
    state = MaterialState(1, ["q"])
    state.initialize_state("q", shape=(1,), initial_value=value)
    return {0: {0: state}}


def test_history_dependent_tangent_replays_step_and_restores_committed_state(
    monkeypatch,
):
    u = SimpleNamespace(x=_Array())
    F_macro = SimpleNamespace(value=np.eye(2))
    previous = _scalar_state(3.0)
    committed = _scalar_state(7.0)
    context = SimpleNamespace(
        material_states=committed,
        state_coefficients={},
        dt_constant=None,
    )
    assignment = SimpleNamespace(has_history_dependence=lambda: True)
    base = np.eye(2)

    def fake_solve(problem, u_arg, F_macro_arg, F_perturbed, context_arg, *args, **kwargs):
        F_macro_arg.value[...] = F_perturbed
        q = context_arg.material_states[0][0].get_state("q")
        # This is a trial step update from the restored previous state.
        q[...] = q + 4.0 + F_perturbed[0, 0]
        return u_arg, 1, 2, 0.0

    def fake_average(
        domain, u_arg, F_macro_arg, assignment_arg, tags, dim, quad, states,
        *args,
    ):
        q = states[0][0].get_state("q")[0, 0]
        return 2.0 * F_macro_arg.value.copy() + q * np.eye(2), 0.0, 1.0

    monkeypatch.setattr(nlhelpers, "_solve_once_with_history", fake_solve)
    monkeypatch.setattr(
        nlhelpers, "_compute_average_P_and_energy_with_state", fake_average
    )

    tangent = nlhelpers._compute_Ceff_fd_with_state(
        problem=object(),
        u=u,
        F_macro=F_macro,
        domain=object(),
        material_assignment=assignment,
        cell_tags=object(),
        Fbar=base,
        dim=2,
        quad_evaluator=object(),
        context=context,
        previous_material_states=previous,
        delta=1.0e-6,
    )

    expected = 2.0 * np.eye(4)
    expected[[0, 3], 0] += 1.0
    np.testing.assert_allclose(tangent, expected, rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(F_macro.value, base)
    np.testing.assert_allclose(u.x.array, [0.25, -0.5])
    assert context.material_states[0][0].get_state("q")[0, 0] == 7.0


def test_history_dependent_tangent_requires_previous_state():
    with pytest.raises(ValueError, match="previous_material_states"):
        nlhelpers._compute_Ceff_fd_with_state(
            problem=object(),
            u=SimpleNamespace(x=_Array()),
            F_macro=SimpleNamespace(value=np.eye(2)),
            domain=object(),
            material_assignment=SimpleNamespace(has_history_dependence=lambda: True),
            cell_tags=object(),
            Fbar=np.eye(2),
            dim=2,
            quad_evaluator=object(),
            context=SimpleNamespace(material_states={}),
        )
