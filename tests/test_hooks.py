import logging
from types import SimpleNamespace

from homicsx.core.homogenization import SimulationState
from homicsx.homogenization.nlhelpers import _execute_hooks


def test_hooks_execute_in_registration_order_and_share_state():
    state = SimulationState()
    data = SimpleNamespace(state=state)
    calls = []

    def first(hook_data):
        calls.append("first")
        hook_data.state.set("value", 2)

    def second(hook_data):
        calls.append("second")
        hook_data.state.set("value", hook_data.state.get("value") * 3)

    _execute_hooks([first, second], data, "test")

    assert calls == ["first", "second"]
    assert state.get("value") == 6


def test_hook_failure_is_reported_and_does_not_skip_later_hooks(caplog):
    calls = []

    def failing(_):
        calls.append("failing")
        raise RuntimeError("intentional hook failure")

    def later(_):
        calls.append("later")

    with caplog.at_level(logging.ERROR):
        _execute_hooks([failing, later], object(), "post-stress")

    assert calls == ["failing", "later"]
    assert "post-stress hook failed" in caplog.text
    assert "intentional hook failure" in caplog.text


def test_simulation_state_preserves_persistent_values_between_load_cases():
    state = SimulationState()
    state.set("material_id", 7, scope="persistent")
    state.set("step_count", 4, scope="load_case")
    state.append_history("energy", 1.25)

    state.clear_load_case()

    assert state.get("material_id") == 7
    assert state.get("step_count") is None
    assert state.get_history("energy") == [1.25]
