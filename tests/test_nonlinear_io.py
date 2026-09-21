import csv
from types import SimpleNamespace

import numpy as np
import pytest

from homicsx.core.material import MaterialAssignment, MaterialState
from homicsx.core.mesh import PhysicalTags
from homicsx.homogenization.nlhelpers import (
    _assert_observational_hooks_preserved_live_data,
    _macroscopic_result_is_physical,
    _mesh_tag_lookup,
    _save_history_csv,
    _save_state_history_csv,
)


def test_macroscopic_result_finiteness_rejects_infinities_and_invalid_jacobian():
    stress = np.eye(2)
    assert _macroscopic_result_is_physical(stress, 1.0, 1.0)
    infinite_stress = stress.copy()
    infinite_stress[0, 0] = np.inf
    assert not _macroscopic_result_is_physical(infinite_stress, 1.0, 1.0)
    assert not _macroscopic_result_is_physical(stress, np.inf, 1.0)
    assert not _macroscopic_result_is_physical(stress, 1.0, np.inf)
    assert not _macroscopic_result_is_physical(stress, 1.0, 0.0)


def test_observational_hook_guard_restores_context_macro():
    macro = SimpleNamespace(value=np.eye(2))
    context = SimpleNamespace(F_macro=macro, material_states=None)
    vector = SimpleNamespace(array=np.array([1.0, 2.0]), scatter_forward=lambda: None)
    function = SimpleNamespace(x=vector)
    macro_before = macro.value.copy()
    macro.value[0, 0] = 9.0

    with pytest.raises(ValueError, match="solver-owned context"):
        _assert_observational_hooks_preserved_live_data(
            context,
            None,
            function,
            function.x.array.copy(),
            macro_before,
            "post_tangent",
        )
    np.testing.assert_array_equal(macro.value, macro_before)


def test_sparse_meshtag_lookup_uses_entity_ids_not_value_positions():
    sparse_tags = SimpleNamespace(
        indices=np.array([2, 9], dtype=np.int32),
        values=np.array([41, 71], dtype=np.int32),
    )
    assert _mesh_tag_lookup(sparse_tags) == {2: 41, 9: 71}


def test_state_initialization_uses_sparse_meshtag_entities():
    class HistoryMaterial:
        @staticmethod
        def requires_history():
            return True

        @staticmethod
        def initialize_state(num_quad_points):
            return MaterialState(num_quad_points, [])

    sparse_tags = SimpleNamespace(find=lambda tag: np.array([2, 9], dtype=np.int32))
    assignment = MaterialAssignment(materials_by_phase={4: HistoryMaterial()})
    states = assignment.initialize_states(
        mesh=object(),
        cell_tags=sparse_tags,
        quad_evaluator=SimpleNamespace(num_quad_points=2),
        physical_tags=PhysicalTags(matrix=41, phase_tag_offset=70),
        matrix_phase_id=4,
    )
    assert set(states[4]) == {2, 9}


def test_history_csv_uses_current_schema_and_fixed_width(tmp_path):
    history = {
        "step": [1, 2],
        "load_type": ["shear", "shear"],
        "load_param": [0.1, 0.2],
        "Fbar": [np.eye(2), np.eye(2)],
        "Pbar": [np.eye(2), 2.0 * np.eye(2)],
        "Wbar": [1.0, 2.0],
        "Jbar": [1.0, 1.0],
        "converged": [1, 1],
        "iters": [2, 3],
        "Ceff": [None, np.eye(4)],
        "tangent_status": ["not_scheduled", "computed"],
    }
    output = tmp_path / "history.csv"
    _save_history_csv(history, str(output), dim=2)
    with output.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 3
    assert all(len(row) == len(rows[0]) for row in rows)
    assert "tangent_status" in rows[0]
    assert rows[1][rows[0].index("tangent_status")] == "not_scheduled"


def _state(variable, values):
    values = np.asarray(values, dtype=float)
    state = MaterialState(values.shape[0], [variable])
    state.initialize_state(variable, values.shape[1:], 0.0)
    state.get_state(variable)[...] = values
    return state


def test_state_csv_supports_heterogeneous_variable_layouts(tmp_path):
    snapshot = {
        0: {2: _state("Cv", np.arange(8).reshape(2, 2, 2))},
        1: {9: _state("damage", np.array([[0.1], [0.2], [0.3]]))},
    }
    output = tmp_path / "states.csv"
    _save_state_history_csv([(0.5, snapshot)], str(output), dim=2)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["variable"] for row in rows} == {"Cv", "damage"}
    assert {int(row["phase_id"]) for row in rows} == {0, 1}
    assert len(rows) == 2 * 4 + 3
