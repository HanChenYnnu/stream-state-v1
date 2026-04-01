from src.mapping.daisee_to_learning_state import map_daisee_to_state


def test_mapping_blocked_priority():
    assert map_daisee_to_state(1, 3, 1, 3) == "S4"


def test_mapping_normal():
    assert map_daisee_to_state(0, 0, 3, 0) == "S0"
