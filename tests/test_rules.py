from src.rules.state_mapper import map_state


def test_map_away_state():
    state, conf = map_state(0.9, 0.1, 0.1, 0.0)
    assert state == "S5"
    assert conf >= 0.5


def test_map_blocked_state():
    state, _ = map_state(0.3, 0.8, 0.9, 1.0)
    assert state == "S4"
