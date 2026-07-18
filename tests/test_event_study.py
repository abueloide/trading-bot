"""Event-study gate: asymmetric drift passes, symmetric/thin samples fail."""
from events.event_study import event_study, gate_event


def test_asymmetric_drift_passes():
    good = [0.02, 0.03, 0.015, -0.005, 0.025, 0.01, -0.008, 0.02, 0.03, 0.012,
            0.018, -0.006, 0.022, 0.014, 0.02, 0.011]
    assert gate_event(event_study(good))["passed"] is True


def test_symmetric_no_edge_fails():
    flat = [0.01, -0.01, 0.012, -0.012, 0.008, -0.008, 0.01, -0.01,
            0.009, -0.009, 0.011, -0.011, 0.007, -0.007, 0.01, -0.01]
    assert gate_event(event_study(flat))["passed"] is False


def test_thin_sample_fails():
    assert gate_event(event_study([0.05, 0.04, 0.03]))["passed"] is False


def test_empty_is_safe():
    s = event_study([])
    assert s["n"] == 0 and gate_event(s)["passed"] is False
