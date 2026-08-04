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


# --- killer tests que subieron la barra (F3/F4) ---------------------------

def test_jackknife_by_event_kills_edge_carried_by_three_days():
    """20 ceros + 3 outliers: expectativa positiva que NO sobrevive sin ellos."""
    from events.event_study import jackknife_by_event
    jk = jackknife_by_event([0.0] * 20 + [0.10, 0.09, 0.08], k=3)
    assert jk["full"]["expectancy_pct"] > 0
    assert jk["jackknifed"]["expectancy_pct"] == 0.0


def test_jackknife_by_event_spares_a_broad_edge():
    from events.event_study import jackknife_by_event
    jk = jackknife_by_event([0.01] * 20, k=3)
    assert jk["jackknifed"]["expectancy_pct"] > 0


def test_leave_one_year_out_flags_single_year_edge():
    from events.event_study import leave_one_year_out
    yr = leave_one_year_out({"2020": [0.20], "2021": [-0.01],
                             "2022": [-0.01], "2023": [-0.01]})
    assert yr["full_exp_pct"] > 0
    assert yr["worst_drop_year"] == "2020"
    assert yr["worst_drop_exp_pct"] < 0
    assert yr["years_positive"] == 1


def test_leave_one_year_out_survives_when_broad():
    from events.event_study import leave_one_year_out
    yr = leave_one_year_out({str(y): [0.01, 0.012] for y in range(2015, 2022)})
    assert yr["worst_drop_exp_pct"] > 0
    assert yr["years_positive"] == 7
