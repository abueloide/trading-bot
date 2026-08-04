"""F4 stop-study: ejecución del stop sobre barras diarias + la trampa del tail_ratio."""
import pandas as pd

from events.event_study import event_study
from events.stop_study import stopped_return, stopped_returns


def _series():
    """Entra a 100. Día 2 el low (93) perfora el stop de 5% sin que el open (96) lo haga."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame(
        {"open": [100, 99, 96, 110, 120],
         "high": [101, 100, 99, 112, 121],
         "low": [99, 96, 93, 105, 118],
         "close": [100, 98, 96, 110, 120]},
        index=idx,
    )
    return df, idx


def test_sin_stop_devuelve_el_retorno_completo():
    df, _ = _series()
    assert abs(stopped_return(df, 0, 4, 50.0) - 0.20) < 1e-9


def test_stop_tocado_por_el_low_sale_en_el_stop():
    df, _ = _series()
    assert abs(stopped_return(df, 0, 4, 5.0) - (-0.05)) < 1e-9


def test_open_bajo_el_stop_sale_al_open_no_al_stop():
    """Gap-through: el stop NO te salva del hueco de apertura."""
    df, idx = _series()
    gapdown = df.copy()
    gapdown.loc[idx[1], ["open", "high", "low", "close"]] = [80, 82, 78, 81]
    assert abs(stopped_return(gapdown, 0, 4, 5.0) - (-0.20)) < 1e-9


def test_evento_sin_ventana_suficiente_se_descarta():
    df, idx = _series()
    assert stopped_returns(df, [idx[-1]], 5, 5.0) == []
    assert len(stopped_returns(df, [idx[0]], 4, 5.0)) == 1


def test_el_stop_infla_el_tail_ratio_mecanicamente():
    """La trampa que obliga a correr el placebo CON el mismo stop (F4)."""
    sin_stop = event_study([0.10, -0.30, 0.10, -0.30])
    con_stop = event_study([0.10, -0.05, 0.10, -0.05])
    assert con_stop["tail_ratio"] > sin_stop["tail_ratio"]
    # misma expectativa NO: el stop también cambia el retorno. Lo que no cambia
    # es que un tail alto por truncamiento no es evidencia de edge.
    assert con_stop["hit_rate"] == sin_stop["hit_rate"]


def test_pata_corta_espejo():
    idx = pd.bdate_range("2020-01-01", periods=5)
    down = pd.DataFrame(
        {"open": [100, 95, 90, 85, 80], "high": [101, 96, 91, 86, 81],
         "low": [99, 94, 89, 84, 79], "close": [100, 95, 90, 85, 80]}, index=idx)
    r = stopped_returns(down, [idx[0]], 4, 50.0, signs={idx[0]: -1.0})
    assert abs(r[0] - 0.20) < 1e-9
