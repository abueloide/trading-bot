"""Re-auditoría de C2 (OpEx 1d drift long-only) contra la barra que subieron F3/F4.

C2 pasó el gate el 2026-07-24 y se desplegó a paper. Desde entonces el método
subió dos veces y C2 nunca enfrentó ninguno de los dos tests nuevos:

  F3 (2026-07-29) — jackknife: quitar los top contribuyentes. Mató un edge que
     pasaba placebo y split de régimen en ambos lados.
  F4 (2026-07-30) — el placebo debe llevar el MISMO condicionamiento que la regla
     real. C2 es long-only condicionada a día verde; su placebo de 2026-07-26
     muestreó días random SIN condicionar a verde → comparó una regla
     condicionada contra un control no condicionado.

Esto audita EXACTAMENTE lo que corre en paper (`strategy_opex_drift`: entra al
cierre del 3er viernes SOLO si cerró verde, sale 1 sesión después), no la versión
de dos patas con la que se validó.

Uso: python -m events.opex_audit
"""
from __future__ import annotations

import random
import statistics as st
import sys
from datetime import date
from typing import Dict, List, Sequence

from events.event_study import (
    CALENDARS, MIN_EVENTS, event_study, jackknife_by_event, leave_one_year_out,
)
from events.panic_study import p90, percentile_of

SYMBOLS = ("SPY", "QQQ", "IWM")
REGIMES = {"OOS 2015-21": (2015, 2022), "IS 2022-24": (2022, 2025)}
WINDOW = 1               # lo único que sobrevivió limpio en C2
PLACEBO_DRAWS = 200
PLACEBO_SEED = 42
JACKKNIFE_K = 3


def long_only_returns(df, event_dates: Sequence[str]) -> Dict[str, List[float]]:
    """Retornos de la regla desplegada, agrupados por año.

    Regla: si el día del evento cerró VERDE, entra al cierre y sale 1 sesión
    después (largo). Si cerró rojo, no opera — igual que `strategy_opex_drift`.
    """
    from events.event_study import _nearest_index

    idx = list(df.index)
    close = df["close"]
    by_year: Dict[str, List[float]] = {}
    for ds in event_dates:
        ts = _nearest_index(idx, ds)
        if ts is None:
            continue
        i = idx.index(ts)
        if i == 0 or i + WINDOW >= len(idx):
            continue
        if float(close.iloc[i] / close.iloc[i - 1] - 1) <= 0:
            continue                              # día rojo: la regla no opera
        fwd = float(close.iloc[i + WINDOW] / close.iloc[i] - 1)
        by_year.setdefault(str(ts.year), []).append(fwd)
    return by_year


def conditioned_placebo(df, exclude_ts: Sequence, n: int) -> Dict[str, object]:
    """Control con el MISMO condicionamiento que la regla real (lección de F4).

    Muestrea días NO-OpEx que además cerraron VERDE, y mide el mismo forward de
    1 sesión. Sin el filtro de verde, el control mide otra cosa que la regla y el
    percentil resultante está inflado por el condicionamiento, no por el evento.
    """
    idx = list(df.index)
    close = df["close"]
    excl = set(exclude_ts)
    pool = [
        i for i in range(1, len(idx) - WINDOW)
        if idx[i] not in excl and float(close.iloc[i] / close.iloc[i - 1] - 1) > 0
    ]
    if n == 0 or len(pool) < n:
        return {"exp_mean": 0.0, "exp_dist": []}
    rng = random.Random(PLACEBO_SEED)
    exps = []
    for _ in range(PLACEBO_DRAWS):
        rets = [
            float(close.iloc[i + WINDOW] / close.iloc[i] - 1)
            for i in rng.sample(pool, n)
        ]
        exps.append(event_study(rets)["expectancy_pct"])
    return {"exp_mean": round(st.mean(exps), 3), "exp_dist": exps}


def audit_cell(df, event_dates: Sequence[str]) -> Dict[str, object]:
    """Todos los descriptores + los dos tests nuevos para una celda símbolo×régimen."""
    from events.event_study import _nearest_index

    by_year = long_only_returns(df, event_dates)
    rets = [r for v in by_year.values() for r in v]
    used_ts = [t for t in (_nearest_index(list(df.index), d) for d in event_dates) if t is not None]
    return {
        "stats": event_study(rets),
        "p90_pct": p90(rets),
        "by_year": by_year,
        "jackknife": jackknife_by_event(rets, JACKKNIFE_K),
        "years": leave_one_year_out(by_year),
        "placebo": conditioned_placebo(df, used_ts, len(rets)),
        "returns": rets,
    }


def run() -> int:
    from backtesting.engine import load_bars

    dates = CALENDARS["OPEX"]
    print(f"C2 re-audit — OpEx long-only (regla desplegada), w={WINDOW}d, "
          f"jackknife k={JACKKNIFE_K}, placebo condicionado a verde ({PLACEBO_DRAWS} draws)\n")
    verdicts: Dict[str, bool] = {}

    for sym in SYMBOLS:
        for label, (y0, y1) in REGIMES.items():
            df = load_bars(sym, date(y0, 1, 1), date(y1, 1, 1), "1d", source="yfinance")
            a = audit_cell(df, dates)
            s, jk, yr, pb = a["stats"], a["jackknife"], a["years"], a["placebo"]
            pct = percentile_of(s["expectancy_pct"], pb["exp_dist"])
            jk_exp = jk["jackknifed"]["expectancy_pct"]

            sample_ok = s["n"] >= MIN_EVENTS
            jk_ok = jk_exp > 0
            yr_ok = yr["worst_drop_exp_pct"] is not None and yr["worst_drop_exp_pct"] > 0
            pb_ok = pct >= 90.0
            ok = sample_ok and s["expectancy_pct"] > 0 and jk_ok and yr_ok and pb_ok
            verdicts[f"{sym} {label}"] = ok

            print(f"{sym} · {label}")
            print(f"  n={s['n']}  exp={s['expectancy_pct']:+.3f}%  hit={s['hit_rate']:.0%}  "
                  f"tail={s['tail_ratio']}  p90={a['p90_pct']:+.2f}%  maxL={s['max_loss_pct']:+.2f}%")
            print(f"  jackknife -top{jk['dropped_k']}: {jk_exp:+.3f}%   {'OK' if jk_ok else 'MUERE'}")
            print(f"  leave-one-year-out: peor={yr['worst_drop_year']} → {yr['worst_drop_exp_pct']:+.3f}%  "
                  f"años+={yr['years_positive']}/{yr['years']}   {'OK' if yr_ok else 'MUERE'}")
            print(f"  placebo verde-condicionado: media={pb['exp_mean']:+.3f}%  "
                  f"pct={pct:.0f}   {'OK' if pb_ok else 'FALLA'}")
            print(f"  → {'PASS' if ok else 'FAIL'}\n")

    both = [s for s in SYMBOLS
            if all(verdicts.get(f"{s} {lbl}") for lbl in REGIMES)]
    print(f"Celdas PASS: {sum(verdicts.values())}/{len(verdicts)}")
    print(f"Símbolos que pasan en AMBOS regímenes: {both or 'NINGUNO'}")
    return 0 if both else 1


def _selfcheck() -> None:
    """Los tests nuevos deben detectar el fraude que buscan."""
    # jackknife: 20 ceros + 3 outliers → expectativa positiva que muere sin ellos
    rigged = [0.0] * 20 + [0.10, 0.09, 0.08]
    jk = jackknife_by_event(rigged, 3)
    assert jk["full"]["expectancy_pct"] > 0, jk
    assert jk["jackknifed"]["expectancy_pct"] == 0.0, jk

    # leave-one-year-out: un solo año carga todo → queda negativa sin él
    by_year = {"2020": [0.20], "2021": [-0.01], "2022": [-0.01], "2023": [-0.01]}
    yr = leave_one_year_out(by_year)
    assert yr["full_exp_pct"] > 0 and yr["worst_drop_year"] == "2020", yr
    assert yr["worst_drop_exp_pct"] < 0, yr
    assert yr["years_positive"] == 1, yr
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        raise SystemExit(run())
