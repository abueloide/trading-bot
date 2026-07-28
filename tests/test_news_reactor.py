"""News reactor: clasificación y reglas de decisión (camino de dinero)."""
from live.news_reactor import MAX_CONCURRENT, classify, decide, pick_symbol


def test_classifies_bullish_and_bearish():
    assert classify("Acme beats Q3 estimates").direction == "bullish"
    assert classify("Acme raises FY guidance").direction == "bullish"
    assert classify("Acme cuts outlook").direction == "bearish"
    assert classify("Acme downgraded to sell at Citi").direction == "bearish"


def test_noise_is_not_a_catalyst():
    assert classify("Acme names new CFO") is None
    assert classify("7 stocks to watch today") is None


def test_multi_ticker_roundup_is_skipped():
    assert pick_symbol(["A", "B", "C"]) is None
    d = decide(news_id="1", headline="Acme beats estimates", symbols=["A", "B"],
               ts="t", open_symbols=[], n_open=0)
    assert d.action == "skipped"


def test_bearish_is_logged_but_not_traded():
    d = decide(news_id="2", headline="Acme misses estimates", symbols=["ACME"],
               ts="t", open_symbols=[], n_open=0)
    assert d.action == "skipped" and d.direction == "bearish"


def test_no_duplicate_position():
    d = decide(news_id="3", headline="Acme beats estimates", symbols=["ACME"],
               ts="t", open_symbols=["ACME"], n_open=1)
    assert d.action == "skipped"


def test_concurrency_cap():
    d = decide(news_id="4", headline="Acme beats estimates", symbols=["ACME"],
               ts="t", open_symbols=["X"], n_open=MAX_CONCURRENT)
    assert d.action == "skipped"


def test_actionable_bullish_buys():
    d = decide(news_id="5", headline="Acme tops expectations", symbols=["ACME"],
               ts="t", open_symbols=[], n_open=0)
    assert d.action == "buy" and d.symbol == "ACME" and d.dollars > 0
