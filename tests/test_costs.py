import numpy as np
import pytest

from core.backtest import trade_costs
from core.costs import CostModel, funding_payment


def test_taker_round_trip_is_13bp_of_gross_notional():
    c = CostModel()
    # 5 bp fee + 1 bp half spread + 0.5 bp slippage per fill
    assert c.per_fill_rate() == pytest.approx(0.00065)
    # gross notional is traded twice (open + close of both legs)
    assert c.round_trip_rate() == pytest.approx(0.0013)


def test_maker_and_stress_multiplier():
    assert CostModel(use_maker=True).per_fill_rate() == pytest.approx(0.0002)
    assert CostModel(multiplier=1.5).per_fill_rate() == pytest.approx(0.00065 * 1.5)


def test_fill_cost_with_impact():
    c = CostModel()
    assert c.fill_cost(1000.0) == pytest.approx(0.65)
    # 1000 USD into a 100k USD bar = 1% participation -> 10bp * sqrt(0.01) = 1bp impact
    assert c.fill_cost(1000.0, 100_000.0) == pytest.approx(0.65 + 1000 * 0.001 * 0.1)


def test_funding_sign_and_window():
    ts = np.array([10, 20, 30], dtype=np.int64)
    rates = np.array([0.0001, 0.0002, -0.0005])
    # long pays positive funding for events with entry < f <= exit
    assert funding_payment(+1, 1000, ts, rates, 5, 20) == pytest.approx(0.3)
    assert funding_payment(-1, 1000, ts, rates, 5, 20) == pytest.approx(-0.3)
    assert funding_payment(+1, 1000, ts, rates, 10, 20) == pytest.approx(0.2)   # opened at the event: not charged
    assert funding_payment(+1, 1000, ts, rates, 5, 15) == pytest.approx(0.1)
    assert funding_payment(+1, 1000, ts, rates, 21, 29) == 0.0
    assert funding_payment(+1, 1000, ts, rates, 25, 40) == pytest.approx(-0.5)  # negative rate: long receives


def test_trade_costs_four_fills_hand_calculation():
    trade = {"w1": 0.5, "w2": 0.5, "side": 1, "entry_p1": 10.0, "entry_p2": 20.0,
             "exit_p1": 11.0, "exit_p2": 20.0, "entry_ts": 0, "exit_ts": 100}
    execution, fund = trade_costs(trade, 1000.0, CostModel(), None, "A", "B")
    # entry: 500 + 500, exit: 550 + 500 notional, each at 6.5 bp
    assert execution == pytest.approx((500 + 500 + 550 + 500) * 0.00065)
    assert fund == 0.0
    funding = {"A": (np.array([50]), np.array([0.001])), "B": (np.array([50]), np.array([0.001]))}
    _, fund = trade_costs(trade, 1000.0, CostModel(), funding, "A", "B")
    # long leg A pays 0.5, short leg B receives 0.5
    assert fund == pytest.approx(0.0)
