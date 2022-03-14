# -*- encoding: utf-8 -*-
#
# Test script for market/orderbook
#
# Created at 2022-03-14.

from core.base import Clock
from core.market import *

CLOCK = Clock('2020-01-01')


def test_order_success_settlement():
    buy = Order(price=0.5, quantity=-300, mode="buy", offerer_id="o1")
    sell = Order(price=0.4, quantity=200, mode="sell", offerer_id="s1")
    # successful sell settlement:
    state, offerer, offeree = buy.settle(sell)
    assert state.price == 0.45
    assert offerer.quantity == -100
    assert offeree.status == "accepted"


def test_order_failure_settlement():
    buy = Order(price=0.5, quantity=-300, mode="buy", offerer_id="o1")
    sell = Order(price=0.6, quantity=200, mode="sell", offerer_id="s1")
    state, offerer, offeree = buy.settle(sell)
    assert state is None
    assert offerer.status == "waiting"
    assert offeree.status == "waiting"


def test_market_validity():
    # use OrderBook to match orders
    ob = OrderBook(CLOCK, frequency='day')
    # Test: market function
    market = CarbonMarket(ob=ob, clock=CLOCK, freq='day')
    market.open()
    market.close()
    assert not market.to_dataframe().empty
