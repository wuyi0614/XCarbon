# -*- encoding: utf-8 -*-
#
# Test script for components of regulated firms
#
# Created at 2022-03-09.

from pathlib import Path

from core.component import *

from core.base import Clock, read_config
from core.market import OrderBook

# Test Pipeline (a loop within a year) by:
root_path = Path('config')
# (1) energy, (2) production, (3) abatement, (4) carbon trade, (5) policy, (6) finance
config = read_config(root_path / 'power-firm-20220206.json')
abate_config = read_config(root_path / 'power-abate-20220206.json')

clock = Clock('2020-01-01')

# test production module (a yearly clear)
production = Production(role='buyer',
                        **config['Production'])
production.forward()

# test energy module (a yearly clear)
energy = Energy(False, Production=production, **config['Energy'])
energy.forward()
pos = energy.Allocation - energy.get_total_emission()
assert pos < 0, f'Failed initiation with Role: {pos}'

# test carbon module
buyer = CarbonTrade(clock=clock, Energy=energy.snapshot(), uid='buyer', **config['Production'])
buyer.forward()  # a yearly forward (initial)

# test abatement module (a yearly clear)
abate = Abatement(Energy=energy.snapshot(),
                  CarbonTrade=buyer.snapshot(),
                  Production=production.snapshot(),
                  industry='power', **abate_config)
abate.forward()

# test daily trading decision and clearance
# - create a carbon account with excessive allowances
prod = Production(role='seller',
                  **config['Production'])
prod.forward()
e = Energy(False, Production=prod, **config['Energy'])
e.forward()

seller = CarbonTrade(clock=clock, Energy=e.snapshot(), uid='seller', **config['Production'])
seller.forward()  # a yearly forward (initial)

book = OrderBook(clock, 'day')
buys, sells = [], []
for _ in range(5):
    p = get_rand_vector(1, 3, low=0.045, high=0.06).pop()
    buy = buyer.trade(p, 0)
    sell = seller.trade(p, 0)
    buys += [buy]
    sells += [sell]
    book.put(sell)
    book.put(buy)
    clock.move(1)

# - clear statements
for idx, s in book.pool.items():
    statement = Statement(**s)
    buyer.clear(statement)
    seller.clear(statement)

# - test monthly clearance of carbon trade (update mean carbon price) and adjust the trading strategy
mean_carbon_price = book.to_dataframe().price.mean()
buyer.forward_monthly(mean_carbon_price)
seller.forward_monthly(mean_carbon_price)

# test policy module (a yearly clear)
policy = Policy(Energy=energy, CarbonTrade=buyer, **config['Policy'])
policy.forward()

# test finance module
finance = Finance(Production=production.snapshot(),
                  Energy=energy.snapshot(),
                  Abatement=abate.snapshot(),
                  CarbonTrade=buyer.snapshot(),
                  Policy=policy.snapshot(),
                  **config['Finance'])
finance.forward()

# test yearly clear for once
# -- test production
prod_price, mat_price = 2.1, 0.99
syn_energy_price = energy.get_synthetic_price()
syn_emission_factor = energy.get_synthetic_factor()

production.forward(SynEnergyPrice=syn_energy_price,
                   SynEmissionFactor=syn_emission_factor,
                   CarbonPriceCache=buyer.cache['CarbonPrice'],
                   ProductPrice=prod_price, MaterialPrice=mat_price)

# -- test energy: keep it in a dict form
energy_price = {'coal': 0.565, 'gas': 0.003, 'oil': 7, 'electricity': 0.0007}
energy.forward(Production=production, EnergyPrice=energy_price)

# -- test carbon trade
carbon_price = 0.06
buyer.forward(CarbonPrice=carbon_price, Energy=energy,
              reset=['Traded', 'TradeCost', 'TradeRevenue'])

# -- test abatement
abate.forward(Energy=energy, CarbonTrade=buyer, Production=production)

# -- test policy
policy.forward(Energy=energy, CarbonTrade=buyer)

# -- test finance
finance.forward(Production=production, Energy=energy, Abatement=abate,
                CarbonTrade=buyer, Policy=policy)
