# -*- encoding: utf-8 -*-
#
# Test script for components of regulated firms
#
# Created at 2022-03-09.

from pathlib import Path

from core.component import *

from core.base import Clock, read_config, jsonify_config_without_desc
from core.market import OrderBook, Order
from core.firm import populate_firm

# (1) energy, (2) production, (3) abatement, (4) carbon trade, (5) policy, (6) finance
FIRM_CONFIG = jsonify_config_without_desc(read_config(Path('config') / 'power-firm-20220206.json'))
ABATE_CONFIG = jsonify_config_without_desc(read_config(Path('config') / 'power-abate-20220206.json'))
CLOCK = Clock('2020-01-01')

# example specification for firm
EXAMPLE_FIRM_SPEC = list(populate_firm('power', 1000)).pop()

# generalized params for production
PRODUCTION_PARAMS = FIRM_CONFIG['Production']
PRODUCTION_PARAMS.update(AnnualOutput=EXAMPLE_FIRM_SPEC['Income'] / PRODUCTION_PARAMS['ProductPrice'],
                         ProductRevenue=EXAMPLE_FIRM_SPEC['Income'],
                         TotalEmission =EXAMPLE_FIRM_SPEC['TotalEmission'])


# test production module (a yearly clear)
def test_component_production_validity():
    production = Production(role='', **PRODUCTION_PARAMS)
    production.forward()
    assert production.cache, 'must not be empty'


def test_component_production_role_setting():
    production = Production(role='buyer', **PRODUCTION_PARAMS)
    production.forward()
    factors = production['Factors']
    assert factors['rho'] > 0.1, 'a higher rho for a buyer'
    assert factors['adjustor'] < 1, 'a lower adjustor for a buyer'


def test_component_energy_validity():
    production = Production(role='', **PRODUCTION_PARAMS)
    production.forward()

    energy = Energy(False, Production=production, **FIRM_CONFIG['Energy'])
    energy.forward()

    assert energy.Emission, 'must not be empty'


def test_component_energy_role_setting():
    for role, cond in (['seller', True], ['buyer', False]):
        production = Production(role=role, **PRODUCTION_PARAMS)
        production.forward()
        energy = Energy(False, Production=production, **FIRM_CONFIG['Energy'])
        energy.forward()

        pos = energy.Allocation > energy.get_total_emission()
        assert pos == cond, f'Failed initiation with Role: {energy.Allocation} - {energy.get_total_emission()}'


def test_component_carbon_trade_validity():
    production = Production(role='', **PRODUCTION_PARAMS)
    production.forward()
    energy = Energy(False, Production=production, **FIRM_CONFIG['Energy'])
    energy.forward()

    carbon = CarbonTrade(clock=CLOCK, Energy=energy.snapshot(), **FIRM_CONFIG['Production'])
    carbon.forward()  # a yearly forward (initial)

    assert carbon.cache, 'must not be empty'


def test_component_carbon_trade_role_setting():
    production = Production(role='buyer', **PRODUCTION_PARAMS)
    production.forward()
    energy = Energy(False, Production=production, **FIRM_CONFIG['Energy'])
    energy.forward()

    carbon = CarbonTrade(clock=CLOCK, Energy=energy.snapshot(), **FIRM_CONFIG['Production'])
    order = carbon.trade()
    assert order, 'must not be Nonetype'


production = Production(role='buyer',
** FIRM_CONFIG['Production'])
production.forward()
energy = Energy(False, Production=production, **FIRM_CONFIG['Energy'])
energy.forward()

buyer = CarbonTrade(clock=CLOCK, Energy=energy.snapshot(), **FIRM_CONFIG['Production'])
buyer.forward()  # a yearly forward (initial)

assert buyer.cache, 'must not be empty'

# test abatement module (a yearly clear)
abate = Abatement(Energy=energy.snapshot(),
CarbonTrade = buyer.snapshot(),
Production = production.snapshot(),
industry = 'power', ** abate_config)
abate.forward()

# test daily trading decision and clearance
# - create a carbon account with excessive allowances
prod = Production(role='seller',
** config['Production'])
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
Energy = energy.snapshot(),
Abatement = abate.snapshot(),
CarbonTrade = buyer.snapshot(),
Policy = policy.snapshot(),
** config['Finance'])
finance.forward()

# test yearly clear for once
# -- test production
prod_price, mat_price = 2.1, 0.99
syn_energy_price = energy.get_synthetic_price()
syn_emission_factor = energy.get_synthetic_factor()

production.forward(SynEnergyPrice=syn_energy_price,
SynEmissionFactor = syn_emission_factor,
CarbonPriceCache = buyer.cache['CarbonPrice'],
ProductPrice = prod_price, MaterialPrice = mat_price)

# -- test energy: keep it in a dict form
energy_price = {'coal': 0.565, 'gas': 0.003, 'oil': 7, 'electricity': 0.0007}
energy.forward(Production=production, EnergyPrice=energy_price)

# -- test carbon trade
carbon_price = 0.06
buyer.forward(CarbonPrice=carbon_price, Energy=energy,
reset = ['Traded', 'TradeCost', 'TradeRevenue'])

# -- test abatement
abate.forward(Energy=energy, CarbonTrade=buyer, Production=production)

# -- test policy
policy.forward(Energy=energy, CarbonTrade=buyer)

# -- test finance
finance.forward(Production=production, Energy=energy, Abatement=abate,
CarbonTrade = buyer, Policy = policy)
