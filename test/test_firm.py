# -*- encoding: utf-8 -*-
#
# Test script for components of regulated firms
#
# Created at 2022-03-09.

from pathlib import Path

from core.base import read_config, Clock
from core.component import *
from core.firm import RegulatedFirm, populate_firm

FIRM_CONFIG = read_config(Path('config') / 'power-firm-20220206.json')
ABATE_CONFIG = read_config(Path('config') / 'power-abate-20220206.json')
CLOCK = Clock('2020-01-01')


def test_firm_trading():
    firm = RegulatedFirm(clock=CLOCK,
                         role='buyer',
                         CarbonIntensity=18,  # 18 ton/ wton
                         Emission=10 * 18,  # 180 kton
                         Income=1e5,  # 1e8 RMB -> 1e5 kton -> 1e4 wton
                         FirmConfig=FIRM_CONFIG,
                         AbateConfig=ABATE_CONFIG,
                         Energy=Energy,
                         CarbonTrade=CarbonTrade,
                         Production=Production,
                         Abatement=Abatement,
                         Policy=Policy,
                         Finance=Finance)

    firm.activate()
    # test firm's trading
    firm.trade(None, 0.02)
    # test monthly forward
    mean_price = 0.55
    firm.forward_monthly(mean_price)

# test long-term forwards
for year in range(4):
    prod_price = get_rand_vector(1, 3, 200, 220, True).pop() / 100
    mat_price = get_rand_vector(1, 3, 95, 115, True).pop() / 100
    carbon_price = get_rand_vector(1, 3, 40, 80, True).pop() / 1000
    energy_price = {'coal': get_rand_vector(1, 3, 400, 800, True).pop() / 1000,
                    'gas': get_rand_vector(1, 3, 20, 60, True).pop() / 1000,
                    'oil': get_rand_vector(1, 3, 600, 800, True).pop() / 100,
                    'electricity': get_rand_vector(1, 3, 60, 80, True).pop() / 1e5}

    firm.forward_yearly(prod_price, mat_price, energy_price, carbon_price)
    print(f'Year: {year}, {firm.Allocation - firm.Emission}')

# test firm-level trading
order = firm.trade(0.06, 0.1)
assert order, f'InvalidFirmSpecification'


def test_firm_population():
    pop = 1000
    i, e, c = populate_firm('power', pop)
    for idx in range(pop):
        firm = RegulatedFirm(clock=ck,
                             random=True,
                             role='',
                             FirmConfig=FIRM_CONFIG,
                             AbateConfig=ABATE_CONFIG,
                             Energy=Energy,
                             CarbonTrade=CarbonTrade,
                             Production=Production,
                             Abatement=Abatement,
                             Policy=Policy,
                             Finance=Finance)
