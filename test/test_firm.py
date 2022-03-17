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

EXAMPLE_SPEC = list(populate_firm('power', 1)).pop()


def test_firm_initiation():
    specs = list(populate_firm('power', 10))
    assert len(specs) == 10, f'10 specs are exported'

    for spec in specs:
        firm = RegulatedFirm(clock=CLOCK,
                             random=True,
                             role='',
                             FirmConfig=FIRM_CONFIG,
                             AbateConfig=ABATE_CONFIG,
                             Energy=Energy,
                             CarbonTrade=CarbonTrade,
                             Production=Production,
                             Abatement=Abatement,
                             Policy=Policy,
                             Finance=Finance,
                             **spec)
        assert firm.TotalEmission == spec['TotalEmission']


def test_firm_monthly_forward():
    firm = RegulatedFirm(clock=CLOCK,
                         role='buyer',
                         FirmConfig=FIRM_CONFIG,
                         AbateConfig=ABATE_CONFIG,
                         Energy=Energy,
                         CarbonTrade=CarbonTrade,
                         Production=Production,
                         Abatement=Abatement,
                         Policy=Policy,
                         Finance=Finance,
                         **EXAMPLE_SPEC)

    firm.activate()
    # test firm's trading
    firm.trade(None, 0.02)
    # test monthly forward
    mean_price = 0.55
    firm.forward_monthly(mean_price)
    assert firm.CarbonTrade.step > 1, f'monthly forward only involves `CarbonTrade`'


def test_firm_yearly_forward():
    firm = RegulatedFirm(clock=CLOCK,
                         role='buyer',
                         FirmConfig=FIRM_CONFIG,
                         AbateConfig=ABATE_CONFIG,
                         Energy=Energy,
                         CarbonTrade=CarbonTrade,
                         Production=Production,
                         Abatement=Abatement,
                         Policy=Policy,
                         Finance=Finance,
                         **EXAMPLE_SPEC)

    for year in range(4):
        prod_price = get_rand_vector(1, 3, 200, 220, True).pop() / 100
        mat_price = get_rand_vector(1, 3, 95, 115, True).pop() / 100
        carbon_price = get_rand_vector(1, 3, 40, 80, True).pop() / 1000
        energy_price = {'coal': get_rand_vector(1, 3, 400, 800, True).pop() / 1000,
                        'gas': get_rand_vector(1, 3, 20, 60, True).pop() / 1000,
                        'oil': get_rand_vector(1, 3, 600, 800, True).pop() / 100,
                        'electricity': get_rand_vector(1, 3, 60, 80, True).pop() / 1e5}

        firm.forward_yearly(prod_price, mat_price, energy_price, carbon_price, firm.Abatement.AbateOption)
        assert firm.step > year + 1, f'forwarding'


