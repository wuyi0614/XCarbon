# -*- encoding: utf-8 -*-
#
# Test script for components of regulated firms
#
# Created at 2022-03-09.

from pathlib import Path

from core.base import read_config, Clock
from core.component import *
from core.firm import RegulatedFirm

root_path = Path('config')
# (1) energy, (2) production, (3) abatement, (4) carbon trade, (5) policy, (6) finance
config = read_config(root_path / 'power-firm-20220206.json')
abate_config = read_config(root_path / 'power-abate-20220206.json')

clock = Clock('2020-01-01')

firm = RegulatedFirm(clock=clock,
                     role='buyer',
                     obj_config=config,
                     abate_config=abate_config,
                     Energy=Energy,
                     CarbonTrade=CarbonTrade,
                     Production=Production,
                     Abatement=Abatement,
                     Policy=Policy,
                     Finance=Finance)

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
