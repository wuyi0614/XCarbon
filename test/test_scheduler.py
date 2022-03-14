# -*- encoding: utf-8 -*-
#
# Test script for scheduler
#
# Created at 2022-03-10.

from warnings import filterwarnings
from scheduler import Scheduler

from core.component import *
from core.firm import RegulatedFirm, populate_firm
from core.market import CarbonMarket, OrderBook
from core.base import Clock, read_config
from core.plot import plot_kline, plot_volume, plot_grid

from pytz_deprecation_shim import PytzUsageWarning

filterwarnings('ignore', category=PytzUsageWarning)

FIRM_CONFIG = read_config('config/power-firm-20220206.json')
ABATE_CONFIG = read_config('config/power-abate-20220206.json')
SCHEDULER_CONFIG = read_config('config/scheduler-20211027.json')


def test_scheduler_daily_clear():
    # create market object
    ck = Clock('2021-07-01')
    ob = OrderBook(ck, 'day')
    cm = CarbonMarket(ob, ck, 'day')
    # create scheduler object
    sch = Scheduler(cm, ck, 'day', **SCHEDULER_CONFIG)

    for spec in populate_firm('power', 10):
        reg = RegulatedFirm(clock=ck,
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
        sch.take(reg)

    # activate agents and offer trades
    sch._run('Scheduler')
    sch.daily_clear([0.2, 0.6])

    # check the following conditions:
    # - firms do not update their daily steps from 1 to 2
    assert sch.step == 0, 'Must be at step=0 for the daily loop'
    assert sch[0].Traded > 0, 'Throughout transactions, firm will have traded volume'


def test_scheduler_monthly_clear():
    # In the scheduler, `run()` action will be stopped automatically at the end of the month
    ck = Clock('2021-07-01')
    ob = OrderBook(ck, 'day')
    cm = CarbonMarket(ob, ck, 'day')

    # create scheduler object
    sch = Scheduler(cm, ck, 'day', **SCHEDULER_CONFIG)
    # create agents (considering both buyer/seller, compliance/non-compliance traders)
    for spec in populate_firm('power', 30):
        buyer = RegulatedFirm(clock=ck,
                              random=True,
                              role='buyer',
                              FirmConfig=FIRM_CONFIG,
                              AbateConfig=ABATE_CONFIG,
                              Energy=Energy,
                              CarbonTrade=CarbonTrade,
                              Production=Production,
                              Abatement=Abatement,
                              Policy=Policy,
                              Finance=Finance,
                              **spec)
        sch.take(buyer)

    for spec in populate_firm('power', 40):
        seller = RegulatedFirm(clock=ck,
                               random=True,
                               role='seller',
                               FirmConfig=FIRM_CONFIG,
                               AbateConfig=ABATE_CONFIG,
                               Energy=Energy,
                               CarbonTrade=CarbonTrade,
                               Production=Production,
                               Abatement=Abatement,
                               Policy=Policy,
                               Finance=Finance,
                               **spec)
        sch.take(seller)

    sch._run('Scheduler')
    sch.daily_clear([0.2, 0.6])
    sch.monthly_clear()


def test_scheduler_yearly_clear():
    ck = Clock('2021-07-16')
    ob = OrderBook(ck, 'day')
    cm = CarbonMarket(ob, ck, 'day')

    # create scheduler object
    sch = Scheduler(cm, ck, 'day', **SCHEDULER_CONFIG)
    for spec in populate_firm('power', 50):
        firm = RegulatedFirm(clock=ck,
                             random=True,
                             role='',
                             compliance=1,
                             FirmConfig=FIRM_CONFIG,
                             AbateConfig=ABATE_CONFIG,
                             Energy=Energy,
                             CarbonTrade=CarbonTrade,
                             Production=Production,
                             Abatement=Abatement,
                             Policy=Policy,
                             Finance=Finance,
                             **spec)
        sch.take(firm)

    # if the left is higher (>0.5), it means small position trading is limited
    sch.run(probs=[0., 1], dist='logistic', show=True)  # demo: [0.2, 0.7]

    report = cm.to_dataframe()
    assert not report.empty, 'Market report should not be empty!'
    array = report[['open', 'close', 'low', 'high']].astype(float).values.tolist()
    dates = report.ts.values.tolist()
    # draw kline price chart
    c1 = plot_kline(array, dates, 'carbon price', 'trade-price-baseline')

    # draw bar volume chart
    volumes = report['volume'].astype(float).values.round(1).tolist()
    c2 = plot_volume(volumes, dates)
    plot_grid(c1, c2, 'trade-volume-baseline-compliance', True)

