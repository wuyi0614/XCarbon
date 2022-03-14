# -*- encoding: utf-8 -*-
#
# Created at 27/10/2021 by Yi Wu, wymario@163.com
#

"""
Specify firm-like agents given theories from economics
"""

from copy import deepcopy
from random import sample
from typing import Any

import pandas as pd

from core.base import generate_id, init_logger, Clock, BaseComponent
from core.stats import get_rand_vector
from core.config import get_energy_input_proportion, get_firm_distribution, \
    FOSSIL_ENERGY_TYPES, ENERGY_BASELINE_FACTORS, DEFAULT_MARKET_CAP, DEFAULT_EMISSION_UNIT

# env vars
logger = init_logger('Firm')

DEFAULT_FIRM_RANDOM_SPECS = ['CarbonIntensity', 'TotalEmission', 'Income']


# functions for firms
def populate_firm(industry: str, population: int):
    """Create industry-specific firms in population randomly or not

    :param industry: which type of industries for population, e.g. power, cement, iron, chemical
    :param population: number of firms
    """
    # obtain the distribution of firms' emission
    emi_dist = get_firm_distribution(industry, 'emission', population)
    emi_ratio = emi_dist / emi_dist.sum()
    emi_by_firm = DEFAULT_MARKET_CAP * emi_ratio  # the default unit: 1000 tonne

    # unit of intensity: t/1000 RMB, and then unit of income:
    int_by_firm = get_firm_distribution(industry, 'intensity', population)
    inc_by_firm = emi_by_firm * DEFAULT_EMISSION_UNIT / int_by_firm  # unit: 1e3 RMB

    # convert data into a pd.DataFrame and make a generator
    df = pd.DataFrame([int_by_firm, emi_by_firm, inc_by_firm]).T
    df.columns = DEFAULT_FIRM_RANDOM_SPECS
    for _, row in df.iterrows():
        yield row.to_dict()


def randomize_firm_specs(specs, role='random', inflation=0.1, random_factors=False):
    """(ABANDONED) Randomly create firm agents according to a template specification with proper randomization

    :param specs: A specification template for firms
    :param role: assign a role to the firm and make sure it has negative (buyer) or positive (seller) position
    :param inflation: firm-level factors will change in the range of [-inflation, inflation] by percents
    :param random_factors: Hardcore randomization that changes the core parameters (dangerous)

    The following parameters and variables will be randomized,

    Production:
    - AnnualOutput
    - ProductPrice
    - MaterialPrice

    Energy:
    - InputWeight, slightly adjustable
    - EnergyPrice, slightly adjustable

    CarbonTrade:
    - ExpectCarbonPrice, slightly adjustable according to firms' carbon intensity
    """
    # logics for firm generation, the principle is, price and output should be positively correlated
    template_spec = deepcopy(specs)

    tag = template_spec['Tag']
    price = template_spec['Price']
    inp = template_spec['Input']
    output = template_spec['Output']
    factors = template_spec['Factors']
    emission = template_spec['Emission']
    emission_factors = template_spec['EmissionFactor']
    abatement = template_spec['Abatement']
    carbon_price = template_spec['CarbonPrice']

    # adjust production function factors (dangerous!)
    if random_factors:
        foo = ['alpha0', 'beta0', 'thetai', 'gammai', 'alpha2', 'theta3', 'gamma3']
        factor_ = {}
        for f in foo:
            # TODO: the fluctuation seems arbitrary here. And more inflation rates are specified below
            r = get_rand_vector(1, low=-inflation, high=inflation, digits=3).pop()
            factor_[f] = factors[f] * (1 + r)
            if f == 'alpha2':
                if factor_[f] < 0.3:
                    factor_[f] = 0.3
                elif factor_[f] > 0.7:
                    factor_[f] = 0.7
                else:
                    continue

        factors.update(factor_)

    # adjust output values: the best value for pi1 ranges in [0.3, 0.4]
    ann_out_ratio = get_rand_vector(1, low=-20, high=20, is_int=True).pop() / 100
    adj_ann_out = output['annual-output'] + (1 + ann_out_ratio)
    if ann_out_ratio < 0:  # biased to the left, use smaller score to adjust
        best_pi1_score = sample(range(300, 350, 1), 1).pop() / 1000
    else:
        best_pi1_score = sample(range(350, 400, 1), 1).pop() / 1000

    alpha1 = adj_ann_out / best_pi1_score
    adj_mon_out = adj_ann_out / 12
    # keep track on annual output
    output.update({'annual-output': adj_ann_out, 'monthly-output': adj_mon_out, 'last-output': adj_ann_out})
    factors.update({'alpha1': alpha1})

    # adjust input price
    scale = 5
    material_price = price['material-price']
    adj_energy_price = get_rand_vector(1, low=-scale, high=scale, is_int=True).pop()
    adj_material_price = get_rand_vector(1, low=-scale, high=scale, is_int=True).pop()
    material_price += adj_material_price
    price.update({'material-price': material_price})
    # adjust emission factors: change in the same direction
    material_factor = emission_factors['material-factor'] * (1 + adj_material_price / 100)
    emission_factors.update({'material-factor': material_factor})

    # adjust input values
    input_output_ratio = 20  # fixed ratio: output / input(energy + material)
    input_all = adj_ann_out / input_output_ratio
    ue, ui = RegulatedFirm._get_inputs(price, carbon_price, factors, emission_factors, adj_ann_out)
    input_ratio = get_rand_vector(1, low=0.3, high=0.7, digits=3, is_int=False).pop()
    if role == 'random':
        ratio = 1  # nothing to adjust
    elif role == 'buyer':
        # temporarily know how many inputs needed
        ratio = 1 - get_rand_vector(1, low=0.15, high=0.4, digits=3, is_int=False).pop()
    elif role == 'seller':
        ratio = 1 + get_rand_vector(1, low=0.15, high=0.4, digits=3, is_int=False).pop()

    # role-deciding part
    prop = get_energy_input_proportion(tag)
    energy_prop = {k: v * ue for k, v in prop.items()}
    input_material = input_all * (1 - input_ratio) * ratio

    energy_prop.update({'energy': ue, 'material': input_material})
    inp.update(energy_prop)

    # adjust emissions (role-deciding)
    energy_emission = sum([energy_prop[k] * ENERGY_BASELINE_FACTORS[k] for k in FOSSIL_ENERGY_TYPES])
    material_emission = input_material * material_factor
    emission.update({'energy-emission': energy_emission, 'material-emission': material_emission,
                     'emission-all': energy_emission + material_emission})

    # adjust abatement: discount and lifespan prefer to remain the same
    adj_abate_ratio = get_rand_vector(1, low=-0.2, high=0.2, digits=3).pop()
    omega = abatement['omega'] * (1 + adj_abate_ratio)
    tau = abatement['tau'] * (1 - adj_abate_ratio)
    adj_cost_ratio = get_rand_vector(1, low=-10, high=30, digits=3).pop()
    abate_cost = abatement['abate-cost'] + adj_cost_ratio
    abatement.update({'omega': omega, 'tau': tau, 'abate-cost': abate_cost})

    # update the specification
    mapping = dict(Price='price', Input='inp', Output='output', Factors='factors',
                   EmissionFactor='emission_factors', Abatement='abatement')

    for key, element in mapping.items():
        template_spec.update({key: locals()[element]})

    return template_spec


# classes for firms
class BaseFirm(BaseComponent):
    """A base firm has the following attributes and behaviors, and it also evolves with the time goes."""

    unit: int = 1000
    status: int = 1  # 1=activated, 0=deactivated
    clock: object
    external: list = []

    def __init__(self, clock: Clock, **config):
        super().__init__(**config)
        # the following are unchangeable attributes for a firm, e.g. `uid` is unique and fixed
        self.uid = config.get('uif', generate_id())
        self.unit = 1000  # the unit of economic value
        self.clock = clock
        self.fit(**config)

    def __getitem__(self, item):
        """Query over all attributes in capital to get the item"""
        for k in dir(self):
            if not k.islower():
                attr = self.__getattribute__(k)
                if item in dir(attr):
                    return attr[item]

    def forward(self, show=False, **kwargs):
        # make changes to their attributes here and enter the next phrase
        if self.uid == 'unknown':
            logger.warning(f'UID not properly setup and remain {self.uid}')

        self._run(**kwargs)
        self._cache()
        self.step += 1
        if show:
            logger.info(f'Firm {self.uid}: forward the next step={self.step}')

        return self


class RegulatedFirm(BaseFirm):
    """Regulated industry firms inherited from BaseFirm"""

    compliance: str = 0  # 1=compliance trader; 0=non-compliance trader
    role: str = ''  # buyer or seller or None
    random: bool = False  # True=allow random, False=not random
    Trader: int = 1  # 1=trader, 0=quitter

    # key indicators (from external instances)
    Profit: float = 0.0
    EnergyInput: float = 0.0
    MaterialInput: float = 0.0

    Income: float = 0.0
    TotalEmission: float = 0.0
    CarbonIntensity: float = 0.0
    AnnualOutput: float = 0.0

    Allocation: float = 0.0
    Abate: float = 0.0
    Traded: float = 0.0
    Position: float = 0.0

    # the objects here are object instances
    external: list = ['Energy', 'CarbonTrade', 'Production', 'Abatement', 'Policy', 'Finance',
                      'FirmConfig', 'AbateConfig']
    Energy: Any
    CarbonTrade: Any
    Production: Any
    Abatement: Any
    Policy: Any
    Finance: Any

    # configurations
    AbateConfig: dict = {}
    FirmConfig: dict = {}

    def __init__(self, clock, **configs):
        """To initiate the Firm instance, take the following steps:

        1. input Clock
        2. input external instances: Energy, Production, etc.
        3. initiate external instances

        :param clock: initiated Clock object
        :param configs: configurations for instances, abatement, and instances

        Must-have parameters:
        - CarbonIntensity: carbon intensity (t/1e4RMB) and the power industry's average level is 18
        - Emission: total emission
        - Income: total income (revenue, not profit)
        """
        super(RegulatedFirm, self).__init__(clock, **configs)
        # set up parameters
        self.fit(**configs)

        # get obj_configs and abate_configs
        self.random = configs.get('random', False)

    def activate(self):
        """Activate instances by setting them up"""
        # initiate external instances
        # align production with our configs
        pparams = self.FirmConfig['Production']
        self.AnnualOutput = self.Income / pparams['ProductPrice']
        self.Production = self.Production(uid=self.uid,
                                          role=self.role,
                                          Income=self.Income,
                                          AnnualOutput=self.AnnualOutput,
                                          TotalEmission=self.TotalEmission,
                                          **pparams).forward()
        # update the other components
        self.Energy = self.Energy(self.random,
                                  uid=self.uid,
                                  Production=self.Production,
                                  TotalEmission=self.TotalEmission,
                                  **self.FirmConfig['Energy']).forward()
        self.CarbonTrade = self.CarbonTrade(uid=self.uid,
                                            clock=self.clock,
                                            Energy=self.Energy.snapshot(),
                                            Factors=self.Production.Factors,
                                            **self.FirmConfig['CarbonTrade']).forward()
        self.Abatement = self.Abatement(uid=self.uid,
                                        Energy=self.Energy.snapshot(),
                                        CarbonTrade=self.CarbonTrade.snapshot(),
                                        Production=self.Production.snapshot(),
                                        industry=self.industry, **self.AbateConfig).forward()

        # the following two instances will not be forwarded at this stage
        self.Policy = self.Policy(uid=self.uid,
                                  Energy=self.Energy,
                                  CarbonTrade=self.CarbonTrade,
                                  **self.FirmConfig['Policy'])
        self.Finance = self.Finance(uid=self.uid,
                                    Production=self.Production.snapshot(),
                                    Energy=self.Energy.snapshot(),
                                    Abatement=self.Abatement.snapshot(),
                                    CarbonTrade=self.CarbonTrade.snapshot(),
                                    Policy=self.Policy.snapshot(),
                                    **self.FirmConfig['Finance'])
        return self

    def trade(self, price, prob, show=False):
        """Forward functions to execute actions of agent and do validation in the end.

        :param show: configure the display option in the schedule to log firm behaviors
        :param kwargs: additional kwargs passed by `.forward()`"""
        if not self.status and self.step > 1:
            logger.warning(f'{self.industry} firm: [{self.uid}] deactivated, please run .activate() first')

        order = self.CarbonTrade.trade(price, prob)
        if show:
            logger.info(f'Firm {self.uid} {order.mode} {order.quantity} permits with {order.price} at step={self.step}')

        return order

    def forward_daily(self, statement):
        """Daily update includes:

        CarbonTrade: clear daily transactions
        """
        self.CarbonTrade.clear(statement)
        self.Traded = self.CarbonTrade.Traded
        return self

    def forward_monthly(self, carbon_price):
        """Monthly update includes:

        CarbonTrade: offset position and adjust carbon price by the average market price
        """
        self.CarbonTrade.forward_monthly(carbon_price)
        self.Position = self.CarbonTrade.AnnualPosition
        return self

    def forward_yearly(self, prod_price, mat_price, energy_price, carbon_price, abate_option):
        """Price included are updated in Scheduler. The yearly update includes:

        :param prod_price: the latest product price from Scheduler
        :param mat_price: the latest material price from Scheduler
        :param energy_price: the updated energy price in a dict form from Scheduler or real-world data
        :param carbon_price: the latest carbon price from the CarbonMarket (Orderbook)
        :param abate_option: the latest abatement option from Scheduler (see details in xxx-abate-yyy.json)

        Production:
        - arguments: ProductPrice, MaterialPrice
        - externals: CarbonTradeCache

        Energy:
        - arguments: EnergyUse, EnergyPrice
        - externals: None

        Abatement:
        - arguments: AbateOption (optional AdoptProb, AbateWeight, etc.)
        - externals: Energy, CarbonTrade, Production

        CarbonTrade:
        - arguments: None
        - externals: Energy, (Factors, initiated by Production)

        Policy:
        - arguments: None
        - externals: Energy, CarbonTrade

        Finance:
        - arguments: None
        - externals: Production, Energy, Abatement, CarbonTrade, Policy
        """
        # update Production instance by:
        self.Production.forward(ProductPrice=prod_price, MaterialPrice=mat_price,
                                CarbonTradeCache=self.CarbonTrade.cache)
        # update energy instance by `EnergyUse` or `EmissionFactor` in the future
        self.Energy.forward(EnergyUse=self.Production.EnergyInput, EnergyPrice=energy_price)
        # plump carbon price into Carbon Trade instance
        self.CarbonTrade.forward(CarbonPrice=carbon_price, Energy=self.Energy,
                                 reset=['Traded', 'TradeCost', 'TradeRevenue'])
        # update Abatement instance
        self.Abatement.forward(Energy=self.Energy, CarbonTrade=self.CarbonTrade, Production=self.Production,
                               AbateOption=abate_option)

        # update Policy instance
        self.Policy.forward(Energy=self.Energy, CarbonTrade=self.CarbonTrade)
        # update Finance instance
        self.Finance.forward(Production=self.Production, Energy=self.Energy, Abatement=self.Abatement,
                             CarbonTrade=self.CarbonTrade, Policy=self.Policy)

        # Note: using `forward` WILL NOT automatically make cache/reset of instances,
        #       because most of the attributes will be automatically refreshed
        self.Profit = deepcopy(self.Finance.Profit)
        self.AnnualOutput = deepcopy(self.Production.AnnualOutput)
        self.EnergyInput = deepcopy(self.Production.EnergyInput)
        self.MaterialInput = deepcopy(self.Production.MaterialInput)

        self.TotalEmission = deepcopy(self.Energy.get_total_emission())
        self.Allocation = deepcopy(self.Energy.Allocation)
        self.Abate = deepcopy(self.Abatement.TotalAbatement)

        # update firm's step and forward instance for cache in a loop
        for attr in self.external:
            if 'forward' in dir(attr):
                self.__getattribute__(attr).forward()

        self.step += 1
        return self


if __name__ == '__main__':
    # test of Firm instance was migrated to `component.py`
    pass
