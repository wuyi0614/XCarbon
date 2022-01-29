# -*- encoding: utf-8 -*-
#
# Created at 27/10/2021 by Yi Wu, wymario@163.com
#

"""
Specify firm-like agents given theories from economics
"""

import numpy as np
from tqdm import tqdm
from copy import deepcopy

from core.base import generate_id, init_logger, jsonify_without_desc, read_config, Clock
from core.stats import mean, logistic_prob, range_prob, get_rand_vector
from core.market import Order, OrderBook, Statement
from core.config import get_energy_input_proportion, FOSSIL_ENERGY_TYPES, ENERGY_BASELINE_FACTORS

# env vars
logger = init_logger('Firm')


# functions for firms
def randomize_firm_specs(specs, role='random', inflation=0.1, random_factors=False):
    """Randomly create firm agents according to a template specification with proper randomization

    :param specs: A specification template for firms
    :param role: assign a role to the firm and make sure it has negative (buyer) or positive (seller) position
    :param inflation: firm-level factors will change in the range of [-inflation, inflation] by percents
    :param random_factors: Hardcore randomization that changes the core parameters (dangerous)

    The following parameters and variables will be randomized:
    - Output
    - Inputs (energy, material)
    - Price
    - EmissionFactor
    - Factors
    - Abatement

    The following varaibles will be fixed and shared across different agents:
    - Unit
    - CarbonPrice
    - Trade

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

        factors.update(factor_)

    # adjust output values
    ann_out_ratio = get_rand_vector(1, low=-20, high=20, is_int=True).pop()
    adj_ann_out = output['annual-output'] + ann_out_ratio
    adj_mon_out = adj_ann_out / 12
    alpha1 = factors['alpha1'] + ann_out_ratio  # keep track on annual output
    output.update({'annual-output': adj_ann_out, 'monthly-output': adj_mon_out, 'last-output': adj_ann_out})
    factors.update({'alpha1': alpha1})

    # adjust input price
    material_price = price['material-price']
    adj_energy_price = get_rand_vector(1, low=-10, high=10, is_int=True).pop()
    adj_material_price = get_rand_vector(1, low=-10, high=10, is_int=True).pop()
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


# TODO: allowance allocator for regulated firms and plz improve it in the future
def compliance_allocator(pos: float, clock: Clock, **spec) -> dict:
    """Allocator returns a generator which accepts Clock object and models in a form of series"""
    compliance_month = spec.get('compliance_month', 12)
    compliance_window = spec.get('compliance_window', 2)
    compliance_ratio = spec.get('compliance_ratio', 0.9)
    # how many months before the compliance
    ex_months = compliance_month - compliance_window - clock.Month + 1
    non_comp_month = pos * (1 - compliance_ratio) / ex_months
    comp_month = pos * compliance_ratio / (12 - ex_months)
    series = [non_comp_month] * ex_months + [comp_month] * (12 - ex_months)
    return {month: series[idx] for idx, month in enumerate(range(clock.Month, 13, 1))}


def non_compliance_allocator(pos: float, clock: Clock, **spec) -> dict:
    """Allocator returns a non-compliance like series of trading allowances,
    and allocator should returns allowances at a monthly basis and then it will be divided at a daily basis.
    """
    ex_months = 12 - clock.Month + 1
    series = [pos / ex_months] * ex_months
    return {month: series[idx] for idx, month in enumerate(range(clock.Month, 13, 1))}


# classes for firms
class BaseFirm:
    """
A base firm has the following attributes and behaviors and it also evolves with the time goes.
    Attributes:
      - production

    Functions:
      ...

    """

    def __init__(self, **kwargs):
        # the following are unchangeable attributes for a firm, e.g. `uid` is unique and fixed
        self.uid = generate_id()
        # the unit of economic value
        self.Unit = 1000
        # basic attributes (init state), the preferable structure of attrs is `dict`
        self.Step = 0
        # `dict` enables records about subsidiary attrs and all attrs should be defined to start with a capitalised char
        self.Attribute = {}  # here .Attribute is a generalized attribute
        self.Tag = 'unknown'  # industry tag that distinguishes firms
        self.Name = kwargs.get('name', self.uid)
        # set up parameters
        self.set_params(**kwargs)
        # not used but recorded stuff
        self._Record = {}

    def __repr__(self):
        """Output more information about the agent"""
        basis = self._snapshot()
        return f"""Firm [{self.uid}] at step={self.Step} with status: 
    {list(basis.items())}"""

    def _record(self, **kwargs):
        kws = {k.replace('_', '-'): v for k, v in kwargs.items()}
        self._Record.update(kws)
        return self

    def set_params(self, **params):
        # TODO: very important initialization process
        # copy-paste specifications from config file
        for k, v in jsonify_without_desc(params).items():
            if k in dir(self):  # the attr exists
                value = self.__getattribute__(k)
                if isinstance(v, dict) and isinstance(value, dict):
                    value.update(v)
                else:
                    value = v

                self.__setattr__(k, value)

    def _snapshot(self):
        """Take a snapshot of the agent"""
        basis = {}
        for attr in dir(self):
            if not attr.islower():
                basis[attr] = self.__getattribute__(attr)

        return basis

    def cache(self):
        """Make a clone of agent's status, which must be a json-like record in a log file"""
        logger.info(f"""Firm Cache [{self.uid}]:{list(self._snapshot().items())}""")

    def forward(self, show=False):
        # make changes to their attributes here and enter the next phrase
        self.action()
        self.Step += 1
        if show:
            logger.info(f'Firm {self.uid}: forward next step={self.Step}')

    def action(self, show=False):
        """Action space allows to specify different events, decisions, and behaviors and is executed by `forward`"""
        # hereby define a list of actions a firm would do in a period
        if show:
            logger.info(f'Firm {self.uid}: take action at step={self.Step}')


class RegulatedFirm(BaseFirm):
    """Regulated industry firms inherited from BaseFirm"""

    def __init__(self, clock, **spec):
        super(RegulatedFirm, self).__init__()
        # additional attributes for regulated firms (attributes should be unique and not crossed
        # NB. the unit should be unified into (Thousand RMB Yuan), e.g. 100,000 (000) output
        self.Input = {'energy': 0, 'material': 0}
        self.Input.update({energy: 0 for energy in FOSSIL_ENERGY_TYPES})

        self.Output = {'annual-output': 0, 'monthly-output': 0, 'last-output': 0}
        # product-price is determined outside of firms
        self.Price = {
            'energy-price': 0, 'material-price': 0, 'last-energy-price': 0, 'last-material-price': 0,
            'product-price': 0, 'last-product-price': 0
        }
        self.CarbonPrice = {'carbon-price': [0], 'last-carbon-price': [0], 'expected-carbon-price': [0]}
        self.Revenue = {'product-revenue': 0, 'last-product-revenue': 0,
                        'carbon-revenue': 0, 'last-carbon-revenue': 0,
                        'fine': 0}

        # because there is only one economic sector here in the model,
        # for the same production function, it has fixed parameters
        self.Factors = {'alpha0': 160, 'beta0': -0.2,  # params for demand function
                        'alpha1': 1000, 'alpha2': 0.3, 'sigma': 5 / 4,  # params for production function
                        'theta3': 0, 'gamma3': 0,  # params for logistic prob function
                        'decay': 1}  # annual decay rate of allocation

        # emission-related variables
        self.EmissionFactor = {'energy-factor': 0, 'material-factor': 0}
        self.Emission = {'energy-emission': 0, 'material-emission': 0, 'emission-all': 0,
                         'last-energy-emission': 0, 'last-material-emission': 0, 'last-emission-all': 0}
        self.Abatement = {'abate-investment': 0, 'last-abate-investment': 0,
                          'abate-cost': 0, 'last-abate-cost': 0,
                          'abate': 0, 'last-abate': 0}  # it's a variable on a yearly basis
        # variables in `Position` can only be determined in the model
        self.Position = {'position': 0, 'last-position': 0, 'trade-position': 0, 'last-trade-position': 0,
                         'monthly-trade-position': {}, 'daily-trade-position': 0}
        self.Allocation = {'allocation': 0, 'last-allocation': 0}

        # trade decision
        self.Trade = {'traded-allowance': [0], 'last-traded-allowance': [0],
                      'price-pressure-ratio': 0, 'quantity-pressure-ratio': 0}

        # agent status
        self.Status = 0  # 1=activated; 0=deactivated
        self.Clock = clock  # shared clock with the scheduler
        self.Compliance = 0  # 1=compliance trader; 0=non-compliance trader

        # set up parameters
        self.set_params(**spec)

        # for those variables that will not be updated but changeable through processing, name them with `_` as prefix
        self._Orders = {'order': [], 'deal': []}

    def activate(self):
        """Activate agent as a valid firm with status=1, step=1"""
        # switch on the agent
        self.Status = 1
        self.Step += 1
        # manually start actions
        self.action(show=True)
        return self

    def validate(self, threshold=None):
        """According to the current policy, firms consumes 26,000 tons coal will be included"""
        threshold = threshold if not threshold else 26000
        if self.Input['energy'] <= threshold:
            self.Status = 0

        return self

    def forward(self, show=False):
        """Forward functions to execute actions of agent and do validation in the end"""
        assert self.Status and self.Step > 1, f'Agent [{self.uid}] deactivated, please run .activate() first'

        self.action(show=show)
        # last, validate the status and over
        self.validate()
        return self

    @staticmethod
    def _get_energy_input(pi0, rho0, rho1, rho2, alpha1, alpha2):
        """Some best params: pi0=100, alpha=130, rho=0.6, alpha2=0.4"""
        pi1 = (pi0 / alpha1) ** rho0  # (pi0 / alpha1) is relatively fixed, but rho0 is not
        item1 = ((1 - alpha2) / pi1) * ((rho1 / rho2 / rho0) * ((1 - alpha2) / alpha2)) ** (rho0 / (1 - rho0))
        item2 = alpha2 / pi1
        return (item1 + item2) ** rho0

    @staticmethod
    def _get_material_input(pi0, rho0, rho1, rho2, alpha1, alpha2):
        pi1 = (pi0 / alpha1) ** rho0  # (pi0 / alpha1) is relatively fixed, but rho0 is not
        item1 = (alpha2 / pi1) * ((rho2 / rho1 / rho0) * (alpha2 / (1 - alpha2))) ** (rho0 / (1 - rho0))
        item2 = (1 - alpha2) / pi1
        return (item1 + item2) ** rho0

    @staticmethod
    def _get_expected_output(ue, ui, alpha1, alpha2, rho0):
        return alpha1 * (alpha2 * ((alpha2 * ue) ** rho0) + (1 - alpha2) * ((alpha2 * ui) ** rho0)) ** (1 / rho0)

    @classmethod
    def _get_inputs(cls, price, carbon_price, factors, emission_factors, pi0: float):
        # TODO: energy inputs should be reconsidered
        # make sure rho0/rho1/rho2 have a bigger value, rho1 and rho2 are relatively fixed values
        rho0 = 1 - 1 / factors['sigma']
        # TODO: ...
        rho1 = price['energy-price'] - mean(carbon_price['last-carbon-price']) * emission_factors['energy-factor']
        rho2 = price['material-price'] - mean(carbon_price['last-carbon-price']) * emission_factors['material-factor']
        rho1 = rho1 if rho1 != 0 else 0.1
        rho2 = rho2 if rho2 != 0 else 0.1
        # if one of them is negative, sigma should change (make sure rho0 * rho1 or rho0 * rho2 is positive)
        if (rho1 * rho0 * rho2) < 0:
            rho0 = 1 - 1 / (- factors['sigma'])

        # compute the optimal inputs
        ue = cls._get_energy_input(pi0, rho0, rho1, rho2, factors['alpha1'], factors['alpha2'])
        ui = cls._get_material_input(pi0, rho0, rho1, rho2, factors['alpha1'], factors['alpha2'])
        return ue, ui

    def _action_production(self):
        """Firms make production plan based on the minimized cost"""
        factors = self.Factors  # make persistence of variables for quick testing
        price = self.Price
        carbon_price = self.CarbonPrice
        emission_factors = self.EmissionFactor
        rho0 = 1 - 1 / factors['sigma']

        # NB. the initial output is given by the config but when Step>1, it's decided by product price
        if self.Step > 1:
            output = factors['alpha0'] * price['product-price'] ** factors['beta0']
            self.Output.update({'annual-output': output, 'monthly-output': output / 12})

        pi0 = self.Output['last-output']
        ue, ui = self._get_inputs(price, carbon_price, factors, emission_factors, pi0)

        # updated at 2021-12-08, energy use will be distributed by energy input structure
        prop = get_energy_input_proportion(self.Tag)
        energy_prop = {k: v * ue for k, v in prop.items()}
        energy_prop.update({'energy': ue, 'material': ui})

        y_ = self._get_expected_output(ue, ui, factors['alpha1'], factors['alpha2'], rho0)  # for purpose of display
        self._record(expected_output=y_)

        # update input and output
        self.Input.update(energy_prop)
        self.Output.update({'last-output': pi0})
        self.Output.update({'annual-output': pi0})

    def _action_allowance(self, decay=1):
        """Allowance allocation: benchmarking mechanism"""
        # compute actual emissions from production
        inputs = self.Input
        factors = self.EmissionFactor
        emission = self.Emission

        # updated at 2021-12-08, do not use `energy-factor` for emission accounting of energy inputs
        energy_emission = sum([inputs[k] * ENERGY_BASELINE_FACTORS[k] for k in FOSSIL_ENERGY_TYPES])
        material_emission = inputs['material'] * factors['material-factor']
        # update historic emissions
        self.Emission.update({'last-emission-all': emission['emission-all']})
        # update emissions in stock
        emission_all = material_emission + energy_emission
        self.Emission.update({'material-emission': material_emission, 'energy-emission': energy_emission,
                              'emission-all': emission_all})
        # update allocation and position
        self.Allocation.update({'last-allocation': self.Allocation['allocation']})
        self.Allocation.update({'allocation': emission['last-emission-all'] * decay})
        self.Position.update({'last-position': self.Position['position']})
        self.Position.update({'position': self.Allocation['allocation'] - emission_all})

    def _action_abatement(self):
        """Firms determine how much abatement they want to invest (yearly basis)"""
        carbon = self.CarbonPrice
        abate = self.Abatement
        position = self.Position

        mean_carbon_price = mean(carbon['last-carbon-price'])
        # compute the actual abatement volume and abatement investment
        # TODO: need to fix the abatement determination function
        # 这里不同的减排技术主要两种，一种是减少能耗；一种是直接捕捉CO2，如CCS
        # 如果企业一开始就有技术目录，他们按照成本来决定选择哪一种技术，会选择成本高的技术的企业被称为"leading firms"
        abate_opt = (mean_carbon_price * position['trade-position']) / (abate['abate-cost'] - mean_carbon_price)
        self._record(abate=abate_opt)

        # only when `ac < mean(pc)`, it's cheaper for firms to invest in abatement technology
        if (abate['abate-cost'] < mean_carbon_price) and position['position'] < 0:
            abate_inv = abate_opt * abate['abate-cost'] * (1 - abate['discount']) / abate['lifespan']
        else:
            abate_opt = 0
            abate_inv = 0

        # update abatement params
        self.Abatement.update({"abate": abate_opt, "abate-investment": abate_inv})
        # update position with abatement
        trade_position = position['position'] - abate_opt
        self.Position.update({'last-trade-position': position['trade-position']})
        self.Position.update({'trade-position': trade_position})
        self._position_adjuster(trade_position)

    def _position_adjuster(self, pos):
        # compliance or non-compliance trader
        if self.Compliance:
            self.Position.update({'monthly-trade-position': compliance_allocator(pos, self.Clock)})
        else:
            self.Position.update({'monthly-trade-position': non_compliance_allocator(pos, self.Clock)})
        return self

    def _action_trade_decision(self):
        """Trade decision happens on a specific frequency basis (usually the monthly basis)"""
        # TODO: 大企业比如华能集团（拥有很多的电厂和配额）的交易行为会因为持有配额数量多而发生改变
        # here you adjust the position of a firm
        carbon = self.CarbonPrice
        trade = self.Trade
        position = self.Position
        factors = self.Factors

        # compute the expected carbon price according its holdings
        current_carbon_price = carbon['carbon-price'][-1]
        month_position = position['monthly-trade-position'].get(self.Clock.Month, position['trade-position'])

        # use `-` to reverse the curve of `trade-position` because more postive holdings will lower the price
        # TODO: the pricing function determines that buyer's price will not exceed seller's, and notably
        # change `gammai` will reverse the trend if the gammi becomes negative.
        gammai = factors['gammai'] - self.Clock.Month / 9  # a bodge but it must be 9 (12 - 3, compliance period)
        if gammai == 0:
            gammai = gammai - 0.01

        # TODO: sellers are not expecting carbon prices falling, but they do hope it could grow
        # The true logic is, when they hold more, they trade more aggressively
        prob = range_prob(month_position / 100, 0.6)
        upper_prob = prob + 1
        lower_prob = 1 - prob
        # upper_prob = logistic_prob(theta=factors['thetai'], gamma=gammai, x=-position['trade-position'])
        # lower_prob = logistic_prob(theta=factors['thetai'], gamma=factors['gammai'], x=-position['trade-position'])
        if lower_prob <= upper_prob:
            prob = get_rand_vector(1, low=lower_prob, high=upper_prob).pop()  # price +- |eps|
        else:
            prob = get_rand_vector(1, low=upper_prob, high=lower_prob).pop()

        exp_carbon_price = current_carbon_price * prob
        self.CarbonPrice.update({'expected-carbon-price': carbon['expected-carbon-price'] + [exp_carbon_price]})
        self.Position.update({'daily-trade-position': month_position / self.Clock.workdays_left()})

        trade_position = position['trade-position']
        # must update monthly-trade-position otherwise the position will not be dynamically updated after trading.
        self._position_adjuster(trade_position)

        # test if it's under pressure either from price or quantity
        if self.Step > 3:  # only when the trading periods go over 4 steps
            if trade_position < 0:
                price_pressure_ratio = current_carbon_price / mean(carbon['carbon-price'][-4: -1])
            else:  # if firms are holding excessive allowances, they have pressure to sell them more quickly
                price_pressure_ratio = mean(carbon['carbon-price'][-4: -1]) / current_carbon_price

            price_drive_position = price_pressure_ratio * trade_position
            qty_drive_position = trade_position
            if trade_position < 0:
                qty_gap = -trade_position - sum(trade['traded-allowance'])
                if qty_gap > 0:
                    qty_pressure_ratio = logistic_prob(theta=factors['theta3'],
                                                       gamma=factors['gamma3'],
                                                       x=position['trade-position'])
                    qty_drive_position = qty_drive_position * qty_pressure_ratio

            if position['trade-position'] < 0:
                position_ = min([trade_position, price_drive_position, qty_drive_position])
            else:
                position_ = max([trade_position, price_drive_position, qty_drive_position])
            # update the trade position
            self._position_adjuster(position_)
            month_position = self.Position['monthly-trade-position'].get(self.Clock.Month, position_)
            self.Position.update({'trade-position': position_,
                                  'daily-trade-position': month_position / self.Clock.workdays_left()})

    def _action_revenue(self):
        """Firms accounts their revenue at the end of the year and affect the next-year's plan (production)"""
        # NB. remember to collect the revenue/cost from carbon trading
        pass

    def _trade(self, current: float):
        """Firm decides a transaction according to its trading needs, and firms react to changes in current price"""
        carbon = self.CarbonPrice
        position = self.Position
        factors = self.Factors

        if round(position['trade-position'], 5) == 0:
            return

        # TODO: use current price to adjust firm's strategy
        exp_carbon_price = carbon['expected-carbon-price'][-1]  # it changes only once a month
        if not current:
            current = exp_carbon_price

        # allow a daily pricing fluctuation from firms' side
        eps = logistic_prob(theta=factors['thetai'], gamma=factors['gammai'], x=exp_carbon_price - current)
        eps = abs(1 - eps) if abs(1 - eps) <= 0.1 else 0.1
        prob = 1 + get_rand_vector(1, low=-eps, high=eps).pop()  # price +- |eps|
        # TODO: apply a better strategy to update firm's expected carbon price

        # agent trades on a daily basis so it should be divided into even shares
        order = Order(price=exp_carbon_price * prob,
                      mode='sell' if position['trade-position'] > 0 else 'buy',
                      quantity=position['daily-trade-position'],
                      offerer_id=self.uid)

        self._Orders['order'] += [order.dict()]  # only for cache
        return order

    def trade(self, prob: float, current: float):
        """Firms trade once a day and they make bid/ask according to their position without reverse operations

        :param prob: the probability here indicates a specific prob that a firm would trade or not
        """
        # firms obtain a random digit to decide if they trade or not
        prob_ = range_prob(self.Position['trade-position'])
        return self._trade(current) if prob_ >= prob else None

    def _clear(self, order: Statement):
        """Offset the position by order's status"""
        position = self.Position
        trade = self.Trade
        qty = order.quantity
        if position['trade-position'] > 0:
            qty = -qty

        self.Position.update({'trade-position': position['trade-position'] + qty})
        self.Trade.update({'traded-allowance': trade['traded-allowance'] + qty})
        return self

    def clear(self, order: Statement):
        """Firms will retrieve accepted orders to change their positions"""
        # agent is independent from market clearing because it can be executed with any frequency
        self._clear(order)
        self._Orders['deal'] += [order.dict()]
        return self

    def action(self, show=False):
        if not self.Status:
            logger.warning(f'Firm {self.uid}: invalid and quit the market')
            return self

        # the following actions should eb executed in this area (especially those functions with `_` as prefix)
        self._action_production()
        self._action_allowance()
        self._action_abatement()
        self._action_trade_decision()

        if show:
            logger.info(f'Firm {self.uid}: take action at step={self.Step}')

        return self

    def quit(self):
        """Firms will quit the market if it emits less than a threshold or it loses money from production"""
        # TODO: combined with revenues, firms will quit.
        pass


if __name__ == '__main__':
    from core.base import Clock

    clock = Clock('2021-01-01')

    # Temporary test zone for all the classes and functions
    # Test: Agent test
    firm1 = BaseFirm(name='f1')
    firm1.cache()

    # Test: create a regulated firm and run single-node test
    conf = read_config('config/power-firm-20211027.json')
    reg1 = RegulatedFirm(clock, **conf)
    reg1.activate()
    order1 = reg1.trade(0)
    print(order1.dict())

    # Test: execute actions
    conf2 = randomize_firm_specs(conf)
    reg2 = RegulatedFirm(clock, **conf2)
    reg2.activate()
    order2 = reg2.trade(0)
    print(order2.dict())

    # Test: order settlement
    state, offerer, offeree = order1.settle(order2)

    ob = OrderBook(frequency='day')
    ob.put(order1)
    ob.put(order2)
    print(ob.pool)

    # Test: continuous listing & trading (mock a daily market)
    orders = {}
    for i in range(20):
        conf_ = randomize_firm_specs(conf)
        reg = RegulatedFirm(clock, **conf_)
        reg.activate()
        order = reg.trade(0, 0)
        orders[reg.uid] = order
        ob.put(order)
        print(f'{order.mode}: {order.price}')

    # Test: run models on daily basis
