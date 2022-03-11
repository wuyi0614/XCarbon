#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Energy and emission abatement module for X-Carbon Modeling
#
# Created at 2022-01-02.
#

import copy

from scipy.optimize import minimize
from collections import defaultdict
from typing import Any

import numpy as np
from tqdm import tqdm

from core.base import init_logger, jsonify_config_without_desc, BaseComponent
from core.stats import random_distribution, get_rand_vector, range_prob, logistic_prob, mean, \
    choice_prob
from core.config import DEFAULT_ELEC_UNIT, DEFAULT_GAS_UNIT
from core.market import Order, Statement

# create a logger
logger = init_logger('energy', level='WARNING')


# instance management
# TODO: use the Pool (like scheduler) to manage the instances (updating)
class Manager:
    """Manager is an instance for the management of objects (e.g. Energy, Abatement, etc.) in bulk"""

    def __init__(self, namespace):
        self.pool = []
        self.namespace = namespace
        self.Step = 0
        self.local = []
        self.globe = []

    def take(self, obj):
        self.pool.append(copy.deepcopy(obj))

    def reset(self):
        self.__init__(self.namespace)
        return self

    def add_local_pipe(self, func):
        """Add localized functions into the pipe that only works on single object.
        The local pipe is prioritised than global pipes.
        """
        self.local += [func]

    def add_global_pipe(self, func):
        """Add global functions into the pipe that works on all the objects, such as
        the marginal_abate_cost must be calculated based on all the objects.
        """
        self.globe += [func]

    def step(self):
        for obj in tqdm(self.pool, desc=f'Object pool running at step {self.Step}'):
            for f in self.local:  # first priority
                obj = f(obj)

            for g in self.globe:
                g(self.pool)

            obj.step()  # usually the final step


class Energy(BaseComponent):
    """The Energy object for an industrial firm with featured attributes"""

    EnergyType: list = []
    EnergyUse: float = 0  # a float/int type aggregate energy consumption (unit: tonne)
    InputWeight: dict = {}  # proportion of each energy type
    InputEnergy: dict = {}  # specific energy input of each energy type

    gas_unit: float = DEFAULT_GAS_UNIT
    elec_unit: float = DEFAULT_ELEC_UNIT

    # emission factor:
    # tCO2/tonne for coal&oil&gas, and unit of gas should be converted from Nm3 to tonne
    # tCO2/kWh for electricity
    EmissionFactor: dict = {}
    Emission: dict = {}
    Allocation: float = 0.0

    # energy prices:
    # 1,000 currency/tonne for coal&oil&gas,
    # 1,000 currency/kWh for electricity
    EnergyPrice: dict = {}
    EnergyCost: float = 0

    # external instances
    external: list = ['Production']
    Production: Any

    # attributes that skip cache and reset (unless assigned new values)
    nocache: list = ['EnergyType', 'InputWeight', 'EmissionFactor']

    def __init__(self, random=False, role=None, **config):
        """Create an Energy object in random or not mode

        :param random: True for randomizing the input weight, False for not
        :param role: `buyer` for emission>allocation, `seller` for emission<allocation, None for random
        :param config: initial configuration params for Energy module
        """
        super().__init__(role=role, **config)
        # list all energies
        self.EnergyType = list(self.InputWeight.keys())

        if random:  # slightly randomize the InputWeight of energy
            vector = random_distribution(list(self.InputWeight.values()))
            for idx, (k, v) in enumerate(self.InputWeight.items()):
                self.InputWeight[k] = vector[idx]

        self.EnergyUse = self.Production['EnergyInput']
        # generate the initial emissions of the firm (historic EnergyUse should be higher)
        if self.Production.role == 'seller':
            last_use = self.EnergyUse * get_rand_vector(1, 3, 100, 120).pop() / 100
        elif self.Production.role == 'buyer':
            last_use = self.EnergyUse * get_rand_vector(1, 3, 80, 100).pop() / 100
        else:
            last_use = self.EnergyUse * get_rand_vector(1, 3, 80, 120).pop() / 100

        energy = self.get_energy(last_use, self.InputWeight)
        self.Emission = self.get_emission(energy, self.EmissionFactor)
        self.Allocation = self.get_total_emission()
        self._cache()

    def _run(self, **kwargs):
        """Implicitly compute emissions and aggregate inputs by running the function"""
        # NB. for gas, the exchange rate between `tonne` and `m3` is `1t=1.4Nm3`
        self.EnergyUse = self.Production['EnergyInput']
        self.InputEnergy = self.get_energy(self.EnergyUse,
                                           self.InputWeight)
        self.Emission = self.get_emission(self.InputEnergy,
                                          self.EmissionFactor)
        # align `Allocation` with the last total Emission
        self.Allocation = sum(list(self.cache['Emission'][-1].values()))
        self.EnergyCost = self.get_cost(self.gas_unit, self.elec_unit)

    @staticmethod
    def get_energy(use, weight):
        """Compute energy input by types"""
        energy = {}
        if use > 0:
            for k, v in weight.items():
                # use `v * use / unit` to convert the unit
                input_ = v * use
                energy[k] = round(input_, 3)

        else:
            logger.error(f'Invalid EnergyUse value: {use}')

        return energy

    @staticmethod
    def get_emission(energy, factors):
        """Compute emissions from energy consumption"""
        emission = {}
        for k, v in energy.items():
            emission[k] = round(v * factors[k], 3)

        return emission

    def get_total_emission(self):
        return sum(list(self.Emission.values()))

    def get_allocation(self):
        """Compute allowance allocation based on the last-year emissions"""
        last_emission = self.cache['Emission'][-1]
        return sum(list(last_emission.values()))

    def get_cost(self, gas_unit, elec_unit):
        """Compute costs of energy consumption"""
        cost = 0
        for k, v in self.InputEnergy.items():
            if k == 'gas':
                input_ = v / gas_unit
            elif k == 'electricity':
                input_ = v / elec_unit
            else:
                input_ = v

            cost += round(input_ * self.EnergyPrice[k], self.digits)

        return cost

    def get_synthetic_price(self):
        """Compute the synthetic energy price"""
        syn = np.array(list(self.EnergyPrice.values())) * np.array(list(self.InputWeight.values()))
        return round(syn.sum(), self.digits)

    def get_synthetic_factor(self):
        """Compute the synthetic emission factor by Emission/EnergyUse"""
        syn = sum(list(self.Emission.values())) / self.EnergyUse
        return round(syn, self.digits)


# local/global functions can be added in Manager's pipe
def global_margin_abate_cost(objs: list):
    """Marginal abatement cost is the concept that describes the cost of abatement will be descending
    when the total abatement goes higher. The adjustment usually happens at a yearly basis.
    """
    pass


def local_func(obj):
    pass


class Abatement(BaseComponent):
    """
    The Abatement object for an industrial firm. It can be understood as a technical package
    where options are available for various energy. The technical options are industry-specific:

    NB. Different industries may have limited availability of options, e.g. cement factories are
    hard to shift a majority of their coal consumption to gas.

    - Efficiency: energy saving & efficiency improvement, use less energy (including renewables for electricity),
                  target at `EmissionFactor`
    - Shift: specific energy consumption shift from one to the other, target at `InputWeight`
    - Negative: negative emission removal techniques such as CCS, target at `Emission`

    If the values are not negative for `efficiency` and `negative`, they could be percentages.

    The logics are that,
        (1) more investment are made, more abatement are generated if the investment doesn't
        exceed the potential of a specific technique.
        (2) The total abatement is decided by optimizing the total cost, and it should be
        distributed across various options based on their costs.
    """
    # NB. emission abatement could be long-term and thus the abatement should be discounted and
    # calculated in the short term (dependent on their lifespans), which should be included in the
    # `AbateOption` attribute.
    TechniqueType: list = []  # efficiency, ccs, shift (i.e. from coal to gas)
    TotalAbatement: float = 0  # total abatement
    AbateCost: float = 0  # aggregate cost of abatement

    SuggestPosition: float = 0  # if abatement, SuggestPosition!=Position; otherwise, equal
    Option: dict = {}  # adopted options (if `shift` in it, update Energy instance)

    # abatement configs
    AbateOption: dict = {}  # abatement options (portfolio) for various energy, except for ccs
    AbateWeight: dict = {}  # share of abatement techniques, decided after the cost optimization

    # adoption probability parameter
    AdoptProb: dict = {}  # the minimum acceptable entry probability
    AdoptProbParam: dict = {}

    # inherit specifications
    external: list = ['CarbonTrade', 'Energy', 'Production']
    Energy: dict = {}  # get from `Energy.snapshot`
    CarbonTrade: dict = {}  # get from `CarbonTrade.snapshot()`
    Production: dict = {}  # get from `Production.snapshot()`

    nocache: list = ['TechniqueType', 'AbateOption', 'AbateWeight', 'AdoptProb', 'AdoptProbParam']

    def __init__(self, **config):
        super().__init__(**jsonify_config_without_desc(config))
        # list all abatement options
        self.TechniqueType = list(self.AbateOption.keys())

    def _run(self, **kwargs):
        """Execute all `get` functions to update.
        When running this at a yearly basis, plump in `Energy`, `CarbonTrade` and `Production`"""
        options = self.get_abate_options()
        self.update_abate(options)

    def update_abate(self, options):
        """Update abatement options and its costs (update optimal position as well)"""
        abatement, cost = 0, 0
        for key, value in options.items():
            if key in self.TechniqueType:
                abatement += value['abate']
                cost += value['cost'].sum()

        # TODO: should persist the structure of abatement options and map them with the price changes
        self.TotalAbatement = round(abatement, self.digits)
        self.AbateCost = round(cost, self.digits)
        self.SuggestPosition = round(self.SuggestPosition - abatement, self.digits)

    @staticmethod
    def get_efficiency_abatement(energy, fuel, potential, price, discount, lifespan, digits=6, **kwargs):
        """Return a tuple of discounted abatement (Emission x Potential) and cost (unit: 1000)"""
        abate = energy['Emission'][fuel] * potential
        cost = price * abate
        discounted = cost / (1 + discount) ** lifespan
        return abate, round(discounted, digits)

    @staticmethod
    def get_shift_abatement(energy, org, tar, potential, price, discount, lifespan, digits=6, **kwargs):
        """Return a tuple of discounted abatement and its cost (unit: 1000)"""
        weights = copy.deepcopy(energy['InputWeight'])
        change = weights[org] * potential
        weights[tar] += change
        weights[org] -= change
        # use new weights to calculate energy use
        use = Energy.get_energy(energy['EnergyUse'], weights)
        emission = Energy.get_emission(use, energy['EmissionFactor'])
        # calculate the actual abatement and its cost
        abate = sum(list(energy['Emission'].values())) - sum(list(emission.values()))
        cost = price * abate
        discounted = cost / (1 + discount) ** lifespan
        return abate, round(discounted, digits)

    @staticmethod
    def get_negative_abatement(energy, potential, price, discount, lifespan, digits=6, **kwargs):
        """Return (discounted abate, cost) of negative technology options"""
        abate = sum(list(energy['Emission'].values())) * potential
        cost = price * abate
        discounted = cost / (1 + discount) ** lifespan
        return abate, round(discounted, digits)

    @staticmethod
    def get_optimal_abate(w, position, carbon_price, abate_price, maximum, init=None) -> np.ndarray:
        """Get optimal abatement options and trade position. Notably,
        :param w: weight for three options, np.ndarray with size 3
        :param position: trade position of the firm (must be buyer and positive), float
        :param carbon_price: annual average carbon price, float
        :param abate_price: abatement price of three options, np.ndarray with size 3
        :param maximum: maximum abatement potential, np.ndarray with size 3
        :param init: initial values for abates, np.ndarray with size 4

        :return np.ndarray: a list of option abatement
        """

        # define the minimum function
        def minimum_abate(x):
            # x: 0=efficiency, 1=shift, 2=negative, 3=trade
            # w: 0=eff-weight, 1=shift-weight, 2=neg-weight
            return w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + x[3]

        if init is None:
            init = np.zeros(4)

        cons1 = [
            # cons1: sum(abates) >= position
            {'type': 'ineq', 'fun': lambda x: x.sum() - position},
            # cons2: mean(abateCost) <= carbonPrice
            {'type': 'ineq', 'fun': lambda x: carbon_price * x[3] - x[:3] * abate_price}
        ]
        # cons3: each abatement cannot exceed their potential
        cons2 = [{'type': 'ineq', 'fun': lambda x: maximum[idx] - x[idx]} for idx in range(3)]
        # cons4: each abatement cannot be negative
        cons3 = [{'type': 'ineq', 'fun': lambda x: x[idx]} for idx in range(4)]
        cons = cons1 + cons2 + cons3

        result = minimize(minimum_abate,
                          init,
                          constraints=cons,
                          method='SLSQP',
                          options={'disp': False})
        return result.x

    def get_abate_options(self):
        """Compute the actual abatement by optimizing the cost. To find the optimized option,
        it uses optimization function to find the best weight."""
        position = sum(list(self.CarbonTrade['MonthPosition'].values()))
        if position > 0:
            return {}
        else:
            position = abs(position)

        parser = {'Efficiency': self.get_efficiency_abatement,
                  'Shift': self.get_shift_abatement,
                  'Negative': self.get_negative_abatement}

        abates, costs = defaultdict(float), defaultdict(float)
        for optType, option in self.AbateOption.items():
            for fuel, param in option.items():
                # get the theoretical abatement by `optType`
                abate, cost = parser[optType](self.Energy, fuel=fuel, **param)
                abates[optType] += abate
                costs[optType] += cost

        # run optimize function
        abate_price = np.array(list(costs.values())) / np.array(list(abates.values()))
        carbon_price = self.CarbonTrade['CarbonPrice']
        optimum = self.get_optimal_abate(np.array(list(self.AbateWeight.values())),
                                         position,
                                         carbon_price,
                                         abate_price,
                                         np.array(list(abates.values())))
        opt_position = optimum[3]
        # export abatement options
        options = {}
        for idx, each in enumerate(self.TechniqueType):
            options[each] = {'abate': optimum[idx], 'cost': abate_price * optimum[idx]}

        # get adoption probability
        options = self.get_adopt_prob(options,
                                      opt_position,
                                      carbon_price,
                                      abate_price)
        if options:
            options.update({'Position': optimum[3]})
            return options
        else:
            return {'Position': position}

    def get_adopt_prob(self, options, opt_position, carbon_price, abate_price):
        """Compute probabilities for options based on firm's financial statement"""
        # factors for the probability calculation:
        #   x Profit, higher profit, higher probability (use -1=failure, 1=profit) [abandoned]
        #   2. Position, lower position, higher probability (optimized position)
        #   3. Carbon Price, higher price, higher probability
        #   4. Abate Cost, lower cost, higher probability
        # NB. the base value here can significantly determine the probability
        for idx, (opt, cost) in enumerate(zip(self.TechniqueType, abate_price)):
            args = [opt_position, carbon_price, cost]
            prob = choice_prob(*args, **self.AdoptProbParam)
            if prob < self.AdoptProb[opt]:
                options.pop(opt)
            else:
                continue

        return options


class CarbonTrade(BaseComponent):
    """Carbon trading component controls firm's trading decisions"""
    # general specification
    frequency: str = 'day'
    clock: Any  # Clock object
    quit: float = 1  # a firm with `position=0` do not trade if `quit=1`
    order: list = []  # where historic orders are stored (`Order` object)
    lockable: float = 1  # 1=True, 0=False, if True, allow to lock the firm when its |position|<0.001
    lock: float = 0  # 1=True, can trade; 0=False, cannot trade; use `Lock` only when lock=1

    # transactions will bring up with costs or revenues
    TradeCost: float = 0.0  # negative for buyer (cost), positive for seller (revenue)
    TradeRevenue: float = 0.0

    # carbon price and adjustment params
    CarbonPrice: float = 0.0  # carbon price unit: yuan/metric ton
    ExpectCarbonPrice: float = 0.0  # higher/lower than the market price
    PricePressureRatio: float = 0
    QuantPressureRatio: float = 0

    # position management
    DayPosition: float = 0.0
    MonthPosition: dict = {}  # a long-term plan on the specific position of each month
    AnnualPosition: float = 0.0  # allowance surplus/shortage in total (yearly basis)
    QuantLastComplete: float = 0.0  # the t-1 completed transaction

    ComplianceTrader: int = 0  # 1=compliance trader; 0=non-compliance trader
    ComplianceSetting: dict = {}  # compliance setting works only if `ComplianceTrader=1`

    Traded: float = 0.0  # traded allowance accounting

    # inherited from external instances in the form `dict`
    external: list = ['Energy', 'Factors']
    Energy: dict = {}  # get from `Energy.snapshot()`
    Factors: dict = {}  # get from `Production.Factors`

    nocache: list = ['Factors', 'PricePressureRatio', 'QuantPressureRatio',
                     'ComplianceTrader', 'ComplianceSetting']

    def __init__(self, **config):
        super().__init__(**config)

    def _run(self, **kwargs):
        """Must be careful about the execution order of functions. The update frequency is `month`.
        Basically it is: adjust_position > get_price > get_decision"""
        self.adjust_position(**kwargs)
        self.get_price(**kwargs)
        if self.step > 1:
            self.forward_yearly(**kwargs)

    def forward_monthly(self, price, **kwargs):
        """Monthly update decision with caches (skipping values at a yearly basis)"""
        # monthly update does not `reset` attributes
        self.CarbonPrice = price
        self.adjust_position(**kwargs)
        self.get_price(**kwargs)

        nocache = ['AnnualPosition', 'MonthPosition', 'DayPosition',
                   'CarbonPrice', 'ExpectCarbonPrice', 'QuantLastComplete', 'Traded']
        self._cache(nocache + self.nocache)
        self.step += 1
        return self

    def forward_yearly(self, **kwargs):
        """Yearly update decision"""
        energy = self.Energy
        position = energy['Allocation'] - sum(list(energy['Emission'].values()))
        if position != 0:
            self.AnnualPosition = position
            self.lock = 0  # set it 0 if position is refreshed

    # NB. allocator can be tailored not only for `compliance/non-compliance`
    def compliance_allocator(self) -> dict:
        """Allocator returns a generator which accepts Clock object and models in a form of series (monthly)"""
        compliance_month = self.ComplianceSetting.get('compliance_month', 12)
        compliance_window = self.ComplianceSetting.get('compliance_window', 2)
        compliance_ratio = self.ComplianceSetting.get('compliance_ratio', 0.8)

        # how many months before the compliance
        ex_months = compliance_month - compliance_window - self.clock.Month

        position = np.zeros(12)
        # sum up the positions before the current month (12 months)
        total_position = self.AnnualPosition
        past = self.clock.Month - 1
        position[:past] = 0  # set 0 for the past months

        # TODO: issue - if compliance month is not Dec, it causes issues, and also if compliance month
        #       happens in the next year, it will be an error too.
        if ex_months > 0:
            noncomp_month = total_position * (1 - compliance_ratio) / ex_months
            position[past: (past + ex_months)] = noncomp_month
        else:
            noncomp_month = 0

        # compliance months are the period within the compliance window
        comp_month = (total_position - noncomp_month) / compliance_window
        position[(past + ex_months): compliance_month] = comp_month
        return {idx: round(position[idx - 1], self.digits) for idx in range(1, 13, 1)}

    def non_compliance_allocator(self) -> dict:
        """Allocator returns a non-compliance like series of trading allowances,
        and allocator should returns allowances at a monthly basis and then it will be divided at a daily basis.
        """
        position = np.zeros(12)
        total_position = self.AnnualPosition
        past = self.clock.Month - 1
        position[:past] = 0  # set 0 for the past months

        position[past:] = total_position / (12 - self.clock.Month + 1)
        return {idx: round(position[idx - 1], self.digits) for idx in range(1, 13, 1)}

    def update_annual_position(self, position: float):
        """Update monthly/daily position by passing a new AnnualPosition (before .reset())"""
        self.AnnualPosition = position
        self.adjust_position()

    def adjust_position(self, **kwargs):
        """Compliance trader shows different patterns"""
        # TODO: 大企业比如华能集团（拥有很多的电厂和配额）的交易行为会因为持有配额数量多而发生改变
        # test if it's under pressure either from price or quantity
        if self.step == 0:  # initiation
            self.forward_yearly(**kwargs)

        elif self.step > 2:  # only when the trading periods go over 3 months
            position = kwargs.get('position', self.get_position('month'))
            price = self.CarbonPrice
            factors = self.Factors

            # get price-driven position adjustment
            hist_price = self.cache['CarbonPrice'][-4: -1]
            mean_price = mean(hist_price) if hist_price else 0

            # NB: mean price could be 0 only in tests
            mean_price = mean_price if mean_price > 0 else price

            price_pressure = price / mean_price if position < 0 else mean_price / price
            price_drive_position = price_pressure * position

            # get quantity-driven position adjustment
            traded_position = sum(self.cache['QuantLastComplete'])
            qty_drive_position = position
            if position < 0:
                qty_gap = -position - traded_position
                if qty_gap > 0:
                    qty_pressure = logistic_prob(theta=factors['theta3'],
                                                 gamma=factors['gamma3'],
                                                 x=self.Traded)
                    qty_drive_position = qty_pressure * position

            # get the minimum/maximum position under pressure or not
            if position < 0:
                position_ = min([position, price_drive_position, qty_drive_position])
            else:
                position_ = max([position, price_drive_position, qty_drive_position])

            # NB: annual position cannot be easily changed!
            #     updated position for trading cannot exceed its annual position
            position_ = min([position_, self.AnnualPosition])
            self.MonthPosition[self.clock.Month] = round(position_, self.digits)

        # run `allocation` only after the reaction of firms to the market
        if self.ComplianceTrader:
            # TODO: will change it to be mode-based function
            position = self.compliance_allocator()
        else:
            position = self.non_compliance_allocator()

        self.MonthPosition = position
        # update daily position for trading (the last day is excluded so we plus 1)
        day_position = self.get_position('month') / (self.clock.workdays_left() + 1)
        self.fit(DayPosition=round(day_position, self.digits))

    def get_position(self, frequency='month'):
        """Get the current trade position (this month) or the day"""
        if frequency == 'day':
            return self.DayPosition
        else:  # `month` or the other frequency, return the monthly position
            return self.MonthPosition[self.clock.Month]

    def get_price(self, dist='logistic', **kwargs):
        """Get trading price and its expected price (ensure `Firm` object is added).
        Support `linear` or `logistic` probability distribution."""
        factors = self.Factors
        cur_price = self.CarbonPrice  # actual market price
        month_position = self.get_position('month')  # the distribution could be from 1 to 1000?

        # Sellers are not expecting carbon prices falling, but they do hope it could grow
        # The true logic is, when they hold more, they trade more aggressively
        if dist == 'logistic':
            gammai = factors['gammai'] - self.clock.Month / 9  # a bodge but it must be 9 (12 - 3, compliance period)
            gammai = gammai - 0.01 if gammai == 0 else gammai
            upper_prob = logistic_prob(theta=factors['thetai'], gamma=gammai, x=-month_position)
            lower_prob = logistic_prob(theta=factors['thetai'], gamma=factors['gammai'], x=-month_position)
        else:
            # TODO: the linear mode is not correct with its `speed` param
            prob = range_prob(month_position, 0.01) - 0.5
            upper_prob, lower_prob = prob + 1, 1 - prob

        # the final probability for the adjustment of carbon price
        if lower_prob <= upper_prob:
            prob = get_rand_vector(1, low=lower_prob, high=upper_prob).pop()  # price +- |eps|
        else:
            prob = get_rand_vector(1, low=upper_prob, high=lower_prob).pop()

        self.ExpectCarbonPrice = round(cur_price * prob, 6)
        return self.ExpectCarbonPrice

    def get_decision(self, realtime_price: float = 0.0, prob: float = 0.0, **kwargs):
        """Firm decides a transaction according to its trading needs, and firms react to variance in current price.

        :param realtime_price: the realtime carbon price in the market for firm's trading decision
        :param prob: the realtime probability for firm to trade (randomly determined)
        """
        factors = self.Factors
        position = self.get_position('month')
        if round(position, 5) == 0 and self.quit:
            logger.warning(f'Firm [{self.uid}] cleared position and quit the market.')
            return

        # use the realtime price to adjust firm's trading decision
        exp_price = self.ExpectCarbonPrice  # it changes only once a month
        realtime_price = exp_price if not realtime_price else realtime_price

        # allow a daily pricing variance from firms' side
        eps = logistic_prob(theta=factors['theta4'], gamma=factors['gamma4'], x=(exp_price - realtime_price))
        eps = abs(1 - eps) if abs(1 - eps) <= 0.1 else 0.1
        pr = 1 + get_rand_vector(1, low=-eps, high=eps).pop()  # eq: price +- |eps|

        # TODO: apply a better strategy to update firm's expected carbon price
        #       还有一种策略是，将orderbook中的订单序列和企业的预期关联起来，实时更新
        #       注意! 还有一种大宗协议交易的方式，一次性卖出或买入10万吨以上配额
        # agent trades on a daily basis, and it should be divided into even shares
        order = Order(price=exp_price * pr,
                      mode='sell' if position > 0 else 'buy',
                      quantity=self.DayPosition,
                      offerer_id=self.uid,
                      ts=self.clock.timestamp())
        # firms obtain a random digit to decide if they trade or not
        return order

    def trade(self, current_price, prob, dist='linear'):
        """Pass in the current carbon price and make a decision if doing bid/ask ..."""
        # self.ExpectCarbonPrice should be updated day by day
        if self.lock:
            return

        self.CarbonPrice = current_price if current_price else self.CarbonPrice
        self.get_price(dist)
        order = self.get_decision(current_price, prob)
        prob_ = range_prob(self.DayPosition, speed=0.05)
        return order if prob_ >= prob else None

    def clear(self, statement: Statement):
        """Transform an order and offset the position. The frequency of calling `clear` function is
        basically daily (not monthly). But `_run()` should be executed by month."""
        position = self.get_position('month')  # generally, traded quantity will not exceed the daily position
        # if order position exceeds the monthly position, it will be distributed to the coming months
        quant = statement.quantity  # quantity in Statement is always positive
        # update the Trade&Order record
        self.Traded += quant
        self.order += [statement.dict()]
        self.cache['CarbonPrice'] += [statement.price]
        # TODO: for experiment only
        # self.CarbonPrice = statement.price

        # measure cost/revenue from trading
        if position < 0:
            self.TradeCost += quant * statement.price
        else:
            self.TradeRevenue += quant * statement.price
            quant = -quant

        self.QuantLastComplete = quant
        self.MonthPosition[self.clock.Month] = position + quant  # opposite
        self.AnnualPosition += quant

        month_sum = sum(list(self.MonthPosition.values()))
        min_pos = min([self.DayPosition, self.AnnualPosition, month_sum])
        if self.lockable and abs(min_pos) < 0.001:  # 1 tonne
            self.lock = 1
            logger.warning(f'Firm [{self.uid}] is locked for trading.')

        if self.DayPosition >= self.AnnualPosition or self.DayPosition > month_sum:
            self.adjust_position(position=min_pos)
        else:
            # adjust month position, day position, leave annual position unchanged
            self.adjust_position()

        return self


class Production(BaseComponent):
    """Production module decides firm's production output and input. The instruction is as followed:
    1. `AnnualOutput/alpha1` controls the size of energy/material consumption, for instance:
        - `900/3000` (with r1=0.86, r2=0.88, r0=2.25) >> `12000/1600`
        - `1200/3000` (with the same rx) >> `2787/385`
        - proper range could be [0.
    2. `alpha2` controls the share between energy and material consumption
        - `900/3000` with alpha2=0.8 >> `111902/519`
        - `1200/3000` with alpha2=0.8 >> `26000/121`
    3. Set `alpha2` within [0.3, 0.7], otherwise it explodes
    4. The unit of AnnualOutput is 10,000 while the unit of energy/material input is 1 tonne
    5. Due to the unit of SynEnergyPrice, ProductPrice and MaterialPrice should be around 1~3
    """
    role: str = None  # buyer or seller and the difference is energy/output ratio
    unit: int = 10000  # regulated firms are relatively big enough

    # explicit factors
    EnergyInput: float = 0.0  # energy use
    MaterialInput: float = 0.0  # material use
    SynEnergyPrice: float = 0.0  # update by `Energy.get_synthetic_price`
    SynEmissionFactor: float = 0.0  # update by `Energy.get_synthetic_factor`

    AnnualOutput: float = 0.0  # annual output
    MonthOutput: float = 0.0  # monthly output (monthly output = annual output / 12)

    ProductPrice: float = 0.0  # product price index
    ProductRevenue: float = 0.0  # sales from products
    MaterialPrice: float = 0.0  # material price index
    MaterialCost: float = 0.0  # material cost
    CarbonPriceCache: list = []  # carbon price vector from Config or CarbonTrade.cache['CarbonPrice']

    # inherit from the external instances
    external: list = ['Factors']
    Factors: dict = {}  # get from `Production.Factors`
    nocache: list = ['Factors']

    def __init__(self, **config):
        super().__init__(**config)
        # TODO: adjust ProductPrice (PP) and MaterialPrice (MP) to control the role of firms
        #       PP ++, Output --, Energy ++, Emission ++
        #       MP ++, Output =, Energy --, Emission --
        #       In Factors, rho1 ++, energy -- ; rho2 ++, energy ++ (eps<0.01)
        if not self.role:
            return

        pb = get_rand_vector(1, 3, 100, 200, True).pop() / 1e4  # positive value
        pb = -pb if self.role == 'buyer' else pb
        self.Factors['rho1'] += pb
        self.ProductPrice = self.ProductPrice * (1 + pb)

    def _run(self, **kwargs):
        # `_run()` is not directly called. It will be called in `forward()`.
        self.get_production(syn_energy_price=self.SynEnergyPrice,
                            syn_emission_factor=self.SynEmissionFactor,
                            carbon_price_cache=self.CarbonPriceCache)

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

    def _get_inputs(self, syn_energy_price, material_price, mean_carbon_price, factors, syn_emission_factor,
                    pi0: float):
        # TODO: the input function is depreciated because when outputs grows, the inputs shrink.
        #       energy/material input should increase when output increases.
        # make sure rho0/rho1/rho2 have a bigger value, rho1 and rho2 are relatively fixed values
        rho0 = 1 - 1 / factors['sigma']

        # TODO: rho1 & rho2 are too important in deciding energy and material inputs
        #       it's better to constrain the variance at 0.01 level (<|0.1| in total)
        #       rho1 ++, energy -- ; rho2 ++, energy ++
        duff_rho1 = syn_energy_price - mean_carbon_price * syn_emission_factor
        diff_rho2 = material_price - mean_carbon_price * syn_emission_factor
        rho1 = logistic_prob(0.2, 2, duff_rho1) * self.Factors['rho1']
        rho2 = logistic_prob(0.2, 0.8, diff_rho2) * self.Factors['rho2']

        # if one of them is negative, sigma should change (make sure rho0 * rho1 or rho0 * rho2 is positive)
        if (rho1 * rho0 * rho2) < 0:
            rho0 = 1 - 1 / (- factors['sigma'])

        # compute the optimal inputs
        ue = self._get_energy_input(pi0, rho0, rho1, rho2, factors['alpha1'], factors['alpha2'])
        ui = self._get_material_input(pi0, rho0, rho1, rho2, factors['alpha1'], factors['alpha2'])
        return ue, ui

    def get_production(self,
                       syn_energy_price,
                       syn_emission_factor,
                       carbon_price_cache):
        """Determine the inputs and outputs of the firm

        :param syn_energy_price: synthetic energy price from Config or Energy instance
        :param syn_emission_factor: synthetic emission factor from Config or Energy instance
        :param carbon_price_cache: an array of carbon price from CarbonTrade.cache
        :param kwargs: optional kwargs in _run() and forward()
        """
        factors = self.Factors  # make persistence of variables for quick testing
        prod_price = self.ProductPrice
        material_price = self.MaterialPrice
        mean_carbon_price = mean(carbon_price_cache, self.digits)

        # NB. the initial output is given by the config but when Step>1, it's decided by product price
        if self.step > 0:
            last_output = self.cache['AnnualOutput'][-1]
            output = last_output * (prod_price - 1) ** factors['beta0']
            self.AnnualOutput = round(output, self.digits)
            self.MonthOutput = round(output / 12, self.digits)

        # infer energy and material input through a CGE function
        ue, ui = self._get_inputs(syn_energy_price,
                                  material_price,
                                  mean_carbon_price,
                                  factors,
                                  syn_emission_factor,
                                  self.AnnualOutput)
        self.EnergyInput = ue
        self.MaterialInput = ui
        self.ProductRevenue = self.ProductPrice * self.AnnualOutput
        self.MaterialCost = ui * self.MaterialPrice
        # TODO: only measure Scope 1&2 emissions regardless of `material use`


class Finance(BaseComponent):
    """Finance module accounts firm's cost and revenue"""

    # general specification
    AllowBankruptcy: bool = True  # if the firm is still eligible
    Status: int = 1  # 1=running, 0=ending
    VerifyDuration: int = 3  # the maximum duration of failure

    CostType: list = []  # a dict of costs with keys as types
    Cost: dict = {}  # a dict of costs with keys as types
    TotalCost: float = 0.0

    Revenue: dict = {}  # a dict of revenues with keys as types
    TotalRevenue: float = 0.0
    Profit: float = 0.0  # profit = TotalRevenue - TotalCost

    # inherit from instances
    external: list = ['Production', 'Energy', 'Abatement', 'CarbonTrade', 'Policy']
    Production: Any
    Energy: Any
    Abatement: Any
    CarbonTrade: Any
    Policy: Any

    nocache: list = ['Status', 'AllowBankruptcy', 'VerifyDuration', 'CostType']

    def __init__(self, **config):
        super().__init__(**config)
        # initiate firm's cost and revenue as its historic data

    def _run(self, **kwargs):
        self.get_cost()
        self.get_revenue()
        self.get_profit()
        self.verify()

    def verify(self):
        """Verify the firm if it's still eligible for the market"""
        if len(self.cache['Profit']) > self.VerifyDuration:
            verified = all([item < 0 for item in self.cache['Profit']])
        else:
            verified = False

        if self.AllowBankruptcy and verified:  # allow firms to bankrupt
            self.Status = 0  # 0=bankruptcy, 1=running (aligned to Firm.Status)

    def get_cost(self):
        """Get aggregate cost after adding the following costs up:
        EnergyCost, MaterialCost, AbateCost, TradeCost, ComplianceCost. """
        self.Cost = {'EnergyCost': self.Energy['EnergyCost'],
                     'MaterialCost': self.Production['MaterialCost'],
                     'AbateCost': self.Abatement['AbateCost'],
                     'ComplianceCost': self.Policy['ComplianceCost'],
                     'TradeCost': self.CarbonTrade['TradeCost']}
        self.TotalCost = round(sum(list(self.Cost.values())), self.digits)
        return self.TotalCost

    def get_revenue(self):
        """Get aggregate revenue after adding the following revenues up:
        ProductRevenue, CarbonRevenue """
        self.Revenue = {'ProductRevenue': self.Production['ProductRevenue'],
                        'TradeRevenue': self.CarbonTrade['TradeRevenue']}
        self.TotalRevenue = round(sum(list(self.Revenue.values())), self.digits)
        return self.TotalRevenue

    def get_profit(self):
        self.Profit = self.TotalRevenue - self.TotalCost
        return self.Profit


class Policy(BaseComponent):
    """Policy module controls industry-specific uncertainty and costs on firms"""
    # general setting
    Holding: float = 0.0  # holding of allowance after compliance
    ComplianceCost: float = 0.0  # penalty from compliance, if penalty>0, the firm is punished
    PenaltyRate: float = 0.0  # configured in config file, the rate of penalty (unit: 1000/tonne)

    # inherit from external instances
    external: list = ['Energy', 'CarbonTrade']
    Energy: Any
    CarbonTrade: Any

    nocache: list = ['PenaltyRate']

    def __init__(self, **config):
        super().__init__(**config)

    def _run(self, **kwargs):
        """Execute uncertainty cost on firm"""
        self.ComplianceCost, self.Holding = self.get_compliance()

    def get_compliance(self):
        """Verify firm's allowance position for compliance and return a tuple of (penalty, holding)"""
        carbon = self.CarbonTrade
        energy = self.Energy
        total_emission = sum(list(energy['Emission'].values()))
        # NB. for `.Trade`, negative means buyer, positive means seller
        holding = carbon['Traded'] + energy['Allocation']
        if holding >= total_emission:  # successful compliance
            # offset its position and allow banking of allowances to the next stage
            holding = holding - total_emission
            penalty = 0
        else:  # failed compliance
            # get penalty
            penalty = round(self.PenaltyRate * (total_emission - holding), self.digits)
            holding = 0

        return penalty, holding

    def get_subsidy(self):
        """Reserved function for subsidy"""
        pass


class Strategy(BaseComponent):
    """Create a strategy to generate firms with specific settings"""

    # NB. scales of production/energy use/emission are important

    def __init__(self):
        pass


if __name__ == '__main__':
    pass
