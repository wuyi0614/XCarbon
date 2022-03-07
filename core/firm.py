# -*- encoding: utf-8 -*-
#
# Created at 27/10/2021 by Yi Wu, wymario@163.com
#

"""
Specify firm-like agents given theories from economics
"""

from copy import deepcopy
from random import sample

from core.base import generate_id, init_logger, read_config, Clock, BaseComponent
from core.stats import get_rand_vector
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
class BaseFirm(BaseComponent):
    """A base firm has the following attributes and behaviors and it also evolves with the time goes."""

    unit: int = 1000
    status: int = 1  # 1=activated, 0=deactivated
    clock: object = None
    Attribute: dict = {}

    def __init__(self, clock: Clock, **config):
        super().__init__(**config)
        # the following are unchangeable attributes for a firm, e.g. `uid` is unique and fixed
        self.uid = config.get('uif', generate_id())
        self.unit = 1000  # the unit of economic value
        self.clock = clock
        self.fit(**config)

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

    compliance = 0  # 1=compliance trader; 0=non-compliance trader
    Trader: int = 1  # 1=trader, 0=quitter

    # the objects here are callable instances
    external: list = ['Energy', 'CarbonTrade', 'Production', 'Abatement', 'Policy', 'Finance']
    Energy: object = None
    CarbonTrade: object = None
    Production: object = None
    Abatement: object = None
    Policy: object = None
    Finance: object = None

    # orders/statements from carbon trade
    Orders: list = []

    def __init__(self, clock, **config):
        super(RegulatedFirm, self).__init__(clock, **config)
        # set up parameters
        self.fit(**config)

    def activate(self):
        """Activate the firm by forwarding its instances"""
        for obj in self.external:
            # the first-time forward is activation and account of historic data
            # NB. CarbonTrade will be executed day by day
            self.__getattribute__(obj).forward()

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

    def daily_clear(self, statement):
        """Clear actions at a daily basis:
        CarbonTrade: clear daily transaction
        """
        self.CarbonTrade.clear(statement)

    def monthly_clear(self, mean_carbon_price):
        """Clear actions executed at a monthly basis:
        CarbonTrade: offset position and adjust carbon price by the average market price
        """
        self.CarbonTrade.CarbonPrice = mean_carbon_price
        self.CarbonTrade.forward()

    def yearly_clear(self, prod_price, mat_price, carbon_price):
        """Clear actions executed at a yearly basis. Backward broadcast the Finance status:

        Scheduler:
        - ProductPrice
        - MaterialPrice
        - CarbonPrice

        Instances:
        - Finance: pass `Profit` to Production and adjust its production plan
        - Production:
        - Energy: adjust `EnergyUse`
        - Production: adjust Input and Output

        Parameters updated every year in the configuration:
        - EnergyInput
        -
        """
        # NB. use `forward` will automatically make caches of instances
        self.Production.fit(ProductPrice=prod_price, MaterialPrice=mat_price,
                            CarbonTradeCache=self.CarbonTrade.cache)
        self.Production.forward()

        # pass carbon price back into abatement



class EngagedFirm(BaseFirm):
    """Engaged firms are a bit different from regulated firms because they make emission mitigation plans first,
    and then calculate their production inputs. Lastly, actual emissions of firms will be computed.
    It is updated at 29th Jan. 2022 by Yi.

    When inheriting from `RegulatedFirm`, only need limited rewriting, i.e. `__init__()`
    """

    def __init__(self, **kwargs):
        # inherit from the target instance
        super().__init__(clock, **kwargs)

    def _action_abatement(self, avg_abate_cost: float):
        """
        The emission mitigation options have two types:
        1. negative emission technology such as CCUS, carbon sink
        2. energy saving technology such as energy efficiency improvement

        The return data of this function is the probability of the firm to invest in an
        emission mitigation plan with actual emission reduction as well as its cost.

        The probability for the firm to make a decision is highly dependent on its industry.
        """
        abate, cost = 0, 0
        #

        return abate, cost


if __name__ == '__main__':
    from core.base import Clock
    clock = Clock('2021-01-01')

    # Temporary test zone for all the classes and functions
    # Test: Agent test
    firm1 = BaseFirm(name='f1')
    firm1.cache()

    # Test: create a regulated firm and run single-node test
    conf = read_config('config/power-firm-20211027.json')
