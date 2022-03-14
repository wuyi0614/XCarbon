# -*- encoding: utf-8 -*-
#
# Created at 03/11/2021 by Yi Wu, wymario@163.com
#

from tqdm import tqdm
from collections import defaultdict

from core.base import init_logger, Clock, shuffler
from core.stats import get_rand_vector, logistic_prob
from core.market import BaseMarket, Statement

# init a logger
logger = init_logger('Scheduler', level='WARNING')


# scheduler
class Scheduler:
    """
    A scheduler collects agents in its pool and set off them in the sync/async way,
    and a BaseScheduler is developed for `BaseFirm` to run demo

    with the following functions:

    take: pipe an agent into its pool
    """

    def __init__(self,
                 market: BaseMarket,
                 clock: Clock,
                 freq=None,
                 psize=None,
                 show=False,
                 **kwargs):
        # general specification
        self.show = show
        self.Frequency = market.Frequency if not freq else freq

        # agent pool
        self.Pool = {}  # a dict-like pool for agents: {'uid': <Firm Object>}
        self.Size = 0  # number of agents in the pool
        self.PSize = 1 if not psize else psize  # a future control param for async operation

        # equip with market
        self.Market = market

        # data monitoring
        self.Data = {'abate': 0, 'output': 0, 'emission': 0, 'allocation': 0}

        # reference data for fitting: configured in the scheduler specification file
        self.FittingData = {
            # 'energy': kwargs.get('energy_data', {}),
            # 'product': kwargs.get('product_data', {}),
            # 'material': kwargs.get('material_data', {}),
            # 'abatement': kwargs.get('abatement_data', {})
        }

        # status monitoring
        self.clock = clock
        self.is_stop = True
        self.step = 0  # loop status record

    def __getitem__(self, uid_or_idx):
        if isinstance(uid_or_idx, str):
            return self.Pool.get(uid_or_idx)
        elif isinstance(uid_or_idx, int):
            uid = list(self.Pool.keys())[uid_or_idx]
            return self.Pool.get(uid)
        else:
            return

    def __repr__(self):
        return f"Scheduler with {self.Size} firms at step={self.step}"

    def take(self, *agents):
        """Pass agents into the Pool"""
        for agent in agents:
            self.Pool[agent.uid] = agent
            if self.show:
                logger.info(f'Take firm [{agent.uid}] into the Pool')

            self.Size += 1

    # the following functions are tailored for firms
    def daily_clear(self, prob_range=None, show=False, **kwargs):
        """Clear market on a daily basis (trading_days=20)"""
        # TODO: in the future, we may support a calendar execution
        # OrderBook's `ts` should be added with one more day
        # The logic is:
        # (1) every firm offers their trade orders (after shuffling),
        # (2) clear the order book and summarize the day,
        # (3) update the market by day X,
        # (4) adjust the carbon price, traded position of agents,
        # (5) enter the next period.

        # initiate the daily market with a prob
        if not prob_range:
            prob_range = [0.5, 1]

        # create order book and launch the market
        market = self.Market
        ob = market.OrderBook
        clock = self.clock

        # use clock to mock a real-world time flow
        while not clock.is_monthend():
            # if it's not a workday, skip the day
            clock.move(1)
            # a new day, a new order of agents
            self.Pool = shuffler(self.Pool)

            if not clock.is_workday():
                logger.info(f'The market is not open today {clock}')
                continue

            prob = get_rand_vector(1, low=prob_range[0], high=prob_range[1]).pop()
            market.open()

            # TODO: allow firms to trade once a day or unlimited trading depending on the
            #       time steps
            pool = self.Pool
            for _, agent in pool.items():
                if not market.to_dataframe().empty:
                    df = market.to_dataframe()
                    mean_ = df['mean'].values[-1]
                else:
                    mean_ = None

                item = agent.trade(mean_, prob)
                if item:
                    ob.put(item)

            # feed agents with accepted orders
            for _, state in ob.pool.items():
                state = Statement(**state)
                orid, oeid = state.offerer_id, state.offeree_id
                self.Pool[orid] = self.Pool[orid].forward_daily(state)
                self.Pool[oeid] = self.Pool[oeid].forward_daily(state)

                if show:
                    logger.info(f'Transaction clear: {orid}>{oeid} at {state.price} with {state.quantity} units.')

            # feed the order book into the market and update the market status
            market.close()  # ob is internally linked and updated
            logger.info(f'Carbon Market at Day {clock.today()} ...')

    def monthly_clear(self, **kwargs):
        """Clear status of agents once a month (use timestamp other than the count)"""
        # firms go through a clear every end of a month, and the steps are:
        # (1) clear the market as usual
        # (2) put carbon price back to agents so that they could make decisions based on that
        # TODO: enable monthly update of abatement cost and thus carbon price will be anchored in that cost

        if not self.clock.is_monthend():
            return self

        # use market information to give feedback to firms
        month_bar = self.Market.aggregate('month')
        if not month_bar:
            logger.info(f'The trading month had no transactoin at {self.clock.today()}')
            self.clock.move(1)
            return self

        self.clock.move(1)
        for _, agent in self.Pool.items():
            agent.forward_monthly(month_bar.mean)
            agent.step += 1

        logger.info(f'The trading month is ended at {self.clock.today()}')
        return self

    @staticmethod
    def prob_pricing(last, now, which=None):
        """ Adjust prices probabilistically according to their last/current values

        :param last: values in the last period
        :param now: values in the current period
        :param which: which product you want to price, default as None (normal params)
        """
        # NB: for product/energy prices, there could be shocks in prices but unless it is asked,
        #     we do not significantly allow them to fluctuate. So they should have a lower theta & gamma
        #     The core logic here is more supply, lower the price.
        is_up = last < now  # if the value is increasing, True=yes, False=no
        if any([now == 0, last == 0]):
            return 1

        r = now / last
        if which in ['product', 'energy']:  # slow pace
            param = dict(gamma=0.6, theta=1.2)  # doubling the value brings 5% growth
        elif which == 'abatement':  # DO NOT be too aggressive before improving it
            param = dict(gamma=0.6, theta=1.5)  # doubling the value brings 10% growth
        else:
            param = dict(gamma=0.8, theta=1.8)  # doubling the value brings 20% growth

        return logistic_prob(x=r, **param) if is_up else 1 / logistic_prob(x=r, **param)

    def yearly_clear(self, **kwargs):
        """ Scheduler's yearly clearance involves the following updates:

        - Production: ProductPrice, MaterialPrice,
        - Energy: EnergyPrice,
        - CarbonTrade: CarbonPrice
        - Abatement: AbateOption (AbatePrice)
        - ....

        Acceptable data inputs from `kwargs` (add more dimensions if there is regional differences):
        - energy price data: {'2021': {'coal': 0.xxx, 'gas': 0.yyy}, '2022': {...}, ...}
        - material price data: {'2021': xxxx, '2022': yyy ...}
        - product price data: {'2021': {'power': 0.xxx, 'iron': 0.yyy}, '2022': {...}, ...}
        - abatement price data: {'2021': {'efficient': 0.xxx, 'shift': 0.yyy}, '2022': {...}, ...}
        """
        # statistics
        stat = defaultdict(list)
        # product-based information
        output = defaultdict(list)
        material = defaultdict(list)
        # energy-based information
        energy = defaultdict(list)
        # abatement information
        abate = defaultdict(list)

        # TODO: allowance allocation and transactions in the last year could significantly alter
        #       the expectations of firms on carbon price / trading strategies
        for _, agent in self.Pool.items():
            # collect output info
            output['now'] += [agent.Production.AnnualOutput]
            output['last'] += [agent.Production.cache['AnnualOutput'][-1]]
            # collect material info
            material['now'] += [agent.Production.MaterialInput]
            material['last'] += [agent.Production.cache['MaterialInput'][-1]]
            # collect energy info
            for typ, v in agent.Energy.InputEnergy.items():
                energy[f'now-{typ}'] += [v]
                energy[f'last-{typ}'] += [agent.Energy.cache['InputEnergy'][-1][typ]]

            # collect abatement info
            # TODO: the structure is {'Efficiency': {'abate': ..., 'cost': ...}}.
            #       In the future, must link the abatement cost with specific energy and technology
            abate['now'] += [agent.Abatement['TotalAbatement']]
            abate['last'] += [agent.Abatement.cache['TotalAbatement'][-1]]
            # collect statistical info
            stat['profit'] += [agent.Profit]
            stat['allocation'] += [agent.Allocation]
            stat['emission'] += [agent.TotalEmission]
            stat['abatement'] += [agent.Abate]
            stat['traded'] += [agent.Traded]

        # TODO: support externally-imported data for fitting but the example codes are incorrect,
        #       the right way is `self.FittingData['energy'][self.clock.year]`
        # according to the value changes, adjust the prices (use the last agent)
        if 'energy' in self.FittingData:
            updated_energy_price = self.FittingData['energy']
        else:
            energy_price = agent.Energy.EnergyPrice
            updated_energy_price = {}
            for etype, price in energy_price.items():
                prob = self.prob_pricing(sum(energy[f'last-{etype}']), sum(abate[f'now-{etype}']))
                updated_energy_price[etype] = price * prob

        if 'product' in self.FittingData:
            product_price = self.FittingData['product']
        else:
            product_price = agent.Production.ProductPrice
            product_price = product_price * self.prob_pricing(sum(output['last']), sum(output['now']))

        if 'material' in self.FittingData:
            material_price = self.FittingData['material']
        else:
            material_price = agent.Production.MaterialPrice
            material_price = material_price * self.prob_pricing(sum(material['last']), sum(material['now']))

        if 'abatement' in self.FittingData:
            updated_abate_opt = self.FittingData['abatement']
        else:
            abate_opt = agent.Abatement.AbateOption
            prob = self.prob_pricing(sum(abate['last']), sum(abate['now']))
            updated_abate_opt = {}
            for tech, sub in abate_opt.items():  # efficiency, shift, negative
                new_sub = {}
                for typ, v in sub.items():  # shift/energy/ccs
                    v['price'] = v['price'] * prob
                    new_sub[typ] = v

                updated_abate_opt[tech] = new_sub

        if 'carbon' in self.FittingData:
            carbon_price = self.FittingData['carbon_price']
        else:
            carbon_price = round(self.Market.to_dataframe()['mean'].mean(), 6)

        # monitor output, emission, abatement, allowance allocation status in the system
        data = {k: sum(v) for k, v in stat.items()}
        self.Data.update(data)

        # TODO: update product price (if there are more industries, categorize them by `Tag`)
        # update agent's status
        for idx, (_, agent) in enumerate(self.Pool.items()):
            agent.forward_yearly(prod_price=product_price,
                                 mat_price=material_price,
                                 energy_price=updated_energy_price,
                                 carbon_price=carbon_price,
                                 abate_option=updated_abate_opt)

        logger.info(f'The trading year is ended at {self.clock.today()}')
        return self

    def _run(self, tag):
        agents = list(self.Pool.items())
        for uid, agent in tqdm(agents, desc=tag):
            # agent is internally mapped in memory
            agent.activate()
            if agent.Energy.get_total_emission() < self.Market.Threshold:
                self.Pool.pop(uid)
                self.Size -= 1
                logger.warning(f'Found firm [{uid}] not reached the threshold.')

        return self

    def run(self, **kwargs):
        """Activate scheduler for agents"""
        self.is_stop = False
        if not self.Pool:
            logger.warning(f'Scheduler detect {len(self.Pool)} agents in the pool')

        # returned results are projected agent objects
        # TODO: call `accelerator` is not doable because of pickle limitation
        self._run(f'Scheduler Step={self.step}: ')

        # run schedules
        probs = kwargs.get('probs', [0, 1])

        while not self.clock.is_end():
            self.daily_clear(prob_range=probs, **kwargs)
            if self.clock.is_monthend() and not self.clock.is_yearend():
                self.monthly_clear(**kwargs)

        # end of loop (monthly_clear is added for December)
        self.monthly_clear(**kwargs)
        self.yearly_clear(**kwargs)
        return self

    def reset(self, **kwargs):
        """Empty the Pool and reset the step"""
        self.__init__(self.Market.reset(**kwargs), **kwargs)
        return self


if __name__ == '__main__':
    pass
