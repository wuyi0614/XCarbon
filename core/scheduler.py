# -*- encoding: utf-8 -*-
#
# Created at 03/11/2021 by Yi Wu, wymario@163.com
#
import pandas as pd

from tqdm import tqdm
from copy import deepcopy

from core.base import init_logger, read_config, Clock, shuffler
from core.stats import get_rand_vector, logistic_prob, mean
from core.firm import BaseFirm
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

    def __init__(self, market: BaseMarket, clock: Clock, freq=None, psize=None, show=False, **kwargs):
        """"""
        # general specification
        self.show = show
        self.Frequency = market.Frequency if not freq else freq

        # agent pool
        self.Pool = {}  # a dict-like pool for agents: {'uid': <Firm Object>}
        self.Size = 0  # number of agents in the pool
        self.PSize = 1 if not psize else psize  # a future control param for async operation

        # adjustment factors (opt params are given)
        self.Factors = {'theta1': kwargs.get('theta1', 1.5), 'gamma1': kwargs.get('gamma1', 0.2),
                        'theta3': kwargs.get('theta3', 2), 'gamma3': kwargs.get('gamma3', 0.2)}

        # equip with market
        self.Market = market

        # status monitoring
        self.Clock = clock
        self.is_stop = True
        self.Step = 0  # loop status record

    def __getitem__(self, uid):
        self.Pool.get(uid)

    def _take(self, agent: BaseFirm):
        self.Pool[agent.uid] = agent
        if self.show:
            logger.info(f'Take firm [{agent.uid}] into the Pool')

    def take(self, *agents):
        """Pass agents into the Pool"""
        for item in agents:
            self._take(item)
            self.Size += 1

        return self

    # the following functions are tailored for firms
    def daily_clear(self, prob_range=None, **kwargs):
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
        clock = self.Clock

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

            # order book should be cleared every end of day
            pool = self.Pool
            for _, agent in pool.items():
                if not cm.to_dataframe().empty:
                    df = cm.to_dataframe()
                    high, mean_, low = df['high'].values[-1], df['mean'].values[-1], df['low'].values[-1]
                else:
                    high, mean_, low = None, None, None

                item = agent.trade(prob, high, mean_, low)
                if item:
                    ob.put(item)

            # feed agents with accepted orders
            for _, state in ob.pool.items():
                state = Statement(**state)
                orid, oeid = state.offerer_id, state.offeree_id
                self.Pool[orid] = self.Pool[orid].clear(state)
                self.Pool[oeid] = self.Pool[oeid].clear(state)

            # feed the order book into the market and update the market status
            market.close(**kwargs)  # ob is internally linked and updated
            logger.info(f'Carbon Market at Day {clock.today()} ...')

    def monthly_clear(self):
        """Clear status of agents once a month (use timestamp other than the count)"""
        # firms go through a clear every end of a month, and the steps are:
        # (1) clear the market as usual
        # (2) put carbon price back to agents so that they could make decisions based on that
        if not self.Clock.is_monthend():
            return self

        # use market information to give feedback to firms
        month_bar = self.Market.aggregate('month')
        if not month_bar:
            logger.info(f'The trading month had no transactoin at {self.Clock.today()}')
            self.Clock.move(1)
            return self

        for _, agent in self.Pool.items():
            carbon = agent.CarbonPrice
            agent.CarbonPrice.update({'carbon-price': carbon['carbon-price'] + [month_bar.mean]})
            agent._action_trade_decision()
            agent.Step += 1

        self.Clock.move(1)
        logger.info(f'The trading month is ended at {self.Clock.today()}')
        return self

    def yearly_clear(self):
        """Clear status of agents once a year"""
        if not self.Clock.is_yearend():
            return self

        # adjust product price by comparing aggregate output of last year and this year
        last_output, current_output, price = [], [], []
        # adjust abatement cost by calculating the overall abatement volume
        allocation, abate, emission = [], [], []
        abate_cost = []
        for _, agent in self.Pool.items():
            last_output += [agent.Output['last-output']]
            current_output += [agent.Output['annual-output']]
            price += [agent.Price['product-price']]

            allocation += [agent.Allocation['allocation']]
            abate += [agent.Abatement['abate']]
            abate_cost += [agent.Abatement['abate-cost']]
            emission += [agent.Emission['emission-all']]

        # TODO: update product price (if there are more industries, categorize them by `Tag`)
        # product price is negatively correlated with output
        factors = self.Factors
        updated_price = mean(price) * logistic_prob(theta=factors['theta3'],
                                                     gamma=factors['gamma3'],
                                                     x=sum(current_output) - sum(last_output))
        # abatement cost (more abate, higher cost)
        # TODO: if it proceeds at element level, it enables to adjust one firm's abate cost
        gap = sum(allocation) - sum(emission) - sum(abate)
        updated_cost = [cost * logistic_prob(theta=factors['theta1'], gamma=factors['gamma1'], x=gap) for
                        cost in abate_cost]
        # update agent's status
        for idx, (_, agent) in enumerate(self.Pool.items()):
            price = deepcopy(agent.Price['product-price'])
            agent.Price.update({'product-price': updated_price, 'last-product-price': price})

            cost = deepcopy(agent.Abatement['abate-cost'])
            agent.Abatement.update({'abate-cost': updated_cost[idx], 'last-abate-cost': cost})

            # summarize firms' trading revenue
            deal = pd.DataFrame(agent._Orders['deal'])
            if deal.empty:
                revenue = 0
            else:
                vol = deal.price.values * deal.quantity.values
                revenue = vol.sum() if agent.Position['daily-trade-position'] > 0 else - vol.sum()

            rev = deepcopy(agent.Revenue['carbon-revenue'])
            agent.Revenue.update({'carbon-revenue': revenue, 'last-carbon-revenue': rev})
            # TODO: fine / production revenues

        self.Clock.move(1)
        logger.info(f'The trading year is ended at {self.Clock.today()}')
        return self

    def _run(self, tag):
        for uid, agent in tqdm(self.Pool.items(), desc=tag):
            # agent is internally mapped in memory
            agent.activate()

        return self

    def run(self, **kwargs):
        """Activate scheduler for agents"""
        self.is_stop = False
        if not self.Pool:
            logger.warning(f'Scheduler detect {len(self.Pool)} agents in the pool')

        # returned results are projected agent objects
        # TODO: call `accelerator` is not doable because of pickle limitation
        self._run(f'Scheduler Step={self.Step}')

        # run schedules
        probs = kwargs.get('probs', [0, 1])
        compress = kwargs.get('compress', False)

        while not self.Clock.is_yearend():
            self.daily_clear(prob_range=probs, compress=compress)
            if self.Clock.is_monthend() and not self.Clock.is_yearend():
                self.monthly_clear()

        # end of loop (monthly_clear is added for December)
        self.monthly_clear()
        self.yearly_clear()
        return self

    def reset(self, **kwargs):
        """Empty the Pool and reset the step"""
        self.__init__(self.Market.reset(**kwargs), **kwargs)
        return self


if __name__ == '__main__':
    from core.firm import RegulatedFirm, randomize_firm_specs
    from core.market import CarbonMarket, OrderBook
    from core.base import Clock
    from core.plot import plot_kline

    # Test: daily clear
    # create objs
    ck = Clock('2021-01-01')
    ob = OrderBook('day')
    cm = CarbonMarket(ob, ck, 'day')

    # create scheduler
    sch_conf = read_config('config/scheduler-20211027.json')
    sch = Scheduler(cm, ck, 'day', **sch_conf)

    # create agents
    conf = read_config('config/power-firm-20211027.json')
    for i in range(10):
        conf_ = randomize_firm_specs(conf, True)
        reg = RegulatedFirm(ck, **conf_)
        sch.take(reg)

    # activate agents and offer trades
    sch._run('Scheduler')
    sch.daily_clear([0.2, 0.6], compress=True)

    # Test: monthly clear
    sch.monthly_clear()
    sch.daily_clear([0.3, 0.7], compress=True)

    # Test: the pipeline
    sch_conf = read_config('config/scheduler-20211027.json')
    sch = Scheduler(cm, ck, 'day', **sch_conf)

    # create agents
    conf = read_config('config/power-firm-20211027.json')
    for i in range(50):
        conf_ = randomize_firm_specs(conf, True)
        reg = RegulatedFirm(ck, **conf_)
        sch.take(reg)

    sch.run(probs=[0., 0.5], compress=True)

    mark_report = cm.to_dataframe()
    array = mark_report[['open', 'close', 'low', 'high']].astype(float).values.tolist()
    dates = mark_report.ts.values.tolist()

    plot_kline(array, dates, 'carbon price', 'carbon-market-year')
