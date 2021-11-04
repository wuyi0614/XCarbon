# -*- encoding: utf-8 -*-
#
# Created at 03/11/2021 by Yi Wu, wymario@163.com
#
from tqdm import tqdm

from core.base import init_logger, read_config, Clock, shuffler
from core.stats import get_rand_vector
from core.firm import BaseFirm
from core.market import BaseMarket, Statement


# init a logger
logger = init_logger('Scheduler')


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

    def _run(self, tag):
        for uid, agent in tqdm(self.Pool.items(), desc=tag):
            # agent is internally mapped in memory
            agent.activate()

    def run(self):
        """Activate scheduler for agents"""
        self.is_stop = False
        if not self.Pool:
            logger.warning(f'Scheduler detect {len(self.Pool)} agents in the pool')

        # returned results are projected agent objects
        # TODO: call `accelerator` is not doable because of pickle limitation
        self._run(f'Scheduler Step={self.Step}')
        return self

    # the following functions are tailored for firms
    def _adj_product_price(self):
        """Product price adjustment happens at a yearly basis"""

        pass

    def _adj_carbon_price(self):
        """Carbon pricing happens every day and then it adjusts carbon price on a daily basis

        The execution steps are:
        - clear deals and update agents' position (with traded position)
        - adjust carbon price
        - run agent's function ._action_trade_decision
        - run agent's trade
        """

        pass

    def _adj_abate_cost(self):
        """Abatement cost adjustment happens at a yearly basis but notably the abatement decision happens monthly"""

        pass

    def pricing(self):
        """After all the firms have finished production planning, scheduler may adjust the market price for product"""
        # 调整产品价格，根据市场的实际产量和上一期的产量关系

        # 可以考虑构建一个变量内部的映射，这样只需要在scheduler内部调整，实际的值就会变化

        # 减排成本要以全部企业的实际减排量为度量，逐步提升

        pass

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
                    logger.info(f'Firm [{agent.uid}] raised an order {item.dict()}')
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
        pass

    def yearly_clear(self):
        pass

    def reset(self, **kwargs):
        """Empty the Pool and reset the step"""
        self.__init__(self.Market.reset(**kwargs), **kwargs)
        return self


if __name__ == '__main__':
    from core.firm import RegulatedFirm, randomize_firm_specs
    from core.market import CarbonMarket, OrderBook
    from core.base import Clock
    from core.plot import plot_kline

    # create objs
    ck = Clock('2021-07-01')
    ob = OrderBook('day')
    cm = CarbonMarket(ob, ck, 'day')
    sch = Scheduler(cm, ck, 'day')

    # create agents
    conf = read_config('config/power-firm-20211027.json')
    for i in range(50):
        conf_ = randomize_firm_specs(conf, True)
        reg = RegulatedFirm(ck, **conf_)
        sch.take(reg)

    # activate agents and offer trades
    sch.run()
    sch.daily_clear([0.4, .8], compress=True)

    mark_report = cm.to_dataframe()
    array = mark_report[['open', 'close', 'low', 'high']].astype(float).values.tolist()
    dates = mark_report.ts.values.tolist()

    plot_kline(array, dates, 'carbon price', 'carbon-market')
