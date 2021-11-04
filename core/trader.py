# The script for simulations of agents
from typing import Optional
from core.base import get_rand_vector, sampling, generate_id, timer, get_current_timestamp, logger
from core.agent import Order, Statement, Bar, BaseModel, OrderedDict


class Decision(BaseModel):
    """A base agent decision instance"""
    # capital
    capital: Optional[float]
    # marginal abatement cost
    mac: Optional[float]
    # units of emissions
    emission: Optional[float]
    # units of allowance, ie. one unit of quota for one equivalent tonne of GHG
    quota: Optional[float]
    # units of allowance in holding
    holding: Optional[int]
    # order mode: `buy` or `sell` depending on the other parameters
    mode: Optional[str]
    # order quantity
    quantity: Optional[int]
    # order price
    price: Optional[float]

    def __init__(self,
                 capital=None,
                 mac=None,
                 emission=None,
                 quota=None,
                 holding=None,
                 mode=None,
                 quantity=None,
                 price=None,
                 **specs):
        """initiate a decision instance"""
        super(Decision, self).__init__(**specs)

        self.capital = get_rand_vector(1, low=100000, high=1000000, is_int=True).pop() if capital is None else capital
        self.mac = get_rand_vector(1, low=10, high=50).pop() if mac is None else mac
        self.emission = get_rand_vector(1, low=1000, high=10000, is_int=True).pop() if emission is None else emission

        ratio = get_rand_vector(1, low=0.8, high=1.2).pop()
        self.quota = int(ratio * self.emission) if quota is None else quota
        self.holding = 0 if holding is None else holding
        self.mode = "" if mode is None else mode
        self.quantity = 0 if quantity is None else quantity
        self.price = 0 if price is None else price


class BasePricingMixin:
    """A basic pricing method ',n ,./n ./jklwhich takes in Decision instance and emits a numeric price,
    future development of pricing methods can derive from this class.
    """

    def __init__(self, **specs):
        self.market_price = specs.get("market_price", get_rand_vector(1, low=10, high=50).pop())
        self.mode = specs.get("mode")

    def __call__(self, item: Decision) -> Decision:
        """Make it callable for decision instance"""
        if self.mode == "buy":
            item.price = item.price if item.price < item.mac else item.mac
        else:  # `sell`
            item.price = item.price if item.price > item.mac else item.mac

        return item


class BaseQuantifyMixin:
    """A basic quantify method which takes in Decision instance and emits an integer quantity,
    future development of quantify methods can derive from this class.
    """

    def __init__(self, **specs):
        pass

    def __call__(self, item: Decision) -> Decision:
        """Make it callable for decision instance"""
        # simplified quantification of trading
        quantity = get_rand_vector(1, low=100, high=1000, is_int=True).pop()
        if item.mode == "buy":
            if item.capital < (item.price * quantity):
                quantity = int(item.capital / item.price) if item.capital > 0 else 0

            item.quantity = quantity

        else:  # `sell`
            if item.holding < quantity:
                quantity = item.holding

            item.quantity = quantity

        return item


class BaseAgent:
    """A base agent instance"""

    def __init__(self, **specs):
        # /// general spec ///
        self.id = specs.get("agent_id", generate_id())
        # specify agent account's details including capital, mac, emission, quota, holding, etc.
        self.base = specs.get("base", Decision().dict())
        self.orders = OrderedDict()

        # /// strategy spec ///
        # strategy pool, including four types of strategies:
        self.spool = {
            # 1) excess supplier, firms who hold excessive allowances and willing to trade
            # 2) in-short buyer, firms who need to buy allowances for compliance
            "pricing": specs.get("pricing", BasePricingMixin),
            "quantifying": specs.get("quantifying", BaseQuantifyMixin)
        }

        # /// statistics features ///
        # None yet

        # /// systemic spec ///
        # the timer (wait-time) pauses agent's action before it's released after the `timer`
        self.timer = specs.get("timer", get_rand_vector(1).pop())

    def __str__(self):
        return str(self.base)

    def __repr__(self):
        return str(self.base)

    def _get_price(self, bar, instance: BasePricingMixin, decision: Decision) -> Decision:
        """
        Generate the time-point price of the agent depending on the specification of pricing

        :param bar      : the market dynamics
        :param instance : a function instance for pricing(callable and replaceable)
        :param decision : a decision instance
        """
        func = instance(**bar.dict())
        return func(decision)

    def _get_quantity(self, bar, instance: BaseQuantifyMixin, decision: Decision) -> Decision:
        """
        Generate the time-point quantity of the agent depending on the specification of demanding / supplying

        :param bar      : the market dynamics
        :param instance : a function instance for quantifying(callable and replaceable)
        :param decision : a batch of parameters that determines the quantity
        """
        func = instance(**bar.dict())
        return func(decision)

    @staticmethod
    def timestamp():
        return get_current_timestamp("%Y-%m-%d %H:%M:%S")

    def get_decision(self, bar: Bar, mode=None) -> Decision:
        self.base["mode"] = mode
        item = Decision(**self.base)
        # simply replace the Decision.price by Bar.price (.mean)
        # TODO: mock decision function and connect Bar data to decisions
        item.price = bar.mean
        item = self._get_price(bar, self.spool["pricing"], item)
        item = self._get_quantity(bar, self.spool["quantifying"], item)
        return item

    def get_order(self, item: Decision):
        # convert Decision object into Order
        return Order(price=item.price,
                     mode=item.mode,
                     quantity=item.quantity,
                     offerer_id=self.id,
                     ts=self.timestamp())

    def bid(self, bar: Bar, **params) -> Order:
        """
        Agent's `bid` action to purchase allowances from the market with an offer price called by .run()

        :param bar: Bar instance which reflects the market dynamics
        :param params: the other parameters that determines the `bid` action
        """
        # with Bar data, agent makes decisions
        item = self.get_decision(bar, "sell")
        order = self.get_order(item)
        self.orders[order.id] = order.dict()
        logger.info("[Order] Agent:{} bids {:,} shares at {:3f}".format(self.id, order.quantity, order.price))
        return order

    def ask(self, bar: Bar, **params) -> Order:
        """
        Agent's `ask` action to sell allowances from the market with an offer price called by .run()

        :param bar   : the market price at the moment
        :param params: the other parameters that determines the `ask` action
        """
        item = self.get_decision(bar, "buy")
        order = self.get_order(item)
        self.orders[order.id] = order.dict()
        logger.info("[Order] Agent:{} asks {:,} shares at {:3f}".format(self.id, order.quantity, order.price))
        return order

    def _year_calibrate(self):
        """
        Annual calibration for agents including those parameters don't happen at a daily base
        - abatement decision
        - marginal abatement cost decision
        - emission volume
        - fine / cost / benefits
        ...
        """
        pass

    def forward(self, **params):
        """
        Learning process: forward parameters from trade or the market to the next step (for Reinforcement)

        :param params: the other parameters that adjust the learning process
        """
        pass

    def backward(self, **params):
        """
        Learning process: backward parameters from trade or the market to adjust the agent's actions (for Reinforcement)

        :param params: the other parameters that adjust the learning process
        """
        pass

    def deal(self, state: Statement):
        """
        Clear offered transactions by receiving statements

        :param state: a Statement instance that is valid or not
        """
        # TODO: adjust agent's `base` parameter (Decision parameters), not only capital / holding
        ids = state.id.split("+")
        for i in ids:
            if i in self.orders:
                order = self.orders[i]
                if order["mode"] == "sell":
                    self.base["capital"] += state.price * state.quantity
                    self.base["holding"] -= state.quantity
                    assert self.base["quantity"] < 0, "invalid transaction: %s" % self.id
                else:
                    self.base["holding"] += state.quantity
                    self.base["capital"] -= state.price * state.quantity
                    assert self.base["capital"] < 0, "invalid transaction: %s" % self.id

        return self

    def run(self, bar: Bar):
        """
        Runtime & parallel process: agent's actions will periodically happen after `run` in threads

        :param bar: a Bar instance reflects the status and dynamics of the running market
        """
        # ... waiting for a random time interval
        timer(self.timer)
        # react to the real-time Bar dynamics (bar)
        orders = []
        orders += self.bid(bar)
        orders += self.ask(bar)

        # `forward` & `backward` gains or losses to adjust agent's specs
        self.forward()
        self.backward()

        return orders


class ComplianceTrader:
    """
    Compliance traders buy / sell allowances for the purpose of fulfillment
    """
    pass


class NonComplianceTrader:
    """
    Non-compliance traders buy / sell allowances for the purpose of hedge
    """

    pass


if __name__ == "__main__":
    # /// quick test ///
    pass
