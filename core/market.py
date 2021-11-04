# The script for market specification
import pandas as pd

from copy import deepcopy
from typing import Optional, Union
from pydantic import BaseModel
from multiprocessing import Queue
from datetime import datetime
from collections import OrderedDict

from core.base import md5string, logger, get_current_timestamp, date_generator, alter_date, DATE_FORMATTER
from core.stats import mean, get_rand_vector, sampling


# MARKET ELEMENTS
class TransactionQueue:
    """
    A queue-like instance (dynamic) for recording e.g. Ticks, Bars, Statements ...,
    and it wraps multiprocessing.Queue and owns all the properties of it.

    Basically all of transactions proposed by agents will be passed into the Queue and thus it
    will be requested by the Order Book (Market).
    """

    def __init__(self, name=None, **kwargs):
        self.name = name if name else "debug"
        self.queue = Queue(kwargs.get("size", 1000))

    def is_full(self):
        return self.queue.full()

    def is_empty(self):
        return self.queue.empty()


class Tick(BaseModel):
    """
    Market tick (i.e. transaction) data instance (static), the basic unit of a bar data of market dynamics,
    which is consisted of a bunch of Statements (within a fixed period).

    Within the tick data:

    :param open     : the open price of the Tick
    :param close    : the close price of the Tick
    :param quantity : the transaction quantity of the Tick
    :param start    : the start time of the Tick
    :param end      : the end time of the Tick
    """

    open: float
    close: float
    quantity: int
    start: str
    end: str


class Bar(BaseModel):
    """
    Market bar data instance (static), the basic unit of a sequence of market dynamics.

    Within the bar data:

    :param open      : open price
    :param close     : close price
    :param mean      : mean price (averaged price by quantities)
    :param prev_close: close price of the previous bar
    :param high_limit: the highest price in a given trade time (e.g. +- 10% change limit)
    :param low_limit : the highest price in a given trade time (e.g. +- 10% change limit)

    :param quantity  : current quantities
    :param volume    : current transaction volume (price * quantities)
    :param ts        : current timestamp
    :param frequency : transaction market frequency: hour, day, month, ...
    """

    # the following are related prices
    open: Optional[float]
    close: Optional[float]
    mean: Optional[float]
    prev_close: Optional[float]
    high: Optional[float]
    low: Optional[float]
    # ... related volumes
    quantity: Optional[int]
    volume: Optional[float]
    # ... characteristics
    ts: Optional[str]
    frequency: Optional[str]

    @staticmethod
    def timestamp():
        return get_current_timestamp("%Y-%m-%d %H:%M:%S")

    def __init__(self,
                 open=None,
                 close=None,
                 mean=None,
                 prev_close=None,
                 high=None,
                 low=None,
                 quantity=None,
                 volume=None,
                 ts=None,
                 frequency=None,
                 **kwargs):
        super(Bar, self).__init__(**kwargs)

        self.open = get_rand_vector(1, low=10, high=40).pop() if open is None else open
        ratio = get_rand_vector(1, low=0.9, high=1.1).pop()
        self.close = round(ratio * self.open, 3) if close is None else close
        self.mean = sum([self.open, self.close]) / 2 if mean is None else mean
        self.prev_close = get_rand_vector(1).pop() if prev_close is None else prev_close
        self.high = self.open * 1.1 if high is None else high
        self.low = self.open * 0.9 if low is None else low

        self.quantity = get_rand_vector(1, low=100, high=1000, is_int=True).pop() if quantity is None else quantity
        self.volume = self.mean * self.quantity if volume is None else volume
        self.ts = self.timestamp() if ts is None else ts
        self.frequency = "1S" if frequency is None else frequency


class Statement(BaseModel):
    """
    Statement instance (static): successful feedback for each transaction

    :param id        : settlement id which is concatenated by both ids of transactions from opposite modes
    :param price     : settlement price
    :param quantity  : settlement quantity
    :param offerer_id: the offerer id in settlement
    :param offeree_id: the offeree id in settlement (optional when the transaction failed)
    :param status    : settlement status either 1 (success) or 0 (failure)
    :param ts        : settlement timestamp in specified date format (optional)
    """

    id: str
    price: float
    quantity: float  # DO NOT have to be `int` which is dependent on its scale
    offerer_id: str
    offeree_id: Optional[str] = ""
    status: Optional[int] = None
    ts: Optional[str] = ""

    def __init__(self, **specs):
        super(Statement, self).__init__(**specs)
        self.ts = specs.get("ts", self._timestamp())
        # /// binary status ///
        self.status = 1 if specs.get("offeree_id") else 0

    @staticmethod
    def _timestamp():
        return get_current_timestamp("%Y-%m-%d %H:%M:%S")


class Order(BaseModel):
    """
    Basic Order instance: the elementary transaction unit of the market

    :param price   : transaction price (float)
    :param mode    : transaction mode: either `buy` or `sell`
    :param quantity: transaction quantity (integers), abort transaction if it's zero
    :param agent_id: transaction proposer's id in string format
    :param ts      : transaction timestamp in specified date format
    :param id      : transaction id which is hashed and unique
    :param digits  : numeric digits for price
    """

    price: Optional[float] = None
    mode: Optional[str] = None
    quantity: Optional[int] = None
    offerer_id: Optional[str] = None
    offeree_id: Optional[str] = None
    ts: Optional[str] = None
    id: Optional[str] = None
    digits: Optional[int] = 3
    status: Optional[str] = "waiting"

    @staticmethod
    def timestamp():
        return get_current_timestamp("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def md5(ts, mode, offerer_id):
        string = "%s+%s+%s" % (ts, mode, offerer_id)
        return md5string(string)

    def __init__(self,
                 price=None,
                 mode=None,
                 quantity=None,
                 offerer_id=None,
                 **specs):
        """Create a dynamic Order instance with the following compulsory parameters"""
        super(Order, self).__init__(**specs)
        # /// random initialization ///
        self.price = get_rand_vector(1).pop() if price is None else price
        self.mode = list(sampling(["sell", "buy"], 1, "normal")).pop() if mode is None else mode
        self.quantity = get_rand_vector(1, low=100, high=1000, is_int=True).pop() if quantity is None else quantity
        id_ = get_rand_vector(1, low=1, high=100, is_int=True).pop()
        self.offerer_id = "test-%d" % id_ if offerer_id is None else offerer_id

        # /// general specs ///
        # transaction id is generated by "timestamp+mode+accountId", which is unique
        self.ts = specs.get("ts", self.timestamp())
        # generate transaction id
        self.id = self.md5(self.ts, self.mode, self.offerer_id)[:16]
        # digits for float values, default 3
        self.digits = specs.get("digits", self.digits)

        # /// validation ///
        # TODO: frequent transaction / invalid transaction date etc. ...
        # transaction status has THREE types: waiting / rejected / accepted
        self.status = "waiting"
        self.offeree_id = ""

    def settle(self, transaction) -> tuple:
        """Make settlement from offeree's proposed transaction (Order instance)

        :param transaction: the Order instance with an opposite mode (buy -> sell / sell -> buy)
        :return a tuple of (Statement(success), self, Order(accepted)) or (None, self, Order(waiting))
        """
        # successful settlement: 1) buy with a lower offeree's price, 2) sell with a higher offeree's price
        success = (self.mode == "buy" and transaction.mode == "sell" and transaction.price <= self.price,
                   self.mode == "sell" and transaction.mode == "buy" and transaction.price >= self.price)

        if any(success):
            # settlement price is the mean value of the two orders
            settle_price = mean([transaction.price, self.price], digits=self.digits)
            # settlement quantity depends on transaction direction
            if success[0]:  # buyer side (self.quantity < 0)
                qty_diff = transaction.quantity - abs(self.quantity)
            else:  # only two elements and hereby seller side (self.quantity > 0)
                qty_diff = abs(transaction.quantity) - self.quantity

            if qty_diff <= 0:
                # offerer is not fulfilled: 1) change quantity to be `diff`, 2) status remains `waiting`
                qty_settle = abs(transaction.quantity)
                # at seller side, the gap is positive
                self.quantity = qty_diff if success[0] else abs(qty_diff)
                self.status = "waiting"
                self.ts = self.timestamp()

                # update transaction
                transaction.status = "accepted"
                transaction.offeree_id = self.offerer_id
                transaction.ts = self.timestamp()

            else:
                # transaction is fulfilled
                qty_settle = deepcopy(abs(self.quantity))
                self.status = "accepted"
                transaction.ts = transaction.timestamp()
                # at the seller side (success[1]=True), transaction is from buyer side and its qty < 0
                transaction.quantity = qty_diff if success[0] else -qty_diff

            assert qty_settle > 0, f'{self} \n {transaction}'
            self.offeree_id = transaction.offerer_id
            settlement = Statement(id="%s+%s" % (self.id, transaction.id),
                                   price=settle_price,
                                   quantity=qty_settle,
                                   offerer_id=self.offerer_id,
                                   offeree_id=transaction.offerer_id,
                                   status=1)
            return settlement, self, transaction

        # ... if price + premium is not allowed, return (None, non_settled(transaction))
        # TODO: allow positive / negative price premium
        return None, self, transaction


# MARKET FUNCTIONS
class OrderBook:
    """
    A basic customised Order Book System which is a `read-` and `append-only` book:
    (1) query the Order Book by specific criterion
    (2) insert verified orders into the Book System
    (3) dump orders and help generate Bar data on a given time-base

    :param frequency : the transaction frequency, default as "1D" - daily, e.g. "xH", "yM", "zY", ...
                       which is consistent with pandas's date range formats (x, y, z are integers)
    :param book      : the OrderDict object for the storage of orders (default in order)
    """

    def __init__(self, frequency=None, book=None, **kwargs):
        self.empty = True
        self.frequency = "1D" if frequency is None else frequency

        # if given a book in pd.DataFrame, transform by pandas twice,
        # NB. a book should use `transaction_id` as keys
        self.book = OrderedDict() if book is None else book
        # ... a copy of self.book for `self.get` method i.e. a reversed list of orders
        self._book = []

        # NB. a statement pool use `statement_id` as keys
        # ... cannot import Statement Pool; can only be generated
        self.pool = OrderedDict()
        self._pool = []

        # NB. the sorting / matching strategy: "price", "ts", "quantity", etc., use price as default
        self.match_by_sort = kwargs.get("match_by_sort", "price")

        if isinstance(self.book, pd.DataFrame):
            self.book = self.book.to_dict(into=OrderedDict, orient="index")

    def __str__(self):
        return "[OrderBook] time-base: {}, records: {}".format(self.frequency, len(self.book))

    def __repr__(self):
        return self.to_dataframe().__repr__()

    def __getitem__(self, item):
        return self.book.get(item)

    def to_dataframe(self, item=None):
        if item is None:
            item = self.book

        return pd.DataFrame.from_dict(item, orient="index").copy(deep=True)

    def iter_by_date(self, book=None, frequency=None) -> iter:
        """
        Return an iterator of pd.DataFrame based on the Order Book data

        :param book     : iterable Order Book instance or Statement Pool instance
        :param frequency: date frequency ("H", "D", "M", "Y", etc.), use self.frequency as default
        """
        if frequency is None:
            frequency = self.frequency

        book = self.to_dataframe(book)

        # fake a list of dates and scale the order dates into the given date gaps
        dates = date_generator(start=book.ts.values[0],
                               end=book.ts.values[-1],
                               frequency=frequency,
                               fmt=DATE_FORMATTER)

        for index in range(len(dates) - 1):
            left, right = dates[index], dates[index + 1]
            masked = (book.ts >= left) & (book.ts <= right)
            res = book[masked]
            yield pd.DataFrame([]) if res.empty else res

    def reset(self, **kwargs):
        """Reset the Order Book back to the initial state (pipe orders to recycling object than drop them)"""
        self.__init__(**kwargs)
        return self

    def _put_into_book(self, item):
        """"""
        self.book[item.id] = item.dict()
        self._book = list(self.book.items())
        return self

    def _put_into_pool(self, item):
        self.pool[item.id] = item.dict()
        self._pool = list(self.pool.items())
        return self

    def put(self, item: Order):
        """Put order items like a Queue in the sorted way"""
        self.empty = False
        # every order will be executed and compared to the Order Book
        if item.mode == "sell":
            query = "price>={} & mode!='{}' & status=='waiting' & ts<='{}'".format(item.price, item.mode, item.ts)
        else:
            query = "price<={} & mode!='{}' & status=='waiting' & ts<='{}'".format(item.price, item.mode, item.ts)

        selected = self.query(query)
        if selected.empty:
            self._put_into_book(item)
            return self

        # TODO: match code - sort values, by "date" or by "price"
        selected = selected.sort_values(by=self.match_by_sort, ascending=False)

        for id_ in selected["id"].values:
            # the process should be continuous if the first order cannot fulfill the request
            state, offerer, offeree = Order(**self.book[id_]).settle(item)
            # check if `state` is None
            if state:
                # update offerer's side
                self._put_into_book(offerer)
                # update offeree's side and recursively refresh item when the new order isn't fulfilled
                if offeree.status == "accepted":
                    self._put_into_book(offeree)
                else:
                    item = offeree

                # record Statements into the Pool and state is mainly determined by `Order` object
                self._put_into_pool(state)
                logger.info("process statement: %s" % state.id)

        # none settlement is done, put the order into the book system
        self._put_into_book(item)
        return self

    def get(self):
        """Get item from the Order Book following the rule of a Queue (first in first out)"""
        return self._book.pop()

    def query(self, q):
        """Query from the order book by specific kwargs from Order's attributes, including:

        price
        mode
        quantity
        offerer_id
        offeree_id
        ts
        id
        digits
        status

        Note: only support a comprehensive query phrase based on pandas.DataFrame.query(),
        strings with blanks within them should be wrapped by another quotation mark i.e. "a 'b c'"
        """
        book = self.to_dataframe()
        if book.empty:
            return pd.DataFrame([])

        return book.query(q)

    # preset alignments for Order Book output, depending on the granularity of dates plus
    # status, mode, price and quantity as well.
    ALIGNMENT = ("minute", "hour", "day", "month", "year",
                 "price_ascend", "price_descend",
                 "quantity_ascend", "quantity_descend",
                 "status", "mode")

    def align(self, orient: str, fmt=None) -> pd.DataFrame:
        """
        Align the Order Book by a given orientation

        :param fmt   : date formatter, default use DATE_FORMATTER
        :param orient: an orientation for alignment choice including:

        minute
        hour
        day
        month
        year
        price_ascend
        price_descend
        quantity_ascend
        quantity_descend
        status
        mode
        """
        fmt = fmt if fmt else DATE_FORMATTER
        if orient not in self.ALIGNMENT:
            return pd.DataFrame([])

        book = self.to_dataframe()
        if orient in ("minute", "hour", "day", "month", "year"):
            formatter = lambda string: datetime.strptime(string, fmt)
            book["datetime"] = book.ts.apply(formatter)
            book["sort_"] = [getattr(item, orient) for item in book["datetime"]]
            return book.sort_values(by="sort_", ascending=True).copy(deep=True)

        elif orient in ("price_ascend", "price_descend", "quantity_ascend", "quantity_descend"):
            key, sort = orient.split("_")
            ascend = True if sort == "ascend" else False
            return book.sort_values(by=key, ascending=ascend).copy(deep=True)

        elif orient in ("status", "mode"):
            return book.sort_values(by=orient, ascending=True).copy(deep=True)

    def save(self, tag=None):
        """Dump the Order Book cache into a disk file"""
        pass


class BaseMarket:
    """A base market instance"""

    def __init__(self, book: OrderBook, frequency, **kwargs):
        """
        Initiate the Basic Market instance given the time base

        :param bar      : the initial Bar to start the market
        :param time_base: a time base (date frequency) for sampling, use "S(second)" by default
        """
        # /// general market spec ///
        self.Frequency = frequency
        self.OrderBook = book

    def clear(self):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def reset(self):
        pass


class CarbonMarket(BaseMarket):
    """A prototyped carbon market inheritted from BaseMarket"""

    def __init__(self, ob: OrderBook, clock, freq='day', **spec):
        super().__init__(ob, freq, **spec)
        self.Bars = []  # data pool
        self.OrderBook = ob  # internally linked with the order book
        self.Frequency = freq
        self.Clock = clock  # a shared clock with scheduler

    def __repr__(self):
        """Display the present status of the market"""
        return f"""Carbon Market runs on a {self.Frequency} basis and the current market:
    {self.to_dataframe()}"""

    def to_dataframe(self):
        bars = [item.dict() for item in self.Bars]
        return pd.DataFrame(bars)

    # the following functions are for plotting
    def display(self, **kwargs):
        report = self.to_dataframe()
        # 使用pyecharts绘制可交互的K线

    def _create_bar_metrics(self, book, freq, mode='order'):
        if mode == 'order':
            open_ = book.price.values[0]
            high = book.price.max()
            low = book.price.min()
            close = book.price.values[-1]
            prev_close = 0 if not self.Bars else self.Bars[-1].close

            volume = book.price.values * book.quantity.values
            mean_ = volume.sum() / book.quantity.values.sum()
            # quantity metrics
            quantity = book.quantity.values.sum()
            volume = volume.sum()

        elif mode == 'state':
            open_ = book['open'].values[0]
            high = max(book['open'].max(), book['close'].max())
            low = min(book['open'].min(), book['close'].min())
            close = book['close'].values[-1]
            prev_close = None

            volume = book.volume.values
            mean_ = volume.sum() / book.quantity.values.sum()
            # quantity metrics
            quantity = book.quantity.values.sum()
            volume = volume.sum()
        else:
            return None

        # update the transaction date
        ts = self.Clock._get(freq)  # DO NOT move clock inside of the market
        return Bar(open=open_,
                   mean=mean_,
                   prev_close=prev_close,
                   high=high,
                   low=low,
                   close=close,
                   volume=volume,
                   quantity=quantity,
                   ts=ts,
                   frequency=freq)

    def _make_bars(self, freq='day', **kwargs):
        """The market will convert accpeted orders(states) into data bars"""
        if self.OrderBook.empty:
            logger.warning(f'No transaction happened at Day {self.Clock.today()}')
            return None

        # NB. assume when a book is passed through, the period is over
        book = pd.DataFrame(self.OrderBook.pool.values())
        # if OrderBook is empty without any accepted orders
        if book.empty:  # all orders are waiting
            return None

        return self._create_bar_metrics(book, freq, 'order')

    def aggregate(self, freq):
        # if aggregate at montly basis, have to filter the book,
        # because bars happen at daily basis (incl. other months)
        book = self.to_dataframe()
        if freq == 'month':
            month = self.Clock._get('month')
            book = book[book.ts.str.contains(month)]

        if book.empty:
            return

        return self._create_bar_metrics(book, freq, 'state')

    def open(self):
        """Market create an order book and starts trading"""
        self.OrderBook = self.OrderBook.reset()
        return self

    def close(self, **kwargs):
        """Market clears the order book when it's closed"""
        bar = self._make_bars(**kwargs)
        self.Bars += [bar] if bar else []
        return self

    def reset(self, **kwargs):
        self.__init__(self.OrderBook.reset(), **kwargs)
        return self


# FUNCTIONS
def bundle_bar_data(last_bar: Bar, iter_orders: pd.DataFrame, frequency) -> Bar:
    """
    Convert a set of ticks into a bundled Bar data within a upper date boundary

    :param last_bar    : the previous bar in the sequence
    :param iter_orders : iterable orders from OrderBook.iter_by_date()
    :param frequency   : the Bar frequency e.g. "H", "D", "M", "Y"
    :return Bar        : return an initiated Bar object
    """
    prices = iter_orders.price.sort_values(ascending=True).values
    volume = iter_orders.price.values * iter_orders.quantity.values
    ts = iter_orders.ts.values.max()

    bar = Bar(open=prices[0], close=prices[-1], mean=prices.mean(),
              prev_close=last_bar.close,
              high_limit=last_bar.high_limit,
              low_limit=last_bar.low_limit,
              quantity=iter_orders.quantity.sum(), volume=volume.sum(),
              ts=ts, frequency=frequency)
    return bar


if __name__ == "__main__":
    from core.base import Clock

    # /// temporary test zone ///
    # TODO: tests should be migrated to /test directory in the next step
    buy = Order(price=0.5, quantity=-300, mode="buy", offerer_id="o1")
    sell = Order(price=0.4, quantity=200, mode="sell", offerer_id="s1")
    # successful sell settlement:
    state, offerer, offeree = buy.settle(sell)
    assert state.price == 0.45
    assert offerer.quantity == -100
    assert offeree.status == "accepted"

    # failed sell settlement:
    buy = Order(price=0.5, quantity=-300, mode="buy", offerer_id="o1")
    sell = Order(price=0.6, quantity=200, mode="sell", offerer_id="s1")
    state, offerer, offeree = buy.settle(sell)
    assert state is None
    assert offerer.status == "waiting"
    assert offeree.status == "waiting"

    # use OrderBook to match orders
    ob = OrderBook(frequency='day')
    clock = Clock('2021-01-01')

    # Test: market function
    market = CarbonMarket(ob=ob, clock=clock, freq='day')
    market.open
    market.close()