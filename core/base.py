# Base methods and instances for the modeling
#
# ver. 2020-10-20
#

import time
import json
import pandas as pd

from uuid import uuid1
from hashlib import md5
from pathlib import Path
from copy import deepcopy

from tqdm import tqdm
from loguru import logger
from datetime import datetime, timedelta
from multiprocessing import Pool

from timeit import default_timer as timer
from dateparser import parse as parse_date

from sklearn.utils import shuffle

# generalised environmental variables
DATE_FORMATTER = '%Y-%m-%d %H:%M:%S'
DATE_ELEMENTS = {'year': 'years', 'month': 'months', 'day': 'days',
                 'hour': 'hours', 'minute': 'minutes', 'second': 'seconds'}
DATE_FORMATS = {'year': '%Y', 'month': '%Y-%m', 'day': '%Y-%m-%d', 'hour': '%Y-%m-%d %H',
                'minute': '%Y-%m-%d %H:%M', 'second': '%Y-%m-%d %H:%M:%S'}
DEFAULT_LOGGING_DIR = Path('logs')
DEFAULT_LOGGING_DIR.mkdir(exist_ok=True, parents=True)


# scaffold for functional operations
def shuffler(pool: dict):
    """A special shuffler for agent pool"""
    items = [v for v in pool.values()]
    return {each.uid: each for each in shuffle(items)}


def jsonify_without_desc(item: dict):
    it = deepcopy(item)
    for k, v in item.items():
        if k.startswith('_desc'):
            it.pop(k)

        if isinstance(v, dict):
            for k_, v_ in v.items():
                if k_.startswith('_desc'):
                    it[k].pop(k_)

    return it


def read_config(file=None):
    if not file:
        return {}

    file = Path(file)
    if not file.exists():
        return {}

    return json.loads(file.read_text())


# hash functions
def md5string(string: str):
    return md5(string.encode()).hexdigest()


def generate_id(string=None):
    if string is None:
        return uuid1().hex[: 16]

    return md5string(string)


# multi-processing
def accelerator(func, iterable, pool_size, tag=None, runtime=False):
    """
    Accelerate the process of func using the multi-processing method, if pool_size is 1, execute `for-loop`.

    :param func: the target function
    :param iterable: iterable object, such as list, numpy.ndarray, ...
    :param pool_size: the size of multi-process pool
    :param runtime: show the runtime of processing
    """
    t0 = timer()
    if pool_size > 1:
        with Pool(pool_size) as pool:
            result = pool.map(func, iterable)

    else:
        result = [func(each) for each in tqdm(iterable, desc=tag)]

    if runtime:
        logger.info(f"Runtime: {timer() - t0} seconds")

    return result


# date processing
class Clock:
    """A clock object that allows to rollback, forward, get timestamp, and etc."""

    def __init__(self, date_string=None, **kwargs):
        # DO NOT change the elements here
        elements = ['year', 'month', 'day', 'hour', 'minute', 'second']
        if date_string:
            date = parse_date(date_string)
        else:
            now = datetime.now()
            default = {k: kwargs.get(k, now.__getattribute__(k)) for k in elements}
            date = datetime(**default)

        self._Raw = deepcopy(date)  # for `reset()`
        self.Date = date
        self.Element = elements
        self.Format = DATE_FORMATTER
        self.Frequency = kwargs.get('frequency', 'day')
        self._calibrate()  # update elements

    def __repr__(self):
        return self.Date.strftime(DATE_FORMATTER)

    def _calibrate(self):
        # granular elements
        for k in self.Element:
            self.__setattr__(k.title(), self.Date.__getattribute__(k))

    def adjust(self, frequency, steps, compress=False):
        """only support year, month, day, hour, minute, second, use `element=<steps>`, steps<0 means """
        self.Date = parse_date(alter_date(self.Date, frequency, steps, compress))
        self._calibrate()
        return self

    def move(self, steps):
        """quick API for Clock to go with steps (steps > 0)"""
        return self.adjust(self.Frequency, steps, True)

    def rollback(self, steps):
        return self.adjust(self.Frequency, -steps, True)

    def reset(self):
        self.__init__(self._Raw)
        return self

    # additional features
    @staticmethod
    def _workday(date):
        # TODO: import true holidays into the module
        num = date.isoweekday()
        return True if 1 <= num <= 5 else False

    @classmethod
    def _workdays_between(cls, early_date, late_date):
        count = 0
        days = cls._days_between(early_date, late_date)
        for day in range(days):
            date_ = early_date + timedelta(days=day)
            count += 1 if cls._workday(date_) else 0

        return count

    @classmethod
    def _days_between(cls, early_date, late_date):
        days = (late_date - early_date).days
        if days < 0:
            logger.warning(f'Late date should be late than early date')

        return abs(days)

    def days_left(self):
        if self.Month == 12:
            last_day = datetime(self.Year, self.Month, 31)
        else:
            last_day = datetime(self.Year, self.Month + 1, 1) - timedelta(days=1)

        return self._days_between(self.Date, last_day)

    def _get(self, frequency):
        return self.Date.strftime(DATE_FORMATS[frequency])

    def today(self):
        return self._get('day')

    def is_workday(self):
        return self._workday(self.Date)

    def is_monthend(self):
        days = self.days_left()
        return True if days == 0 else False

    def is_yearend(self):
        return self.Month == 12 and self.Day == 31

    def workdays_left(self):
        last_day = datetime(self.Year, self.Month + 1, 1) - timedelta(days=1)
        return self._workdays_between(self.Date, last_day)


def pause(secs):
    time.sleep(secs)
    return


def alter_date(tar, frequency, steps, compress=False, fmt=None):
    """
    Alter a date object / string given a frequency and steps

    :param tar: a date object or string
    :param frequency: second, minute, hour, day, month, year
    :param steps: an integer, positive means move forward, negative means rollback dates
    :param compress: compress the output by its frequency, e.g. from `2021-10-10 12:00:00` to `2021-10-10` (day)
    :param fmt: date format, default "%Y-%m-%d %H:%M:%S"
    """
    if not fmt:
        fmt = "%Y-%m-%d %H:%M:%S"
    if isinstance(tar, str):
        date = datetime.strptime(tar, fmt)
    else:
        date = tar

    if steps > 0:
        new = date + timedelta(**{DATE_ELEMENTS[frequency]: steps})
    else:
        new = date - timedelta(**{DATE_ELEMENTS[frequency]: -steps})

    if compress:
        fmt = DATE_FORMATS[frequency]

    return new.strftime(fmt)


def get_current_timestamp(fmt: str = "%Y-%m-%d"):
    """
    Return a string-type timestamp by a given date format

    :param fmt: a date format at least including "%Y", "%m" and "%d", default: "%Y-%m-%d"
    """
    return datetime.now().strftime(fmt)


def date_generator(start, end=None, periods=None, frequency=None, fmt=None, closed=None) -> list:
    """
    Wrapped pd.data_range() method for date generation.

    :param start: start date in well-known formats of dates, e.g. ``dd/mm/yyyy``, ``yyyy-mm-dd HH:MM:SS``, etc.
    :param end: end date in well-known formats of dates
    :param periods: number of periods to generate in integers
    :param frequency: frequency strings can have multiples, e.g. ``3 days: 3D``, ``5 hours: 5H``, etc.
    :param fmt: a full description about date format, ``dd/mm/yyyy`` by default (EUTL's official format)
    :param closed: make the interval closed to the given frequency to the 'left', 'right', or both sides (None default)
    :return: a sequence of generated dates
    """

    # default parameters
    if periods is None and end is None:
        periods = 1

    if frequency is None:
        frequency = "D"

    if fmt is None:
        fmt = DATE_FORMATTER

    dates = pd.date_range(start, end, periods, freq=frequency, closed=closed)
    dates = [date.strftime(fmt) for date in dates]

    # NB. when the interval isn't a uniform distribution, append ``end`` at the bottom
    if end:
        end = parse_date(end).strftime(fmt)
        if end not in dates:
            dates += [end]

    return dates


# logging
def init_logger(name, out_dir=None, level='INFO'):
    if out_dir is None:
        out_dir = DEFAULT_LOGGING_DIR

    out_name = out_dir / name
    logger.add(out_name.with_suffix(".log"), format="{time} {level} {message}", level=level)
    return logger


if __name__ == "__main__":
    # /// temporary test zone ///
    pass
