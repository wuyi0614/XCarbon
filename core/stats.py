# -*- encoding: utf-8 -*-
#
# Created at 29/10/2021 by Yi Wu, wymario@163.com
#
import numpy as np
from core.base import init_logger


logger = init_logger('core-stats')


def mean(args, digits=None):
    """Wrapped mean method for a numeric vector"""
    return sum(args) / len(args) if digits is None else round(sum(args) / len(args), digits)


def sampling(vector, size, distribution, replace=False) -> iter:
    """
    Shuffle and sample given size of results from the target vector with / without a specific distribution

    :param vector       : the vector of elements
    :param size         : the expected size of selections
    :param distribution : statistical distribution of odds, e.g. normal, uniform distribution, etc.
    :param replace      : boolean, allow duplicates if True, otherwise the size must equals to the length of vector
    """
    if len(vector) < size:
        logger.warning("`size` is smaller than len(vector), use `replace=True` instead.")
        replace = True

    # create specified distribution
    if hasattr(np.random, distribution):
        rand = getattr(np.random, distribution)
        p = rand(size=len(vector))
        # ... scale the probability to the sum: `1` using `softmax`
        p = p - p.max()
        p = np.exp(p) / np.exp(p).sum()

    else:
        logger.error("unsupported distribution: %s against [poisson, uniform, normal, exponential]" % distribution)
        return iter([])

    return iter(np.random.choice(vector, size, p=p, replace=replace))


def get_rand_vector(size, digits=3, low=0, high=1, is_int=False):
    """
    A random number generator by given the size of vector and the number of digits

    :param size: the length / size of vector for randomly generated numbers
    :param digits: the digits of each number in the vector, default: 3
    :param low: the lower boundary of the vector, default: 0
    :param high: the upper boundary of the vector, default: 1
    :param is_int: generate random integer if True, otherwise random float, default: False
    """

    if is_int:
        return list(np.random.randint(low, high, size))

    return [round(i * (high - low) + low, digits) for i in np.random.random(size)]


def logistic_prob(gamma, theta, x):
    """A logistic-like function that returns probabilities"""
    return 1 + gamma * (1 / (1 + theta ** (-x)) - 0.5)


def range_prob(x, speed=0.6):
    """A probability function that returns [0, 1] probs whatever x is positive or negative"""
    return (abs(x) ** speed) * 0.5


# default emission accounting parameters
def emission_account(energy_use, em_factor):
    """Make sure the units of energy use and emission factor are consistent, for instance:

    - if energy use is x t(coal), then em_factor should use tCO2/t (coal)
    - if energy use is x MWh, then em_factor should use tCO2/MWh
    """
    pass

