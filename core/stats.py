# -*- encoding: utf-8 -*-
#
# Created at 29/10/2021 by Yi Wu, wymario@163.com
#
import numpy as np
from core.base import init_logger


logger = init_logger('core-stats')


def mean(args, digits=None, keep_zero=True):
    """Wrapped mean method for a numeric vector"""
    if not keep_zero:
        args = [each for each in args if each != 0]

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


def get_rand_vector(size, digits=3, low=0.0, high=1.0, is_int=False):
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


def normalization(vector):
    gap = np.max(vector) - np.min(vector)
    return (vector - np.min(vector)) / gap


def standardization(vector):
    mu = np.mean(vector, axis=0)
    sigma = np.std(vector, axis=0)
    return (vector - mu) / sigma


def distribution(vector, mode='linear', digits=3):
    """Convert a vector into a distribution"""
    if mode == 'linear':
        out = np.array(vector) / np.sum(vector)

    elif mode == 'softmax':
        vector = np.array(vector)
        vector -= np.max(vector)
        out = np.exp(vector) / np.sum(np.exp(vector))

    else:
        logger.error(f'Invalid input for distribution conversion mode: {mode} and use `linear` by default')
        out = np.array(vector) / np.sum(vector)

    return out.round(digits)


def random_distribution(vector, digits=3, var=0.01, mode='linear'):
    """Randomize a vector by putting a variance `var` into the vector"""
    vec = np.array(vector)
    if np.sum(vec) != 1:
        vec = distribution(vec, mode, digits)
        logger.warning(f'Invalid distribution vector: {vector} and now {vec}')

    vec -= np.array(get_rand_vector(len(vec), digits, low=-var, high=var))
    return distribution(vec)


def logistic_prob(gamma, theta, x):
    """A logistic-like function that returns probabilities"""
    return 1 + gamma * (1 / (1 + np.exp(-theta * x)) - 0.5)


def range_prob(x, speed=0.6):
    """A probability function that returns [0, 1] probs whatever x is positive or negative"""
    return (abs(x) ** speed) * 0.5


def choice_prob(*args, **kwargs):
    """Choice probability of technology options:
    The base value is x of ( 1/(1 + x^y) ) function, args are values of y
    The sum of power values is 1.
    kwargs are ties of `base=power`.
    """
    assert len(args) == len(kwargs), f'Mismatched sizes of `args`({len(args)}) and `kwargs`({len(kwargs)}'
    values = []
    kwargs = list(kwargs.values())
    for idx, arg in enumerate(args):
        base, power = list(kwargs[idx].values())
        v = (1 / (1 + base ** arg)) ** power
        values.append(v)

    return np.array(values).cumprod()[-1]
