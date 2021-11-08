# I/O operations within the Carbon Market modeling
#
# Created at 2021-11-08
#

import shelve

from copy import deepcopy
from pathlib import Path


DEFAULT_SAVE_PATH = Path('save')


# functions for I/O
def save(file, **objs):
    p = Path(file)
    f = shelve.open(str(p))
    for name, obj in objs.items():
        f[name] = obj

    f.close()


def load(file) -> dict:
    assert Path(file).exists(), f'Invalid shelve file path: {file}'

    p = Path(file)  # in case it's not a string
    out = {}
    f = shelve.open(str(p))
    for name, obj in f.items():
        out[name] = deepcopy(obj)

    return out
