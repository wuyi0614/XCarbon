# Base methods and instances for the modeling
#
# Created at 2021-12-08
#

# industries that will possibly be included in ETS
# the assumption is firms in different industries have different energy input structure
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fitter import Fitter

INDUSTRY = ['Power', 'Iron', 'Cement', 'Chemistry', 'Construction']

# energy input structure of firms in different industries
# TODO: move the configurations to a config module in the future
FOSSIL_ENERGY_TYPES = ('coal', 'gas', 'oil', 'electricity')
RENEWABLE_ENERGY_TYPES = ('wind', 'solar', 'biomass', 'thermal')  # TODO: will be used in the future

# the unit of emission factors is `tCO2/t` (standardized coal use), electricity unit: tCO2/kWh
# source: http://www.tanpaifang.com/tanjiliang/2014/0914/38053.html
ENERGY_BASELINE_FACTORS = {'coal': 0.1, 'gas': 0.05, 'oil': 0.07, 'electricity': 0.01}

# 能源换算单位
DEFAULT_GAS_UNIT = 1 / 1.16 * 1e-3  # unit exchange rate for gas from Nm3 to ton: `m3` is `1~1.33t=1Nm3`
DEFAULT_ELEC_UNIT = 1 / 0.1229 * 1e-3  # exchange rate for electricity from kWh to tone: `0.1229t=1kwh`
DEFAULT_EMISSION_UNIT = 1000

# 能源折标煤系数: https://wenku.baidu.com/view/e2d8af92f11dc281e53a580216fc700abb685295.html, unit: tCoal/t
ENERGY_TO_COAL_MAPPING = {'coal': 0.71, 'gas': 1.25, 'oil': 1.4}

# 发热量/单位的转换系数 unit: GJ/t (差别不会很大）
ENERGY_TO_HEAT_MAPPING = {'coal': 20908, 'gas': 38931, 'oil': 41816}

# energy input proportion in a specific industry (unit: %)
ENERGY_INPUT_POWER = {'coal': 0.7, 'gas': 0.2, 'oil': 0.05, 'electricity': 0.05}
ENERGY_INPUT_IRON = {'coal': 0.7, 'gas': 0.2, 'oil': 0, 'electricity': 0.1}
ENERGY_INPUT_CEMENT = {'coal': 0.5, 'gas': 0.3, 'oil': 0, 'electricity': 0.2}
ENERGY_INPUT_CHEMISTRY = {'coal': 0.3, 'gas': 0.3, 'oil': 0.2, 'electricity': 0.2}
ENERGY_INPUT_CONSTRUCTION = {'coal': 0.1, 'gas': 0.2, 'oil': 0, 'electricity': 0.7}
ENERGY_INPUT_GENERAL = {'coal': 0.25, 'gas': 0.25, 'oil': 0.25, 'electricity': 0.25}

# TODO: energy input price and it should follow the changes in energy consumption

# carbon market general settings
# source: 中国碳市场回顾与展望(2022), 王科
DEFAULT_MARKET_CAP = 45e5  # unit: 1000 tonnes
DEFAULT_MARKET_DECAY = 0  # it is temporarily 0
DEFAULT_MARKET_THRESHOLD = 68  # 68,000 tonnes


def get_energy_input_proportion(industry):
    """Return the energy input proportion of a specific industry"""
    mapping = {
        'power': ENERGY_INPUT_POWER,
        'iron': ENERGY_INPUT_IRON,
        'cement': ENERGY_INPUT_CEMENT,
        'chemistry': ENERGY_INPUT_CHEMISTRY,
        'construction': ENERGY_INPUT_CONSTRUCTION,
        'unknown': ENERGY_INPUT_GENERAL
    }
    return mapping.get(industry.lower(), ENERGY_INPUT_GENERAL)


# carbon intensity statistics and distribution
# source: https://www.sohu.com/a/504797445_121106994
INDUSTRY_CARBON_INTENSITY = {'power': 18, 'cement': 12, 'iron': 4, 'chemical': 6}
# income (unit: 1e8 RMB), emission (unit: 1e4 tonnes, intensity (unit: tonne/10000RMB)
CARBON_INTENSITY_DATA_PATH = Path('config') / 'carbon-intensity-20220206.json'


def get_firm_distribution(industry: str, element: str, population: int, datafile: Path = None) -> np.ndarray:
    """Return the mean and std values for a normal distribution of firm carbon intensity

    :param industry: which industry firms belong to
    :param element: which element selected for a fitted distribution, income, emission
    :param datafile: Path object for datafile
    :param population: the population of regulated firms
    """
    file = datafile if datafile else CARBON_INTENSITY_DATA_PATH
    data = file.read_text('utf8')
    data = pd.DataFrame(json.loads(data)[industry])

    # NB: we assume that it is normal distribution, and the units:
    #     income: 1e8 RMB, emission: 1e4 tonne, intensity: t/1e4 RMB
    f = Fitter(data[element].values, distributions=['norm'])
    f.fit()
    param = f.get_best(method='sumsquare_error')['norm']

    # simulate the distribution of element
    thresholds = dict(emission=DEFAULT_MARKET_THRESHOLD,
                      income=DEFAULT_MARKET_THRESHOLD * 1e3 / 18 / 1e4,  # unit: 1e8 RMB
                      intensity=data['intensity'].min())
    units = dict(emission=10,
                 income=10e5,  # unit: 1e6 -> energy: 2e4 ~ 5e4 -> emission 5e4 ~ 15e4 (500w ~ 1500w)
                 intensity=10)

    out = np.empty([0])
    while population:
        sim = np.random.normal(param[0], param[1], population)
        sim = sim[sim > thresholds[element]]  # only preserve the positive values
        out = np.append(out, sim)
        population = population - sim.size

    return out * units[element]
