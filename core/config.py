# Base methods and instances for the modeling
#
# Created at 2021-12-08
#

# industries that will possibly be included in ETS
# the assumption is firms in different industries have different energy input structure
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

# 能源折标煤系数: https://wenku.baidu.com/view/e2d8af92f11dc281e53a580216fc700abb685295.html, unit: tCoal/t
ENERGY_TO_COAL_MAPPING = {'coal': 0.71, 'gas': 1.25, 'oil': 1.4}

# 发热量/单位的转换系数 unit: GJ/t (差别不会很大）
ENERGY_TO_HEAT_MAPPING = {'coal': 20908, 'gas': 38931, 'oil': 41816}

# energy input proportion in a specific industry (unit: %)
ENERGY_INPUT_POWER = {'coal': 0.7, 'gas': 0.3, 'oil': 0, 'electricity': 0}
ENERGY_INPUT_IRON = {'coal': 0.7, 'gas': 0.2, 'oil': 0, 'electricity': 0.1}
ENERGY_INPUT_CEMENT = {'coal': 0.5, 'gas': 0.3, 'oil': 0, 'electricity': 0.2}
ENERGY_INPUT_CHEMISTRY = {'coal': 0.3, 'gas': 0.3, 'oil': 0.2, 'electricity': 0.2}
ENERGY_INPUT_CONSTRUCTION = {'coal': 0.1, 'gas': 0.2, 'oil': 0, 'electricity': 0.7}
ENERGY_INPUT_GENERAL = {'coal': 0.25, 'gas': 0.25, 'oil': 0.25, 'electricity': 0.25}

# TODO: energy input price and it should follow the changes in energy consumption


# carbon market general settings
DEFAULT_MARKET_CAP = 20
DEFAULT_MARKET_DECAY = 0
DEFAULT_MARKET_THRESHOLD = 0.068


def get_energy_input_proportion(industry):
    """return the energy input proportion of a specific industry"""
    mapping = {
        'power': ENERGY_INPUT_POWER,
        'iron': ENERGY_INPUT_IRON,
        'cement': ENERGY_INPUT_CEMENT,
        'chemistry': ENERGY_INPUT_CHEMISTRY,
        'construction': ENERGY_INPUT_CONSTRUCTION,
        'unknown': ENERGY_INPUT_GENERAL
    }
    return mapping.get(industry.lower(), ENERGY_INPUT_GENERAL)
