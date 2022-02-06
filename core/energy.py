#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Energy and emission abatement module for X-Carbon Modeling
#
# Created at 01/02/2022.
#

import json
from pathlib import Path
from collections import defaultdict

from pydantic import BaseModel
from core.base import init_logger, snapshot, jsonify_config_without_desc
from core.stats import random_distribution

# create a logger
logger = init_logger('energy', level='WARNING')


class Energy(BaseModel):
    """The Energy object for an industrial firm with featured attributes"""

    industry: str = 'unknown'  # firm's industry
    uid: str = 'unknown'  # firm's unique id
    cache: defaultdict = defaultdict(list)

    EnergyType: list = []
    EnergyUse: float = 0  # a float/int type aggregate energy consumption (unit: tonne)
    InputWeight: dict = {}  # proportion of each energy type
    InputEnergy: dict = {}  # specific energy input of each energy type
    GasUnitFactor: float = 1.4  # unit exchange rate for gas from Nm3 to tonne

    # emission factor:
    # tCO2/tonne for coal&oil&gas, and unit of gas should be converted from Nm3 to tonne
    # tCO2/kWh for electricity
    EmissionFactor: dict = {}
    Emission: dict = {}

    # energy prices:
    # 1,000 currency/tonne for coal&oil&gas,
    # 1,000 currency/kWh for electricity
    EnergyPrice: dict = {}
    EnergyCost: float = 0

    def __init__(self, random=False, **config):
        """Create an Energy object in random or not mode"""
        super().__init__(**config)
        # list all energies
        self.EnergyType = list(self.InputWeight.keys())

        if random:  # slightly randomize the InputWeight of energy
            vector = random_distribution(list(self.InputWeight.values()))
            for idx, (k, v) in enumerate(self.InputWeight.items()):
                self.InputWeight[k] = vector[idx]

    def __getitem__(self, item):
        return self.InputEnergy[item]

    def __repr__(self):
        snapshot = self._snapshot()
        return f"""Energy object of Firm [{self.uid}] at step={len(self.cache)} with status: 
        {list(snapshot.items())}"""

    def _snapshot(self):
        """Take a snapshot of attributes"""
        return snapshot(self)

    def to_json(self, path=None):
        """Convert attributes into a JSON file"""
        snapshot = self._snapshot()
        if path:
            Path(path).write_text(json.dumps(snapshot))

        return snapshot

    def from_json(self, path_or_data=None):
        """Load energy configuration from a JSON file"""
        pass

    def _compute(self):
        """Implicitly compute emissions and aggregate inputs by running the function"""
        # NB. for gas, the exchange rate between `tonne` and `m3` is `1t=1.4Nm3`
        self.get_energy()
        self.get_emission()
        self.get_cost()

    def fit(self, **kwargs):
        """Fit the Energy object on a set of parameters (consistent with its attributes).
        You may want to update InputWeight, EnergyPrice, or EmissionFactor ...
        """
        for k, v in kwargs.items():
            if k in dir(self):
                self.__setattr__(k, v)

    def get_energy(self):
        """Compute energy input by types"""
        if self.EnergyUse > 0:
            for k, v in self.InputWeight.items():
                if k == 'gas':
                    input_ = v * self.EnergyUse / self.GasUnitFactor
                else:
                    input_ = v * self.EnergyUse

                self.InputEnergy[k] = input_

        else:
            logger.error(f'Invalid EnergyUse value: {self.EnergyUse}')

    def get_emission(self):
        """Compute emissions from energy consumption"""
        for k, v in self.InputEnergy.items():
            self.Emission[k] = v * self.EmissionFactor[k]

    def get_cost(self):
        """Compute costs of energy consumption"""
        for k, v in self.InputEnergy.items():
            self.EnergyCost += v * self.EnergyPrice[k]

    def step(self):
        """Take the next step and make caches for all the inputs"""
        if self.uid == 'unknown':
            logger.warning(f'UID not properly setup and remain {self.uid}')

        self._compute()

        # make caches by steps
        for k in dir(self):
            if not k.islower() and k != 'Config':  # attributes for cache are capitalized
                self.cache[k] += self.__getattribute__(k)

        return self


def find_abate_option(config, industry) -> dict:
    """Find the industry-specific abatement options by searching the configuration"""
    opts = {}  # the keys are the actual names of options, e.g. efficiency, ccs, or else
    for typ, industries in jsonify_config_without_desc(config).items():
        for ind, opt in industries.items():
            if ind == industry:
                opts[typ] = opt

    if not opts:
        logger.warning(f'Invalid industry: {industry}, not in the configurations')

    return opts


class Abatement(BaseModel):
    """The Abatement object for an industrial firm. It can be understood as a technical package
    where options are available for various energy. The technical options are industry-specific:

    NB. Different industries may have limited availability of options, e.g. cement factories are
    hard to shift a majority of their coal consumption to gas.

    - Efficiency: energy saving & efficiency improvement, use less energy (including renewables for electricity),
                  target at `EmissionFactor`
    - Shift: specific energy consumption shift from one to the other, target at `InputWeight`
    - Negative: negative emission removal techniques such as CCS, target at `Emission`

    If the values are not negative for `efficiency` and `negative`, they could be percentages.

    The logics are that, (1) more investment are made, more abatement are generated if the investment doesn't
    exceed the potential of a specific technique. (2) The total abatement is decided by optimizing the total cost, and
    it should be distributed across various options based on their costs.
    """

    industry: str = 'unknown'  # firm's industry
    uid: str = 'unknown'  # firm's unique id
    cache: defaultdict = defaultdict(list)

    # NB. emission abatement could be long-term and thus the abatement should be discounted and
    # calculated in the short term (dependent on their lifespans), which should be included in the
    # `AbateOption` attribute.
    TechniqueType: list = []  # efficiency, ccs, shift (i.e. from coal to gas)
    TotalAbatement: float = 0  # total abatement

    AbateOption: dict = {}  # abatement options (portfolio) for various energy, except for ccs
    AbateWeight: dict = {}  # share of abatement techniques, decided after the cost optimization
    AbatePrice: dict = {}  # prices of abatement techniques
    AbateCost: float = 0  # aggregate cost of abatement

    EntryProb: dict = {}  # the default entry probability for firms to take options (Efficiency, Shift, Negative)

    def __init__(self, **config):
        super().__init__(**config)

        # list all abatement options
        self.TechniqueType = list(self.AbateOption.keys())

    def __repr__(self):
        pass

    def _snapshot(self):
        """Take a snapshot of attributes"""
        return snapshot(self)

    def update_option(self):
        """Update abatement options for their price and potential (incl. discount and lifespan) step by step"""
        pass

    def get_prob(self):
        """Compute probabilities for options"""

        pass
