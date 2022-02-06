#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Created at 27/10/2021 by Yi Wu, wymario@163.com
#
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime

from core import *

from scheduler import Scheduler
from core.firm import randomize_firm_specs, RegulatedFirm
from core.market import CarbonMarket, OrderBook
from core.base import Clock, read_config
from core.plot import plot_kline, plot_volume, plot_grid
from core.config import ENERGY_BASELINE_FACTORS, INDUSTRY

# The first option for ABM illustration is `streamlit: https://mp.weixin.qq.com/s/EH0w_PBNkFu6OgGfTB8CtQ`
# Advanced tutorial on streamlit operation and design: https://blog.streamlit.io/introducing-submit-button-and-forms/


# global settings of page
st.set_page_config('XCarbon Carbon Market Simulator', layout='wide')

# Top-level Demonstration banner
st.title('XCarbon: Carbon Market Simulator (powered by CIFE, China)')
st.markdown('To run the simulation of a national ETS in China, '
            'please specify and confirm parameters before pressing **Simulate** button')

# TODO: for those global parameters, use `sidebar` instead
glob = st.sidebar.form(key='global')

# market-level specifications should go into the sidebar
glob.header('Global settings')
# - market specification
glob_cap = glob.slider('The Cap of the Carbon Market', 1, 200, 20)
glob_ca_price = glob.slider('The initial carbon price', 1, 100, 50)
glob_decay = glob.number_input('Annual decay rate of allowance allocation', 0.0, 0.5, 0.05, step=0.02)
glob_threshold = glob.number_input('The entry threshold for regulated firms', 0.0, 0.2, 0.068, step=0.005)
glob_industry = glob.multiselect('Select entry industries in the carbon market ', options=INDUSTRY, default=['Power'])

# - simulation date
glob_start = glob.date_input('Select Start Date for simulation', datetime(2021, 1, 1),
                             min_value=datetime(2020, 1, 1), max_value=datetime(2021, 12, 31))
glob_end = glob.date_input('Select End Date for simulation', datetime(2021, 12, 31),
                           min_value=datetime(2020, 1, 1), max_value=datetime(2021, 12, 31))

# - scheduler specification
glob_buyer = glob.slider('How many buyers?', 1, 200, 20)
glob_seller = glob.slider('How many sellers?', 1, 200, 15)

# - output options: 1. show market report; 2. show runtime
glob.subheader('System/output options')
glob_allow_quit = glob.checkbox('Allow negative-profit firm to quit ', False)
glob_show_report = glob.checkbox('Show market report', False)
glob_show_runtime = glob.checkbox('Show simulation runtime', False)

# - submit the changes in the sidebar
confirm = glob.form_submit_button('Confirm')


# The followings are major parameters for simulation
form = st.form(key='parameters')
# The energy emission factor inputs
emi_factors = {}
with form:  # it's active when users add items in the `sidebar.glob_industry`
    st.subheader('Energy Module')
    with st.expander('Click to change the factors (or in default)'):
        cols = st.columns(len(glob_industry))
        for idx, col in enumerate(cols):
            ind = glob_industry[idx]
            with col:
                for each, factor in ENERGY_BASELINE_FACTORS.items():
                    emi_factors[each] = st.number_input(f'Emission factor of **{each}** in {ind}',
                                                        0.01, 10.0, factor, step=0.01)

    # The scheduler specifications
    st.subheader('Scheduler Module')
    with st.expander('Click to change Scheduler specifications (or in default)'):
        # ratio of compliance trader
        ratio_compliance = st.number_input('Ratio of compliance traders in the buyer/seller side',
                                           0.0, 1.0, 0.1, step=0.1)
        inflation = st.number_input('Inflation rate of firm-level factors for randomly generation',
                                    0.0, 1.0, 0.1, step=0.1)
        # trading probabilities
        probs = st.slider('Select a range of trading probabilities (lower probs and more transactions)',
                          0.0, 1.0, (0.1, 0.5))

    # The regulation module
    st.subheader('Regulation Module')
    with st.expander('Click to input regulation specification (or in default)'):
        st.number_input('Select penalty fee for firms failing to comply (1 mRMB/mt)',
                        1.0, 100.0, 10.0, step=1.0)
        st.number_input('Select the carbon tax level for the regulated firms (1 mRMB/mt)',
                        1.0, 100.0, 10.0, step=1.0)

    # The shock module
    st.subheader('Shock Module')
    with st.expander('Click to add shocks (or in default: no shocks)'):
        st.warning('Under construction: you may be able to select shocks on specific dates')

# - submit
submit = form.form_submit_button('Simulate')


if submit:  # only run simulations when it's certainly changed
    # Running simulation
    ck = Clock(glob_start.strftime('%Y-%m-%d'), glob_end.strftime('%Y-%m-%d'))
    ob = OrderBook('day')
    cm = CarbonMarket(ob, ck, 'day')

    sch_conf = read_config('config/scheduler-20211027.json')
    sch = Scheduler(cm, ck, 'day', **sch_conf)

    conf = read_config('config/power-firm-20211027.json')
    comp_buyer = ratio_compliance * glob_buyer
    for i in range(glob_buyer):
        conf_ = randomize_firm_specs(conf, 'buyer')
        buyer = RegulatedFirm(ck, Compliance=0, **conf_) if i < comp_buyer else RegulatedFirm(ck, Compliance=1, **conf_)
        sch.take(buyer)

    comp_seller = ratio_compliance * glob_seller
    for i in range(glob_seller):
        conf_ = randomize_firm_specs(conf, 'seller')
        seller = RegulatedFirm(ck, Compliance=0, **conf_) if i < comp_seller else RegulatedFirm(ck, Compliance=1,
                                                                                                **conf_)
        sch.take(seller)

    sch.run(probs=probs, compress=True)
    report = cm.to_dataframe()
    array = report[['open', 'close', 'low', 'high']].astype(float).values.tolist()
    dates = report.ts.values.tolist()
    # draw kline price chart
    c1 = plot_kline(array, dates, 'carbon price', 'carbon-market-300')

    # draw bar volume chart
    volumes = report['volume'].astype(float).values.round(1).tolist()
    c2 = plot_volume(volumes, dates)
    plot_grid(c1, c2, 'carbon-market', True)

    # show the orderbook
    st.dataframe(report)

    # HTML figures
    html = Path('carbon-market.html').read_text()
    components.html(html, height=600, width=704, scrolling=True)

if __name__ == '__main__':
    # TODO: in the next development round, put the demo below
    pass
