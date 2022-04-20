#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Simplified crawler for data collection
#
# Created at 2022-04-20.
#

import re
import time
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')


import pandas as pd
import requests

from datetime import datetime

from tqdm import tqdm
from bs4 import BeautifulSoup as Soup

# China's national ETS data
ROOT_URL = 'https://www.cneeex.com/'

CNEEX_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36'}


def get_page_data(urls: list, root_url=ROOT_URL, headers=CNEEX_HEADERS):
    """crawl data from each page"""
    pages = []
    for page_url in urls:
        # get urls from the page
        response = requests.get(page_url, headers=headers)
        soup = Soup(response.text)
        page_urls = [item.get('href') for item in soup.select('li[class="hidden-lg hidden-sm hidden-md"] a')]

        # collect data at each page
        for url in tqdm(page_urls, desc=f'Crawling on {page_url}'):
            if url.endswith('shtml'):
                url = f'{root_url}{url}' if not url.startswith(root_url) else url
                res = requests.get(url, headers=headers)
                soup = Soup(res.text.encode(res.encoding))
                pages += [[page_url,
                           soup.select('h3').pop().get_text(),
                           soup.select('div[class="detail-con"]')[0].get_text()]]

                # pause for a while
                time.sleep(2)
        # pause for a while for each page
        time.sleep(2)

    return pages


def get_date(x):
    date = re.findall(r'\d+\-\d+\-\d+', x).pop()
    return datetime.strptime(date, '%Y-%m-%d')


def get_trade_data(x):
    """extract trade data from content including listed trade and bulk trade"""
    def get(phrase, x):
        found = re.findall(phrase, x)
        if not found:
            return '0', '0'
        else:
            return found.pop()

    def get_price(phrase, x):
        found = re.findall(phrase, x)
        if not found:
            return '0.0'
        else:
            return found.pop()

    # trade quantity & volume
    # list, bulk, total, accumulate
    list_qty, list_vol = get(r'挂牌协议交易成交量([\d,]+)吨，成交额([\d,\.]+)元', x)
    bulk_qty, bulk_vol = get(r'大宗协议交易成交量([\d,]+)吨，成交额([\d,\.]+)元', x)
    total_qty, total_vol = get(r'总成交量([\d,]+)吨，总成交额([\d,\.]+)元', x)
    acc_qty, acc_vol = get(r'累计成交量([\d,]+)吨，累计成交额([\d,\.]+)元', x)
    qty = [locale.atoi(it) for it in [list_qty, bulk_qty, total_qty, acc_qty]]
    vol = [locale.atof(it) for it in [list_vol, bulk_vol, total_vol, acc_vol]]

    # trade price
    # open, high, low, close
    open_ = get_price(r'开盘价([\d\.]+)元/吨', x)
    high = get_price(r'最高价([\d\.]+)元/吨', x)
    low = get_price(r'最低价([\d\.]+)元/吨', x)
    close = get_price(r'收盘价([\d\.]+)元/吨', x)
    prices = [locale.atof(it) for it in [open_, high, low, close]]

    result = {
        'date': get_date(x),
        'list-quantity': qty[0],
        'bulk-quantity': qty[1],
        'total-quantity': qty[2],
        'accumulate-quantity': qty[3],
        'list-volume': vol[0],
        'bulk-volume': vol[1],
        'total-volume': vol[2],
        'accumulate-volume': vol[3],
        'open': prices[0],
        'high': prices[1],
        'low': prices[2],
        'close': prices[3]
    }
    return result


if __name__ == '__main__':
    urls_2021 = ['https://www.cneeex.com/qgtpfqjy/mrgk/2021n/index.shtml'] + \
                [f'https://www.cneeex.com/qgtpfqjy/mrgk/2021n/index_{i}.shtml' for i in range(2, 6, 1)] + \
                [f'https://www.cneeex.com/cneeex/catalog/15372/pc/index_{i}.shtml' for i in range(6, 9, 1)]

    urls_2022 = ['https://www.cneeex.com/qgtpfqjy/mrgk/2022n/index.shtml'] + \
                [f'https://www.cneeex.com/qgtpfqjy/mrgk/2022n/index_{i}.shtml' for i in range(2, 6, 1)]

    results = get_page_data(urls_2022)
    data = pd.DataFrame(results, columns=['url', 'title', 'content'])
    data.to_excel('experiment/china-ets-raw-data-20220420.xlsx', index=False)

    records = list(data.content.apply(get_trade_data).values)
    records = pd.DataFrame(records)
    records = records.sort_values('date')

    records.to_excel('experiment/china-ets-kline-data-20220420.xlsx', index=False)

