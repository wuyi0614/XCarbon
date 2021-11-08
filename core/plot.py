# -*- encoding: utf-8 -*-
#
# Created at 04/11/2021 by Yi Wu, wymario@163.com
#

from pyecharts.charts import Kline, Bar, Grid
from pyecharts import options as opts


def plot_kline(array, dates, ylabel='kline', name='kline-example', render=False):
    c = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis(ylabel, array,
                   itemstyle_opts=opts.ItemStyleOpts(
                       color="#ec0000",
                       color0="#00da3c",
                       border_color="#8A0000",
                       border_color0="#008F28",
                    )
                )
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=[opts.DataZoomOpts(pos_bottom='-2%')],
            title_opts=opts.TitleOpts(title=f'daily {name} kline', pos_left='10%'),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )
    if render:
        c.render(f'{name}.html')

    return c


def plot_volume(volumes, dates, name='bar-volume-example', render=False):
    c = (
        Bar()
            .add_xaxis(dates)
            .add_yaxis("trade volume", volumes)
            .set_global_opts(
            title_opts=opts.TitleOpts(title=f'daily {name} volume', pos_right='10%'),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(is_show=False)
        ).set_series_opts(markpoint_opts=opts.MarkPointItem(value=None))
    )
    if render:
        c.render(f'{name}.html')

    return c


def plot_grid(prices, volumes, render=False):
    grid = Grid()
    grid.add(prices, grid_opts=opts.GridOpts(pos_bottom='60%'))
    grid.add(volumes, grid_opts=opts.GridOpts(pos_top='60%'))
    if render:
        grid.render('carbon-market-grid.html')

    return grid
