# -*- encoding: utf-8 -*-
#
# Created at 04/11/2021 by Yi Wu, wymario@163.com
#

from pyecharts.charts import Kline
from pyecharts import options as opts


def plot_kline(array, dates, ylabel='kline', name='kline-example'):
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
            title_opts=opts.TitleOpts(title=f'daily-{name}-kline'),
        )
            .render(f'{name}.html')
    )
