import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
import datetime

# import matplotlib.pyplot as plt
# import seaborn as sns

st = datetime.datetime(2016, 1, 1)
en = datetime.datetime(2018, 9, 24)
df = data.DataReader(name='AAPL', data_source='iex', start=st, end=en)
df.index = df.index.map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

p = figure(x_axis_type='datetime', width=1000, height=300, sizing_mode='scale_width')

p.title.text = "Candlestick Chart"
hours_12 = 12 * 60 * 60 * 1000
p.grid.grid_line_alpha = 0


def inc_dec(c, o):
    if c > o:
        value = "Increase"
    elif c < o:
        value = "Decrease"
    else:
        value = "Equal"
    return value


df['Status'] = [inc_dec(c, o) for c, o in zip(df.close, df.open)]
df['Middle'] = (df.open + df.close) / 2
df['Height'] = abs(df.open - df.close)

print(df.head())

p.segment(df.index, df.high, df.index, df.low, color="Black")

p.rect(df.index[df.Status == 'Increase'],
       df.Middle[df.Status == 'Increase'],
       hours_12,
       df.Height[df.Status == 'Increase'],
       fill_color="#CCFFFF",
       line_color="black")

p.rect(df.index[df.Status == 'Decrease'],
       df.Middle[df.Status == 'Decrease'],
       hours_12,
       df.Height[df.Status == 'Decrease'],
       fill_color="#FF3333",
       line_color="black")

# output_file("candlestick.html")
script1, div1, = components(p)

cdn_js = CDN.js_files
cdn_css = CDN.css_files

print(div1)

print(cdn_css)
# show(p)
