import numpy as np
import pandas as pd
# fix
pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data
import fix_yahoo_finance as yf
import arrow

# import plotly.plotly as py
import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)
cf.go_offline()

#  set time variables - 10 years from now
start = arrow.now().shift(years=-12).format('YYYY-MM-DD')
end = arrow.now().format('YYYY-MM-DD')

yf.pdr_override()

# Bank of America
BAC = data.get_data_yahoo("BAC", start, end)

# CitiGroup
C = data.get_data_yahoo("C", start, end)

# Goldman Sachs
GS = data.get_data_yahoo("GS", start, end)

# JP Morgan Chase
JPM = data.get_data_yahoo("JPM", start, end)

# Morgan Stanley
MS = data.get_data_yahoo("MS", start, end)

# Wells Fargo
WFC = data.get_data_yahoo("WFC", start, end)

# setup keys for pd.concat
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

# dataframe setup
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)

# name columns
bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']

# heatmap plot of closing prices
close_corr = bank_stocks.xs(key='Close', axis=1, level='Stock Info').corr()
plot([go.Heatmap(z=close_corr.values.tolist(), colorscale='rdylbu')], image='png', filename='tmp/heatmap_example.html')

# candlestick plot of open, close, low, high for 2015
plot([go.Candlestick(x=BAC.loc['2015-01-01':'2016-01-01'].index.get_level_values('Date').tolist(),
                     open=BAC.Open.values.tolist(),
                     high=BAC.High.values.tolist(),
                     low=BAC.Low.values.tolist(),
                     close=BAC.Close.values.tolist())], image='png', filename='tmp/candlestick_example.html')

# fig = MS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')
# BAC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll')
# plotly.offline.plot(fig, image='png')
