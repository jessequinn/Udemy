import numpy as np
import pandas as pd
# fix
pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data, wb
import fix_yahoo_finance as yf
import datetime
import arrow

# visual stuff
import matplotlib.pyplot as plt
import seaborn as sns

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

# load pickle database of bank information
# pd.read_pickle('tmp/all_banks')

# set time variables - 10 years from now
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

# print(bank_stocks.head())
# for tick in tickers:
#     print(tick, bank_stocks[tick]['Close'].max())

# returns max stock close value
# print(bank_stocks.xs(key='Close', axis=1, level='Stock Info').max())

returns = pd.DataFrame()

for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()

# print(returns.head())

# pairplot of all returns - removed first row due to NaN
# sns.pairplot(returns[1:])

# returns single worse loss/best gain
# print(returns.idxmin())
# print(returns.idxmax())

print('*' * 40)

# std deviation
print(returns.std())

print('*' * 40)

# std deviation for 2015
print(returns.loc['2015-01-01':'2015-12-31'].std())

print('*' * 40)

# setup multiple plots
fig, ax = plt.subplots(3, 2, figsize=(12, 8))

# sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'], color='green', bins=50, hist_kws=dict(edgecolor="k", linewidth=2))
s1 = sns.distplot(returns.loc['2015-01-01':'2015-12-31']
                  ['MS Return'], color='green', bins=50, ax=ax[0][0])
s1.set(xlabel='Morgan Stanley Returns')

s2 = sns.distplot(returns.loc['2008-01-01':'2008-12-31']
                  ['C Return'], color='red', bins=50, ax=ax[0][1])
s2.set(xlabel='CitiGroup Returns')

# lineplot
# for tick in tickers:
#     ax[1] = bank_stocks[tick]['Close'].plot(label=tick)

bank_stocks.xs(key='Close', axis=1, level='Stock Info').plot(ax=ax[1][0])
ax[1][0].legend(loc=1)

# print(bank_stocks.xs(key='Close', axis=1, level='Stock Info'))
# plt.legend()

# s3 = sns.heatmap(data=bank_stocks.xs(key='Close', axis=1, level='Stock Info'), ax=ax[1][0])


# moving averages

BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(
    window=30).mean().plot(ax=ax[1][1], label='30 day rolling average')
BAC['Close'].loc['2008-01-01':'2009-01-01'].plot(
    ax=ax[1][1], label='BAC Close')
ax[1][1].legend(loc=1)

# correlation of closing/opening stock prices between banks
s4 = sns.heatmap(bank_stocks.xs(key='Close', axis=1,
                                level='Stock Info').corr(), annot=True, ax=ax[2][1])
s4.set(xlabel='Bank', ylabel='Bank')

# s5 = sns.heatmap(bank_stocks.xs(key='Open', axis=1,
#                                 level='Stock Info').corr(), annot=True, ax=ax[2][0])
# s5.set(xlabel='Bank', ylabel='Bank')

# sns.clustermap(bank_stocks.xs(key='Close', axis=1,
#                               level='Stock Info').corr(), annot=True)

C['Close'].loc['2016-01-01':'2017-01-01'].rolling(
    window=30).mean().plot(ax=ax[2][0], label='30 day rolling average')
C['Close'].loc['2016-01-01':'2017-01-01'].plot(
    ax=ax[2][0], label='C Close')
ax[2][0].legend(loc=1)

# tighten up layout
fig.tight_layout()

# add title
fig.suptitle("Finance Report - 6 Major Banks")

# adjust space for title
fig.subplots_adjust(top=0.9)

# show plot
plt.show()

# save plots
fig.savefig("seaborn_example2.png")
