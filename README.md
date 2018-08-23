# Udemy

## Advance CSS and Sass Course

Advance CSS and Sass Course folder is based on the starter package from [Jonas Schmedtmann](https://github.com/jonasschmedtmann/advanced-css-course). The folder contains three projects that have been completed.

The Natours Project requires npm for SCSS compilation.

to view the site `npm install live-server -g` and `live-server` the project.

---

## Complete Python Masterclass

[Tim](http://learnprogramming.academy)

---

## Python for Data Science and Machine Learning

Course uses [jupyter](http://jupyter.org/install).

run `jupyter notebook` within the course directory.

I use [Homebrew](#) to install [pyenv](https://anil.io/blog/python/pyenv/using-pyenv-to-install-multiple-python-versions-tox/) that controls my python environment.

Packages covered in the course:
[numpy](http://www.numpy.org)
[pandas](http://pandas.pydata.org)
[matplotlib](http://www.matplotlib.org/)
[seaborn](https://seaborn.pydata.org)
[plotly-2.7.0](https://plot.ly)
[cufflinks](https://github.com/santosjorge/cufflinks)

Example plots with [seaborn](https://seaborn.pydata.org):

![seaborn scatter plot](img/seaborn_scatter.png "Seaborn Scatter Plot")
![seaborn regression plot](img/seaborn_regression.png "Seaborn Regression Plot")

#### Capstone Project:

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# useful for jupter
# %matplotlib inline

# Updated to latest 911 data from kaggle.
df = pd.read_csv('tmp/911.csv')

# displays info about dataframe
# df.info()

# displays top 5 rows from dataframe
# print(df.head(5))

# displays top 5 rows of counts on zip code
# print(df['zip'].value_counts().head())

# displays top 5 rows of counts on township
# print(df['twp'].value_counts().head())

# displays all unique titles
# print(df['title'].nunique())

# lambda expression to split title and capture just a reason EMS, FIRE, TRAFFIC
df['reason'] = df['title'].apply(lambda reason: reason.split(':')[0])

# displays top 1 row of counts on reason column
# print(df['reason'].value_counts().head(1))

# grab fig and ax of subplot - subplot with specific dimension and plot layout 2x2
fig, ax = plt.subplots(2, 2, figsize=(12, 6))

# first plot assigned and axis labels adjusted
s1 = sns.countplot(x='reason', data=df, palette='viridis', ax=ax[0][0])
s1.set(xlabel='Reason', ylabel='Count')

# reassign format of timeStamp column
# print(type(df['timeStamp'].iloc[0]))
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
# print(type(df['timeStamp'].iloc[0]))

# make new columns with hour, month, and dayofweek
# time = df['timeStamp'].iloc[0]
# print(time.hour)
df['hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['month'] = df['timeStamp'].apply(lambda time: time.month)
df['dayofweek'] = df['timeStamp'].apply(lambda time: time.dayofweek)
# print(df.head())

# map the dayofweek column
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['dayofweek'] = df['dayofweek'].map(dmap)
# print(df.head())

# add second plot
s2 = sns.countplot(x='dayofweek', data=df, hue='reason',
                   palette='viridis', ax=ax[0][1])
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
# s2.legend_.remove()
s2.set(xlabel='Day of Week', ylabel='Count')
s2.legend(title='Reason')
# s2.set_title('Reason')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# was third plot
# s3 = sns.countplot(x='month', data=df, hue='reason',
#                    palette='viridis', ax=ax[2])
# s3.set(xlabel='Month', ylabel='Count')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# fix missing month data as seen in countplots
byMonth = df.groupby('month').count()
# print(byMonth.head(12))

# was third plot
# s3 = sns.lineplot(x='month', y='lat', data=byMonth.reset_index(),
#                   palette='viridis', ax=ax[1][0])
# s3.set(xlabel='Month', ylabel='Latitude')
# s3.legend()
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# was third plot - will append to grid system
# byMonth['lat'].plot()

# new date column
# t = df['timeStamp'].iloc[0]
# print(t.date())
df['date'] = df['timeStamp'].apply(lambda time: time.date())
# print(df.head())

# various groupbys
# print(df.groupby('date').count()['lat'])
# df.groupby('date').count()['lat'].plot()
# df[df['reason'] == 'Traffic'].groupby('date').count()['lat'].plot()
# df[df['reason'] == 'Fire'].groupby('date').count()['lat'].plot()
# df[df['reason'] == 'EMS'].groupby('date').count()['lat'].plot()

# plot 3 and 4 data
dayHour = df.groupby(by=['dayofweek', 'hour']).count()['reason'].unstack()
dayMonth = df.groupby(by=['dayofweek', 'month']).count()['reason'].unstack()
# print(dayHour)

# heatmaps plots 3 and 4
s4 = sns.heatmap(dayHour, cmap='viridis', ax=ax[1][1])
s4.set(xlabel='Hour', ylabel='Day of Week')

s3 = sns.heatmap(dayMonth, cmap='viridis', ax=ax[1][0])
s3.set(xlabel='Month', ylabel='Day of Week')

# independent lmplot
# s5 = sns.lmplot(x='month', y='twp',
#                 data=byMonth.reset_index(), palette='viridis')
# s5.set(xlabel='Month', ylabel='Township')

# independent clustermap
# s4a = sns.clustermap(dayHour, cmap='viridis')
# s4a.set(xlabel='Hour', ylabel='Day of Week')

# tighten up layout
fig.tight_layout()

# add title
fig.suptitle("Emergency - 911 Calls - Montgomery County, PA")

# adjust space for title
fig.subplots_adjust(top=0.9)

# show plot
plt.show()

# save plots
fig.savefig("seaborn_example1.png")
# s5.savefig("seaborn_lmplot_example.png")
```

![capstone project](img/seaborn_example1.png "Capstone Project")

#### Finance Project:

```Python
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
```

![finance project](img/seaborn_example2.png "Finance Project")
