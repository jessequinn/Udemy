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

Capstone Project:

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

"""Updated to latest 911 data from kaggle.
"""
df = pd.read_csv('tmp/911.csv')

# df.info()

# print(df.head(5))

# print(df['zip'].value_counts().head())

# print(df['twp'].value_counts().head())

# print(df['title'].nunique())

df['reason'] = df['title'].apply(lambda reason: reason.split(':')[0])

# print(df['reason'].value_counts().head(1))

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
s1 = sns.countplot(x='reason', data=df, palette='viridis', ax=ax[0][0])
s1.set(xlabel='Reason', ylabel='Count')

# print(type(df['timeStamp'].iloc[0]))
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
# print(type(df['timeStamp'].iloc[0]))

# time = df['timeStamp'].iloc[0]
# print(time.hour)

df['hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['month'] = df['timeStamp'].apply(lambda time: time.month)
df['dayofweek'] = df['timeStamp'].apply(lambda time: time.dayofweek)

# print(df.head())

dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['dayofweek'] = df['dayofweek'].map(dmap)

# print(df.head())

s2 = sns.countplot(x='dayofweek', data=df, hue='reason',
                   palette='viridis', ax=ax[0][1])
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
# s2.legend_.remove()

s2.set(xlabel='Day of Week', ylabel='Count')
s2.legend(title='Reason')
# s2.set_title('Reason')

# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# s3 = sns.countplot(x='month', data=df, hue='reason',
#                    palette='viridis', ax=ax[2])
# s3.set(xlabel='Month', ylabel='Count')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# mix missing month data
byMonth = df.groupby('month').count()
# print(byMonth.head(12))

# s3 = sns.lineplot(x='month', y='lat', data=byMonth.reset_index(),
#                   palette='viridis', ax=ax[1][0])
# s3.set(xlabel='Month', ylabel='Latitude')
# s3.legend()
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# byMonth['lat'].plot()

# t = df['timeStamp'].iloc[0]
# print(t.date())
df['date'] = df['timeStamp'].apply(lambda time: time.date())

# print(df.head())

# print(df.groupby('date').count()['lat'])
# df.groupby('date').count()['lat'].plot()
# df[df['reason'] == 'Traffic'].groupby('date').count()['lat'].plot()
# df[df['reason'] == 'Fire'].groupby('date').count()['lat'].plot()
# df[df['reason'] == 'EMS'].groupby('date').count()['lat'].plot()

dayHour = df.groupby(by=['dayofweek', 'hour']).count()['reason'].unstack()
dayMonth = df.groupby(by=['dayofweek', 'month']).count()['reason'].unstack()

# print(dayHour)

s4 = sns.heatmap(dayHour, cmap='viridis', ax=ax[1][1])
s4.set(xlabel='Hour', ylabel='Day of Week')

s3 = sns.heatmap(dayMonth, cmap='viridis', ax=ax[1][0])
s3.set(xlabel='Month', ylabel='Day of Week')

# s5 = sns.lmplot(x='month', y='twp',
#                 data=byMonth.reset_index(), palette='viridis')
# s5.set(xlabel='Month', ylabel='Township')

# s4a = sns.clustermap(dayHour, cmap='viridis')
# s4a.set(xlabel='Hour', ylabel='Day of Week')

fig.tight_layout()
fig.suptitle("Emergency - 911 Calls - Montgomery County, PA")
fig.subplots_adjust(top=.9)
plt.show() 

fig.savefig("seaborn_example1.png")
# s5.savefig("seaborn_lmplot_example.png")

```


![capstone project](img/seaborn_example1.png "Capstone Project")
