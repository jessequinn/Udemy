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
