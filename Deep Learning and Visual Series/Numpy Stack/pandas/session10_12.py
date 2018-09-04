# coding: utf-8
import pandas as pd
df = pd.read_csv('data_2d.csv')
df.head()
df = pd.read_csv('data_2d.csv', header=None)
df.head()
type(df)
df.info()
M = X.as_matrix()
M = df.as_matrix()
type(M)
M = df.value()
M = df.values()
M = df.values
type(M)
df[0]
df.head()
df[0]
type(df[0])
type(M[0])
df.iloc[0]
df.ix[0]
df[0,2]
df[[0,2]]
df[df[0] < 5]
df = pd.read_csv('international-airline-passengers.csv',engine="python",skipfooter=3)
df.head()
df.columns = ["month", "passengers"]
df.head()
df.passengers
df['ones'] =1
df.head()
from datetime import datetime
datetime.strptime("1949-05",%Y-%m)
datetime.strptime("1949-05","%Y-%m")
df['dt'] = df.apply(lambda row: datetime.strptime(row['month
df['dt'] = df.apply(lambda row: datetime.strptime(row['month'], "%Y-%m"), axis=1)
df.info()
df.head()
t1 = pd.read_csv('table1.csv')
t2 = pd.read_csv('table2.csv')
m = pd.merge(t1, t2, on='user_id')
m.head()
m
t1
t2
t1.merger(t2, on='user_id')
t1.merge(t2, on='user_id')
