import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

df = pd.read_csv('Linear-Regression/USA_Housing.csv')

# grab information about data
# print(list(df.columns))
# df.info()
# print(df.describe())

# plots to view data
# sns.pairplot(df)
# sns.distplot(df['Price'])
# sns.heatmap(df.corr(), annot=True)

columns = list(df.columns)
columns.remove('Price')
columns.remove('Address')
# pass in a list of column names - features
X = df[columns]

# target variable - what i want to predict
y = df['Price']

# common to do 30% - 40% for test_size, random splits = 101
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101)

print('*' * 40)
# create linear regression model object
lrm = LinearRegression()
lrm.fit(X_train, y_train)
print('Intercept: {}\nCoefficients: {}'.format(lrm.intercept_, lrm.coef_))

print('*' * 40)
# coefficients
cdf = pd.DataFrame(lrm.coef_,columns,columns=['Coeff'])
print(cdf)

# real data 1970s housing
# boston = load_boston()
# boston.keys()
# print(boston['DESCR'])



# tighten up layout
plt.tight_layout()

# show plot
# plt.show()
