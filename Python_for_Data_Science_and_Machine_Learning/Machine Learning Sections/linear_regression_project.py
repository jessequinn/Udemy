import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)


# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

customers = pd.read_csv('Linear-Regression/Ecommerce Customers')

# print(customers.head())
# customers.info()
# print(customers.describe)

# setup multiple plots
fig, ax = plt.subplots(1, 2, figsize=(12, 3))

# sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')
# sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent')
# sns.jointplot(data=customers, x='Time on App', y='Length of Membership', kind='hex')
# sns.pairplot(customers)

# sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

columns = list(customers.columns)
columns.remove('Yearly Amount Spent')
columns.remove('Email')
columns.remove('Address')
columns.remove('Avatar')

# pass in a list of column names - features
X = customers[columns]

# target variable - what i want to predict
y = customers['Yearly Amount Spent']

# common to do 30% - 40% for test_size, random splits = 101
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# print('*' * 40)
# create linear regression model object
lrm = LinearRegression()
lrm.fit(X_train, y_train)
# print('Intercept: {}\nCoefficients: {}'.format(lrm.intercept_, lrm.coef_))

predictions = lrm.predict(X_test)

# check if prediction is good
ax[0].scatter(y_test, predictions)
ax[0].set_xlabel('Y Test (True Values)')
ax[0].set_ylabel('Predicted Values')

# histogram of the residuals
# sns.distplot((y_test-predictions))

# look at various types of error
print('*' * 40)
print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test, predictions)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test, predictions)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))
print('Variance Score (R^2): {}'.format(np.sqrt(metrics.explained_variance_score(y_test, predictions))))
print('*' * 40)

# residuals
sns.distplot((y_test-predictions), bins=50, ax=ax[1])

# coefficients
cdf = pd.DataFrame(lrm.coef_,columns,columns=['Coeff'])
print(cdf)
print('*' * 40)

# tighten up layout
plt.tight_layout()

# show plot
plt.show()

# save plots
fig.savefig("machine_learning_lrm.png")
