import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)


# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

ad_data = pd.read_csv('Logistic-Regression/advertising.csv')

print('*' * 40)
print(ad_data.head(5))
print('*' * 40)
print(ad_data.info())
print('*' * 40)

# sns.distplot(ad_data['Age'].dropna(), kde=False, bins=30)
# sns.jointplot(data=ad_data, x='Age', y='Area Income')
# sns.jointplot(data=ad_data, x='Age', y='Daily Time Spent on Site', kind='kde')
# sns.jointplot(data=ad_data, x='Daily Time Spent on Site', y='Daily Internet Usage')
sns_plot =sns.pairplot(data=ad_data)

X = ad_data[['Daily Time Spent on Site', 'Age',
             'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

logrm = LogisticRegression()
logrm.fit(X_train, y_train)

predictions = logrm.predict(X_test)
# predictions = pd.DataFrame(predictions, columns=[
                        #   'predictions'])

# print(predictions)

print(classification_report(y_test, predictions))
print('*' * 40)
print(confusion_matrix(y_test, predictions))
print('*' * 40)

# tighten up layout
plt.tight_layout()

# show plot
plt.show()

# save plots
sns_plot.savefig("machine_learning_logrm2.png")
