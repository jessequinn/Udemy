import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

loans = pd.read_csv('Decision-Trees-and-Random-Forests/loan_data.csv')

print(loans.head())
print('-'*40)

plt.figure(1, figsize=(10, 6))
loans[loans['credit.policy'] == 1]['fico'].hist(
    bins=35, color='blue', label='Credit Policy = 1', alpha=0.6)
loans[loans['credit.policy'] == 0]['fico'].hist(
    bins=35, color='red', label='Credit Policy = 0', alpha=0.6)
plt.grid(False)

plt.legend()
plt.xlabel('FICO')

plt.figure(2, figsize=(10, 6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(
    bins=35, color='blue', label='Not Fully Paid = 1', alpha=0.6)
loans[loans['not.fully.paid'] == 0]['fico'].hist(
    bins=35, color='red', label='Not Fully Paid = 0', alpha=0.6)
plt.grid(False)

plt.legend()
plt.xlabel('FICO')

plt.figure(3, figsize=(10, 6))
sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')

sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')

sns.lmplot(data=loans, y='int.rate', x='fico', hue='credit.policy',
           col='not.fully.paid', palette='Set1')

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

# print(final_data.info())

X = final_data.drop('not.fully.paid', axis=1)

y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

t_predictions = dtree.predict(X_test)

print('*' * 40)
print('Decision Tree')
print('*' * 40)
print(classification_report(y_test, t_predictions))
print('*' * 40)
print(confusion_matrix(y_test, t_predictions))
print('*' * 40)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)

print('*' * 40)
print('Random Forest')
print('*' * 40)
print(classification_report(y_test, rfc_predictions))
print('*' * 40)
print(confusion_matrix(y_test, rfc_predictions))
print('*' * 40)

# show plot
# plt.show()

# save plots
for i in range(1, 6):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('tree_project_figure_' + str(i) + '.png')
    plt.close()
