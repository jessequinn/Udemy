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

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

df = pd.read_csv('Decision-Trees-and-Random-Forests/kyphosis.csv')

print(df.head())
print('-'*40)

# sns.pairplot(df, hue='Kyphosis')

X = df.drop('Kyphosis', axis=1)

y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

t_predictions = dtree.predict(X_test)

print(classification_report(y_test, t_predictions))
print('*' * 40)
print(confusion_matrix(y_test, t_predictions))
print('*' * 40)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)

print(classification_report(y_test, rfc_predictions))
print('*' * 40)
print(confusion_matrix(y_test, rfc_predictions))
print('*' * 40)

# tighten up layout
plt.tight_layout()

# show plot
# plt.show()

# save plots
# sns_plot.savefig("machine_learning_knn.png")
# fig.savefig("machine_learning_knn2.png")
