import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

cancer = load_breast_cancer()

# print(cancer.keys())
# print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

print(df_feat.head())
print('-'*40)

X = df_feat

y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

svm = SVC()

svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

print('*' * 40)
print('SVM without GridSearchCV')
print('*' * 40)
print(classification_report(y_test, svm_predictions))
print('*' * 40)
print(confusion_matrix(y_test, svm_predictions))
print('*' * 40)

param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]} 

grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)

print('*' * 40)
print('SVM with GridSearchCV')
print('*' * 40)
print(classification_report(y_test, grid_predictions))
print('*' * 40)
print(confusion_matrix(y_test, grid_predictions))
print('*' * 40)
