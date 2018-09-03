import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# ignore warnings
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

iris = sns.load_dataset('iris')

sns.pairplot(data=iris, hue='species', palette='Set1')

plt.figure(2, figsize=(10, 6))
setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'],
            cmap='plasma', shade=True, shade_lowest=False)

X = iris.drop('species', axis=1)

y = iris['species']

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

param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}

grid = GridSearchCV(SVC(), param_grid, verbose=2)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)

print('*' * 40)
print('SVM with GridSearchCV')
print('*' * 40)
print(classification_report(y_test, grid_predictions))
print('*' * 40)
print(confusion_matrix(y_test, grid_predictions))
print('*' * 40)

# save plots
for i in range(1, 3):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('svm_project_figure_' + str(i) + '.png')
    plt.close()
