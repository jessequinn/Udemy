import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

df = pd.read_csv('K-Nearest-Neighbors/KNN_Project_Data', index_col=0)

print(df.head())

# use seaborn 0.8 as hue was removed in 0.9
# sns_plot = sns.pairplot(df, hue='TARGET CLASS', palette='coolwarm')

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

# remove TARGET CLASS
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

error_rate = []

# n_neighors = 37 has the lowest error rate
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    predictions_i = knn.predict(X_test)
    # avg not equal to actual y_test values
    error_rate.append(np.mean(predictions_i != y_test))

knn = KNeighborsClassifier(n_neighbors=37)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print(classification_report(y_test, predictions))
print('*' * 40)
print(confusion_matrix(y_test, predictions))
print('*' * 40)

fig = plt.figure(figsize=(10, 6))
plt.plot(range(1, 100), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# tighten up layout
plt.tight_layout()

# show plot
plt.show()

# save plots
# sns_plot.savefig("machine_learning_knn.png")
fig.savefig("machine_learning_knn2.png")
