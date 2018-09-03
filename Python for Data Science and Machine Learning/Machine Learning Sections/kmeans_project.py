import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# ignore warnings of lapack bug
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

df = pd.read_csv('./K-Means-Clustering/College_Data', index_col=0)

# print(df.head())
# print(df.info())

# plt.figure(1, figsize=(10, 6))
sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private',
           fit_reg=False, palette='coolwarm', size=6, aspect=1.67)

sns.lmplot(x='Outstate', y='F.Undergrad', data=df, hue='Private',
           fit_reg=False, palette='coolwarm', size=6, aspect=1.67)

g = sns.FacetGrid(data=df, hue='Private',
                  palette='coolwarm', size=6, aspect=1.67)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)

# higher than 100
# df[df['Grad.Rate']>100]
df.at['Cazenovia College', 'Grad.Rate'] = 100

g = sns.FacetGrid(data=df, hue='Private',
                  palette='coolwarm', size=6, aspect=1.67)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))

# print(kmeans.cluster_centers_)
# print(kmeans.labels_)

# function to convert to binary 1 0 column
def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Private'].apply(converter)

print('*' * 40)
print(classification_report(df['Cluster'], kmeans.labels_))
print('*' * 40)
print(confusion_matrix(df['Cluster'], kmeans.labels_))
print('*' * 40)


# show plot
# plt.show()

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('kmeans_figure_' + str(i) + '.png')
    plt.close()
