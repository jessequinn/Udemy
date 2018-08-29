import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

cancer.keys()

print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

print(df.head())

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca.shape)

plt.figure(1, figsize=(10, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])

plt.figure(2, figsize=(10,6))
sns.heatmap(df_comp, cmap='plasma')
plt.xticks(rotation='vertical')
# plt.tight_layout()
# plt.show()

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('pca_figure_' + str(i) + '.png')
    plt.close()