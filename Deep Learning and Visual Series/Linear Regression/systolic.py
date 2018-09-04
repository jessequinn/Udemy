# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('mlr02.xls')
X = df.values

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

# using years
axs[0].scatter(X[:, 1], X[:, 0])
# axs[2].scatter(X[:, 1], X[:, 0])

# using weight
axs[1].scatter(X[:, 2], X[:, 0])
# axs[3].scatter(X[:, 2], X[:, 0])

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

# weighted


def get_r2(X, Y, index):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)

    # correlation
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    axs[index].plot(sorted(X.iloc[:, 0]), sorted(Yhat),
                    label=r' %0.3f $\mathregular{R^2}$' % (r2))
    axs[index].legend(loc=0)
    return r2


print("R^2 for x2 only:", get_r2(X2only, Y, 0))
print("R^2 for x3 only:", get_r2(X3only, Y, 1))
# print("R^2 for both:", get_r2(X, Y, 2))
# print("R^2 for both:", get_r2(X, Y, 3))


# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('systolic_figure_' + str(i) + '.png')
    plt.close()
