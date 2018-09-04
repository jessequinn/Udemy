import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('./data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)

# weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# correlation R^2
d1 = Y - Yhat
d2 = Y - Y.mean()

ax.plot(sorted(X[:, 0]), sorted(X[:, 1]), sorted(Yhat))

plt.figure(2)
plt.scatter(X[:, 0], Y)
plt.plot(sorted(X[:, 0]), sorted(Yhat))

plt.figure(3)
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))

# = 1 - SS_res / SS_tot
# SS_res = E(y_i - yhat_i)^2
# SS_tot = E(y_i - y_bar)^2
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print('The R^2: {}'.format(r2))

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('multiple_linear_regression_figure_' + str(i) + '.png')
    plt.close()
