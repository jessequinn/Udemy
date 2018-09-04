import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('./data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:, 1], Y)

# weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# correlation R^2
d1 = Y - Yhat
d2 = Y - Y.mean()

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
    plt.savefig('polynomial_regression_figure_' + str(i) + '.png')
    plt.close()
