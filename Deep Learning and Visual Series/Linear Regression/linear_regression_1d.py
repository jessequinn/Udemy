'''
initial linear regresion example
'''

import numpy as np
import matplotlib.pyplot as plt

# load data
X = []
Y = []

for line in open('./data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

denominator = X.dot(X) - X.mean() * X.sum()

# = (X dot Y - X_barY_bar)/(X dot X-X_bar^2)
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator

# = (Y X dot X - X X dot Y)/(X dot X - X^2)
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# = a X + b
Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# correlation R^2
d1 = Y - Yhat
d2 = Y - Y.mean()

# = 1 - SS_res / SS_tot
# SS_res = E(y_i - yhat_i)^2
# SS_tot = E(y_i - y_bar)^2
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print('The R^2: {}'.format(r2))
