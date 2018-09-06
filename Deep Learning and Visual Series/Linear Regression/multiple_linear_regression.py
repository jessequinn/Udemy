import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('./data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

# weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
print('W shape: {}'.format(w.shape))

Yhat = np.dot(X, w)
print('Yhat shape: {}'.format(Yhat.shape))

# correlation R^2
d1 = Y - Yhat
d2 = Y - Y.mean()

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], Y, c='r', s=50)

X = []
Y = []

for line in open('./data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
# print('W shape: {}'.format(w.shape))

xx, yy = np.meshgrid(X[:, 0], X[:, 1])

# plt.figure(2)
# plt.plot(xx, yy, marker='.', color='k', linestyle='none')

XYpairs = np.vstack((xx.flatten(), yy.flatten())).T

print('XYpairs shape: {}'.format(XYpairs.shape))

# Z = np.tile(Y, 100).flatten()
Z = XYpairs.dot(w)

print('Z shape: {}'.format(Z.shape))

# Yhat = np.dot(XYpairs, w)
# print('Yhat shape: {}'.format(Yhat.shape))

plt.figure(1)
ax.plot_trisurf(XYpairs[:, 0], XYpairs[:, 1], Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')


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
