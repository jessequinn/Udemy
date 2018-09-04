# https://github.com/qing99/machine-learning-class/tree/master/numpy_class

import numpy as np
import matplotlib.pyplot as plt


def getCoordinates(theta, r):
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return (x, y)


N = 200

# generate random t, 0 <= t < 60
t = np.random.random(N) * 1

noise = np.random.normal(0, 0.2, 2 * N).reshape(2, N)

# generate theta and r
theta = -2 * t

k = 5
r = k * t + 0.5

# generate one spiral
spiral = np.asarray(getCoordinates(theta, r)) + noise

spirals = spiral
colors = np.full(N, fill_value='darkblue')

# generate all spirals and colors array
for i in range(1, 6):
    spirals = np.concatenate((spirals, np.asarray(
        getCoordinates(theta + i * np.pi / 3, r)) + noise), axis=1)
    if i % 2 == 1:
        colors = np.append(colors, np.full(N, fill_value='darkred'))
    else:
        colors = np.append(colors, np.full(N, fill_value='darkblue'))

# plot
plt.scatter(spirals[0, :], spirals[1, :], c=colors, alpha=0.5)
plt.axis('equal')
plt.show()
