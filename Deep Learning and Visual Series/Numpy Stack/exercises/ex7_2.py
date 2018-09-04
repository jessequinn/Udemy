'''
Generate and plot this dataset (donuts/ concentric circles)
'''

import numpy as np
import matplotlib.pyplot as plt

# giving theta and r, get x and y coordinates of a point to be plot


def getCoordinates(theta, r):
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return (x, y)


N = 500

# generate float theta randomly(0 <= r < 360)
theta = np.random.random(N) * np.pi * 2

noise = np.random.normal(0, 1, N)

# generate the inside circle points, add some noise to r
inside_circle = np.asarray(getCoordinates(theta, 10 + noise))

# generate outside circle points, add some noise to r
outside_circle = np.asarray(getCoordinates(theta, 20 + noise))

# concatenate of 2D nparray, horizontally
donuts = np.concatenate((inside_circle, outside_circle), axis=1)

# generate colors array, first N elements are 'darkblue', next N elements are 'darkred'
colors = np.append(np.full(N, fill_value='darkblue'),
                   np.full(N, fill_value='darkred'))


plt.scatter(donuts[0, :], donuts[1, :], c=colors, alpha=0.5)
plt.axis('equal')
plt.show()
