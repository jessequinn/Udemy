import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["patch.force_edgecolor"] = True


def cartesian(radius, rad):
    tmp = np.zeros((rad.shape[0], 2))
    tmp[:, 0] = radius * np.cos(rad)
    tmp[:, 1] = radius * np.sin(rad)
    return tmp


points = 500
r1 = 1  # Inner circle radius = 1
r2 = 2  # Outer circle radius = 2
rad = np.linspace(0, 2 * np.pi, points)
print(rad.shape[0])

c1 = cartesian(r1, rad)
c2 = cartesian(r2, rad)


def noise(x): return x + np.random.normal(0, 0.1)


vfunc = np.vectorize(noise)
c1 = vfunc(c1)
c2 = vfunc(c2)

plt.scatter(c1[:, 0], c1[:, 1], color='blue')
plt.scatter(c2[:, 0], c2[:, 1], color='red')
plt.axis('equal')
plt.show()
