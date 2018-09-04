import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["patch.force_edgecolor"] = True

pixel = np.random.uniform(-1, 1, (5000, 2))

x_less_than_0 = pixel[pixel[:, 0] < 0]
x_greater_than_0 = pixel[pixel[:, 0] > 0]

xy_less_than_0 = x_less_than_0[x_less_than_0[:, 1] < 0]
x_less_than_0_y_greater_than_0 = x_less_than_0[x_less_than_0[:, 1] > 0]

xy_greater_than_0 = x_greater_than_0[x_greater_than_0[:, 1] > 0]
x_greater_than_0_y_less_than_0 = x_greater_than_0[x_greater_than_0[:, 1] < 0]

plt.scatter(xy_less_than_0[:, 0], xy_less_than_0[:, 1], color='blue')
plt.scatter(
    x_less_than_0_y_greater_than_0[:, 0], x_less_than_0_y_greater_than_0[:, 1], color='red')
plt.scatter(xy_greater_than_0[:, 0],
            xy_greater_than_0[:, 1], color='blue')
plt.scatter(
    x_greater_than_0_y_less_than_0[:, 0], x_greater_than_0_y_less_than_0[:, 1], color='red')

plt.show()
