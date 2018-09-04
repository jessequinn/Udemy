# coding: utf-8
import pandas as pd

A = pd.read_csv('data_1d.csv', header=None).values
x = A[:,0]

y = A[:,1]

import matplotlib.pyplot as plt

plt.hist(x)
plt.show()

import numpy as np
R = np.random.random(10000)

plt.hist(R)
plt.show()

plt.hist(R, bins=20)
plt.show()

y_actual = 2*x+1

residuals = y - y_actual

plt.hist(residuals)
plt.show()