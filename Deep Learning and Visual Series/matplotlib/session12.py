# coding: utf-8
import pandas as pd
A = pd.read_csv('data_1d.csv', header=None).values
A.head()
A
x = A[:,0]
y = A[:,1]
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.show()
import numpy as np
x_line = np.linspace(0, 100, 100)
y_line = 2*x_line +1
plt.plot(x_line,y_line)
plt.show()
plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.show()
