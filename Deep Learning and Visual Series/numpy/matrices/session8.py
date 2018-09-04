# coding: utf-8
import numpy as np
A = np.array([[1,1],[1.5,4]])
b = np.array([2200,5050])
x = np.linalg.solve(A,b)
x
