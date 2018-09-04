# coding: utf-8
import numpy as np
A = np.array([[1,2],[3,4]])
A
b = np.array([1,2])
b
x = np.linalg.inv(A).dot(b)
X
x
x = np.linalg.solve(A,b)
x
# solving linear systems
