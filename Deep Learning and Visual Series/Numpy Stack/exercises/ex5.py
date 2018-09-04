import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_symmetric(matrix, tol=1e-08):
    return np.allclose(matrix, matrix.T, atol=tol)

print(is_symmetric(np.array([[1e10,1e-7]])))
print(is_symmetric(np.array([[1e10,1e10]])))