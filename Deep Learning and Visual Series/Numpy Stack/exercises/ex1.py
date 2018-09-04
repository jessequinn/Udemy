import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.3, 0.6, 0.1],
              [0.5, 0.2, 0.3],
              [0.4, 0.1, 0.5]])

v = np.array([1/3, 1/3, 1/3])

s = []

for i in range(25):
    v_prime = v.dot(A)
    s.append(np.linalg.norm(v_prime - v))
    v = v_prime

plt.plot(s)
plt.show()
