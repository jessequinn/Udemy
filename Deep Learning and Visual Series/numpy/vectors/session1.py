# coding: utf-8
import numpy as np
L = [1, 2, 3]
A = np.array([1, 2, 3])

for e in L:
    print e

for e in L:
    print(e)

for e in A:
    print(e)

L.append(4)
L
A.append(4)
L = L + [5]
L
A = A + [4, 5]
L2 = []
for e in L:
    L2.append(e + e)

L2
A + A
2*A
2*L
L**2
A**2
L2 = []
for e in L:
    L2.append(e*e)

L2
np.sqrt(A)
np.log(A)
np.exp(A)
