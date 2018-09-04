# coding: utf-8
import numpy as np
A = np.array([[1,2],[3,4]])
Ainv = np.linalg.inv(A)
A
Ainv
Ainv.dot(A)
A.dot(Ainv)
np.linalg.det(A)
np.diag(A)
np.diag([1,2])
a = np.array([1,2])
b = np.array([3,4])
np.outer(a, b)
np.innder(a,b)
np.inner(a,b)
a.dot(b)
np.diag(A).sum()
np.trace(A)
X = np.random.randn(100,3)
X
cov = np.cov(X)
cov
cov.shape
co v= np.cov(X.T)
cov= np.cov(X.T)
cov
cov.shape
np.linalg.eigh(cov)
np.linalg.eig(cov)
