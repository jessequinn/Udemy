# coding: utf-8
from scipy.stats import norm
norm.pdf(0)
norm.pdf(0, loc=5, scale=10)
import numpy as np
r = np.random.randn(10)
norm.pdf(r)
norm.logpdf(r)
norm.logcdf(r)
norm.cdf(r)
r = np.random.randn(10000)
import matplotlib.pyplot as plt
plt.hist(r)
plt.show()
r = 10*np.random.randn(10000) + 5
# std. dev 10 and mean of 5
plt.hist(r)
plt.show()
plt.show()
plt.hist(r)
plt.hist(r)
plt.show()
r = np.random.randn(10000, 2)
plt.scatter(r[:,0],r[:,1])
plt.show()
r[:,1] = 5*r[:,1]+2
plt.scatter(r[:,0],r[:,1])
plt.show()
plt.scatter(r[:,0],r[:,1])
plt.axis('equal')
plt.show()
cov = np.array([[1,0.8]],[0.8,3]])
cov = np.array([[1,0.8],[0.8,3]])
from scipy.stats import multivariate_normal as mvn
mu = np.array([0,2])
r = mvn.rvs(mean=mu, cov=cov, size=1000)
plt.scatter(r[;,0],r[:,1])
plt.scatter(r[:,0],r[:,1])
plt.show()
r = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
plt.scatter(r[:,0],r[:,1])
plt.show()
