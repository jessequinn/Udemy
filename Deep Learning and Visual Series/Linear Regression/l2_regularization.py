import numpy as np
import matplotlib.pyplot as plt

# sample size
N = 50

X = np.linspace(0,10,N)

# gaussian distributed noise
Y = 0.5*X + np.random.randn(N)

# make outliers
Y[-1] += 30
Y[-2] += 30

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)

# plot the data
axs[0].scatter(X, Y)

# add bias term
X = np.vstack([np.ones(N), X]).T

# plot the maximum likelihood solution
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
axs[1].scatter(X[:,1], Y)
axs[1].plot(X[:,1], Yhat_ml)

# probably don't need an L2 regularization this high in many problems
# everything in this example is exaggerated for visualization purposes
l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
axs[2].scatter(X[:,1], Y)
axs[2].plot(X[:,1], Yhat_ml, label='maximum likelihood')
axs[2].plot(X[:,1], Yhat_map, label='map')
axs[2].legend()

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('l2_regularization_figure_' + str(i) + '.png')
    plt.close()