import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["patch.force_edgecolor"] = True

#############################################
# configuration
#############################################

# samples
N = 100000

# random variables
random_variables = ([1, 5, 10, 20, 50])
#############################################


fig, axs = plt.subplots(nrows=len(random_variables), ncols=1, sharex=True)

def central_limit_theorem(rv, N, index, size):
    x = np.zeros((N))

    for i in range(N):
        for j in range(rv):
            x[i] += np.random.random()
        x[i] *= 1/rv

    axs[index].hist(x, 100, density=True, label=" %d RVs" % (rv))

    variance = np.var(x)
    mean = np.mean(x)

    print('[{}] Mean: {}, and Variance: {}'.format(index, mean, variance))

    # plot gaussian pdf
    sigma = 1/np.sqrt(2*np.pi*variance)
    dist = sigma*np.exp(-(np.linspace(0, 1, 100)-mean)**2/(2*variance))
    axs[index].plot(np.linspace(0, 1, 100), dist, label='CLT')
    axs[index].legend(loc=1)

for i in range(np.size(random_variables)):
    central_limit_theorem(
        random_variables[i], N, i, np.size(random_variables))

plt.show()
