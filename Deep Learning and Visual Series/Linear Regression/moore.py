'''
moore's law example
'''

import re
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('./moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))

    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

axs[0].scatter(X, Y)
# plt.show()

Y = np.log(Y)
axs[1].scatter(X,Y)
# plt.show()

denominator = X.dot(X) - X.mean() * X.sum()

# = (X dot Y - X_barY_bar)/(X dot X-X_bar^2)
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator

# = (Y X dot X - X X dot Y)/(X dot X - X^2)
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# = a X + b
Yhat = a*X + b

axs[2].scatter(X, Y)
axs[2].plot(X, Yhat)

# correlation R^2
d1 = Y - Yhat
d2 = Y - Y.mean()

# = 1 - SS_res / SS_tot
# SS_res = E(y_i - yhat_i)^2
# SS_tot = E(y_i - y_bar)^2
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print('a: {} b: {} '.format(a,b))
print('The R^2: {}'.format(r2))
print('Time to double: {} years'.format(np.log(2)/a))

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('moore_figure_' + str(i) + '.png')
    plt.close()