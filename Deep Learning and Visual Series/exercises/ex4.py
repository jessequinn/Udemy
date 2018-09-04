import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["patch.force_edgecolor"] = True

images = pd.read_csv('./train.csv').values

for i in range(10):
    number_mean = np.mean(images[images[:, 0] == i][:, 1:], axis=0)
    number_mean = number_mean.reshape(28, 28)
    number_mean = np.rot90(number_mean, k=3)

    plt.imshow(number_mean, cmap='gist_gray')
    plt.tight_layout()
    plt.savefig('./tmp/figure_' + str(i) + '.png')
    plt.close()
