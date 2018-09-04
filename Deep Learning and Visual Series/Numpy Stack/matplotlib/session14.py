# coding: utf-8
import pandas as pd
df = pd.read_csv('train.csv')
df.shape
M = df.values
im =M[0,1:]
im
im.shape
im.reshape(28,28)
im = im.reshape(28,28)
im.shape
import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()
M[0,0]
plt.imshow(im, cmap='gist_gray')
plt.show()
plt.imshow(255-im, cmap='gist_gray')
plt.show()
