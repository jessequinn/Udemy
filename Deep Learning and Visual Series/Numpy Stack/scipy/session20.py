# coding: utf-8
import numpy as np
x = np.linspace(0, 100, 10000)
y = np.sin(x) + np.sin(3*x) + np.sin(5*x)
import matplotlib.pyplot as plt
plt.plot(y)
plt.show()
# fft
Y = np.fft.fft(y)
plt.plt(np.abs(Y))
plt.plpt(np.abs(Y))
plt.plot(np.abs(Y))
plt.show()
2*np,pi*16/100
2*np.pi*16/100
2*np.pi*48/100
2*np.pi*80/100
