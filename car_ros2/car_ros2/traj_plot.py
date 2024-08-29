import numpy as np
import matplotlib.pyplot as plt
import os

arr = np.loadtxt('track.csv')
plt.plot(arr[:,0], arr[:,1])
plt.axis('equal')
plt.show()
