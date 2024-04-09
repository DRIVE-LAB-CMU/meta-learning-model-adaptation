import numpy as np
import matplotlib.pyplot as plt

raceline = np.load('carla_raceline.npz')
plt.plot(raceline['x'],raceline['y'])
center = np.loadtxt('carla_center.txt',delimiter=',').T
plt.plot(center[:,0],center[:,1])
plt.axis('equal')
plt.show()