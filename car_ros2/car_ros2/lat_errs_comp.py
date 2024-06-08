import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle

exp_names = ['short_horizon_kin_model','short_horizon_large_alpha','short_horizon_less_alpha','large_horizon_large_alpha']
t = 10

for exp_name in exp_names:
    filename = 'data/' + exp_name + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        plt.plot(np.array(data['lat_errs']),label=str(exp_name))
plt.xlabel('Iter')
plt.ylabel('Lateral error')
plt.title("Lateral error (in m)")
plt.legend()
plt.savefig('lat_errs.png')
plt.show()