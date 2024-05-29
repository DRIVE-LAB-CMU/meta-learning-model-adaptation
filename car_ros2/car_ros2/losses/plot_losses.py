import numpy as np
import math
import matplotlib.pyplot as plt

exp_names = ['both_model_params_delay','const_model_params','only_delay','only_model_params_delay5','only_model_params']

for exp_name in exp_names:
    filename = exp_name + '.txt'
    data = np.loadtxt(filename)
    plt.plot(data,label=exp_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
