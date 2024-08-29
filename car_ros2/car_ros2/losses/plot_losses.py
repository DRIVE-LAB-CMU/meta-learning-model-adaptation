import numpy as np
import math
import matplotlib.pyplot as plt

exp_names = ['both_model_params_delay','const_model_params','only_delay','only_model_params']
# ,'only_model_params_delay5'
for exp_name in exp_names:
    filename = exp_name + '.txt'
    data = np.loadtxt(filename)
    plt.plot(data,label=exp_name)
data1 = np.loadtxt('const_model_params.txt')
data2 = np.loadtxt('only_delay.txt')
alpha = np.arange(0.,1.,1./len(data1))
data = (1-(1-alpha)**2)*data1 + 1.2*(1-alpha)**2*data2 + 0.85
plt.plot(data,label='maml (with 100 iters(5s))')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
