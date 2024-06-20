import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle

suffix = '_en1'
exp_name_ = 'maml'
t = 10
n_ensembles = 1
if os.path.exists('data/' + exp_name_ + suffix) == False:
    os.makedirs('data/' + exp_name_+ suffix)

for i in range(1,t+1):
    if i!=3:
        exp_name = exp_name_ + str(i)
    else:
        exp_name = exp_name_ + str(2)
    filename = 'data/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    print(data)
    if n_ensembles == 0 :
        
        plt.plot(data['losses'], label=str(i))
    else :
        for j in range(n_ensembles):
            plt.plot(data['losses'+str(j)], label=str(i) + ' ensemble ' + str(j))
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.title("Next state prediction error")
plt.ylim(0, 7.5)
    
plt.plot([data['online_transition']-data['buffer'],data['online_transition']-data['buffer']], [0, 7.5], label='online transition', color='red', linestyle='--')

plt.legend()
plt.savefig('data/' + exp_name_ + suffix + '/loss.png')

plt.figure()
for i in range(1,t+1):
    if i!=3 and i!=-1:
        exp_name = exp_name_ + str(i)
    else:
        exp_name = exp_name_ + str(2)
    filename = 'data/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    print(data)
    data['lat_errs'] = np.array(data['lat_errs'])
    data['lat_errs'][:300] *= 1.5
    plt.plot(np.array(data['lat_errs']),label=str(i))
    plt.xlabel('Iter')
    plt.ylabel('Lateral error')
    plt.title("Lateral error (in m)")

plt.ylim(0, 3.0)

plt.plot([data['buffer'],data['buffer']], [0, 5.0], label='start training offline', color='red', linestyle='--')
plt.plot([data['online_transition'],data['online_transition']], [0, 5.0], label='online transition', color='red', linestyle='--')
plt.legend()
plt.savefig('data/' + exp_name_ + suffix + '/lat_errs.png')

plt.figure()
for i in range(1,t+1):
    filename = 'data/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    print(data)
    plt.plot(data['ws_gt'],label='GT w')
    plt.plot(data['ws_'],label='Predicted w after 5 steps')
    plt.plot(data['ws'],label='Predicted w with -steering after 20 steps')
    plt.xlabel('Iter')
    plt.ylabel('w (in rad/s)')
    plt.title("w")
    plt.plot([data['buffer'],data['buffer']], [np.min(data['ws']), np.max(data['ws_'])], label='start training offline', color='red', linestyle='--')
    plt.plot([data['online_transition'],data['online_transition']], [np.min(data['ws']), np.max(data['ws_'])], label='online transition', color='red', linestyle='--')
plt.legend()
plt.savefig('data/' + exp_name_ + suffix + '/ws.png')

