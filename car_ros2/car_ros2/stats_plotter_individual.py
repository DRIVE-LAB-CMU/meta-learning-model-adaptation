import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle

plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'

exp_name_ = 'random'
t = 10
n_ensembles = 3
suffix = '_en1'
if os.path.exists('data_sim/' + exp_name_+suffix) == False:
    os.makedirs('data_sim/' + exp_name_+suffix)

for i in range(1,t+1):
    plt.figure()
    exp_name = exp_name_ + str(i)
    filename = 'data_sim/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    print(data)
    # if n_ensembles == 0 :
    #     plt.plot(data['losses'], label=str(i))
    # else :
    #     for j in range(n_ensembles):
    #         plt.plot(data['losses'+str(j)], label=str(i) + ' ensemble ' + str(j))
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.title("Next state prediction error")
    plt.ylim(0, 7.5)
        
    plt.plot([data['online_transition']-data['buffer'],data['online_transition']-data['buffer']], [0, 7.5], label='online transition', color='red', linestyle='--')

    plt.legend()
    plt.savefig('data_sim/' + exp_name_ + suffix + '/loss'+str(i)+'.png')

for i in range(1,t+1):
    plt.figure()
    exp_name = exp_name_ + str(i)
    filename = 'data_sim/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    print(data)

    plt.plot(data['lat_errs'],label=str(i))
    plt.xlabel('Time (in s)')
    plt.ylabel('Lateral error (in m)')
    # plt.title("Lateral error (in m)")

    plt.ylim(0, 3.0)

    plt.plot([data['buffer'],data['buffer']], [0, 5.0], label='start training offline', color='red', linestyle='--')
    plt.plot([data['online_transition'],data['online_transition']], [0, 5.0], label='online transition', color='red', linestyle='--')
    plt.legend()
    plt.savefig('data_sim/' + exp_name_ + suffix + '/lat_errs'+str(i)+'.png', bbox_inches='tight')

for i in range(1,t+1):
    plt.figure()
    exp_name = exp_name_ + str(i)
    filename = 'data_sim/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    # print(data)
    times = np.arange(len(data['ws']))*0.05
    plt.plot(times, data['ws_gt'],label='GT w')
    plt.plot(times, data['ws_'],label='Predicted w after 5 steps')
    plt.plot(times, data['ws'],label='Predicted w with -steering after 20 steps')
    plt.xlabel('Time (in s)')
    plt.ylabel('w (in rad/s)')
    # plt.title("w")
    plt.plot([data['buffer']*0.05,data['buffer']*0.05], [np.min(data['ws']), np.max(data['ws_'])], label='start training offline', color='red', linestyle='--')
    plt.plot([data['online_transition']*0.05,data['online_transition']*0.05], [np.min(data['ws']), np.max(data['ws_'])], label='online transition', color='red', linestyle='--')
    plt.ylim(-2, 5)  # Set y-axis limits
    plt.legend()
    plt.savefig('data_sim/' + exp_name_ + suffix + '/ws'+str(i)+'.png', bbox_inches='tight')  # Save the figure

