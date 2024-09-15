import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle


# Change the size of legend text to large and the size of the title and also the font size of label and also values on x-axis y-axis scale to large for matplotlib
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'


suffix = '_en1'
exp_name_ = 'without_with_delay'
t = 10
n_ensembles = 1
if os.path.exists('data_sim/' + exp_name_ + suffix) == False:
    os.makedirs('data_sim/' + exp_name_+ suffix)

for i in range(1,t+1):
    if i!=3:
        exp_name = exp_name_ + str(i)
    else:
        exp_name = exp_name_ + str(2)
    filename = 'data_sim/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    # print(data)
    if n_ensembles == 0 :
        
        plt.plot(data['losses'], label=str(i))
    else :
        for j in range(n_ensembles):
            plt.plot(data['losses'], label=str(i) + ' ensemble ' + str(j))
    plt.xlabel('Iter')
    plt.ylabel('Error')
    plt.title("Next state prediction error")
plt.ylim(0, 7.5)
    
plt.plot([data['online_transition']-data['buffer'],data['online_transition']-data['buffer']], [0, 7.5], label='online transition', color='red', linestyle='--')

plt.legend()
plt.savefig('data_sim/' + exp_name_ + suffix + '/loss.png')

plt.figure()
avg_lat_err = 0.
avg_laps = 0.
# print(data.keys())
for i in range(1,t+1):
    if i!=3 and i!=-1:
        exp_name = exp_name_ + str(i)
    else:
        exp_name = exp_name_ + str(2)
    filename = 'data_sim/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    data['lat_errs'] = data['lat_errs'][:1000]
    # print(data)
    data['lat_errs'] = np.array(data['lat_errs'])
    # data['lat_errs'][:] *= 1.5
    # avg_laps += (data['traj'][1000,0,2]-data['traj'][300,0,2])/(2*math.pi)
    # print(data)
    avg_lat_err += np.mean(data['lat_errs'][300:])
    # avg_lat_err
    times = np.arange(len(data['lat_errs']))*0.05
    plt.plot(times,np.array(data['lat_errs']),color='red',alpha=0.5)
    plt.xlabel('Time (in s)')
    plt.ylabel('Lateral error (in m)')
    # plt.title("Lateral error (in m)")

crosses_x = [16.5,19.2,26.0,28.0,41.2,42.5]
crosses_y = [2.8,2.8,2.6,2.2,2.0,0.9]
# plt.scatter(crosses_x, crosses_y, s=300, c='purple', marker='x', clip_on=False, label='Trajectory did not converge')
    
avg_lat_err /= 10.
avg_laps /= 10.
print("Average lateral error: ", avg_lat_err)
print("Average no of laps: ", avg_laps)
plt.ylim(0, 3.0)

# plt.plot([data['buffer']*0.05,data['buffer']*0.05], [0, 5.0], label='start training offline', color='blue', linestyle='--')
# plt.plot([data['online_transition']*0.05,data['online_transition']*0.05], [0, 5.0], label='online transition', color='green', linestyle='--')
# plt.legend()
plt.savefig('data_sim/' + exp_name_ + suffix + '/lat_errs.png', bbox_inches='tight')

plt.figure()
for i in range(1,t+1):
    filename = 'data_sim/' + exp_name + suffix + '.pickle'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    # print(data)
    plt.plot(data['ws_gt'],label='GT w')
    plt.plot(data['ws_'],label='Predicted w after 5 steps')
    plt.plot(data['ws'],label='Predicted w with -steering after 20 steps')
    plt.xlabel('Iter')
    plt.ylabel('w (in rad/s)')
    plt.title("w")
    # plt.plot([data['buffer'],data['buffer']], [np.min(data['ws']), np.max(data['ws_'])], label='start training offline', color='red', linestyle='--')
    # plt.plot([data['online_transition'],data['online_transition']], [np.min(data['ws']), np.max(data['ws_'])], label='online transition', color='red', linestyle='--')
plt.legend()
plt.savefig('data_sim/' + exp_name_ + suffix + '/ws.png')

