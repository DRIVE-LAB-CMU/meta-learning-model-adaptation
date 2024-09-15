import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# filename = 'data/circuit_bias.pickle'
# with open(filename, 'rb') as file:
#     data1 = pickle.load(file)

# filename = '/home/dvij/lecar-car/ref_trajs/berlin_2018-large_with_speeds.csv'
# ref_traj = np.loadtxt(filename, delimiter=',')[:,1:] + np.array([[200., 200., 0.]])

# filename = 'safety_results/stats__07_16_2024_22_39_10.pkl'
# filename = 'safety_results/stats__07_16_2024_22_51_44.pkl'
# filename = 'safety_results/stats__07_16_2024_20_31_28.pkl'
filenames = [ \
    'safety_results/stats__07_16_2024_20_31_28.pkl',\
    'safety_results/stats__07_16_2024_22_51_44.pkl',\
    'safety_results/stats__07_16_2024_22_39_10.pkl',\
    'safety_results/stats__07_16_2024_20_05_03.pkl']
names = ['Without CBF', 'With CBF + DOB', 'With CBF + residual learning', 'With CBF + DOB + residual learning']
colors = ['brown', 'blue', 'yellow', 'green']
# filename = 'data/smooth_tires_maml1.pickle'
for k in range(4) :
    thres_dist = 0.3
    filename = filenames[k]
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    data['traj'] = np.array(data['traj'])
    n_passes = 0
    # print("Avg lateral error: ",np.mean(data['lat_errs'][400:1500]))
    for i in range(len(data['traj'])-1):
        if data['traj'][i+1,0,0]*data['traj'][i,0,0] < 0.:
            n_passes += 1
    print(n_passes)
    curr_status = 1
    inds = []
    rewards = [0.]
    n_viols = []
    for i in range(data['traj'].shape[0]):
        # plt.plot(data['traj'][i,0,0], data['traj'][i,0,1], 'o', markersize=2, markerfacecolor='blue')
        x,y = data['traj'][i,0,0], data['traj'][i,0,1]
        x2,y2 = data['traj'][i-1,0,0], data['traj'][i-1,0,1]
        if i!=0.:
            if curr_status == 2:
                dist1 = np.sqrt((x+1.7)**2 + (y+1.7)**2)
                dist2 = np.sqrt((x2+1.7)**2 + (y2+1.7)**2)
            if curr_status == 1:
                dist1 = np.sqrt((x-1.7)**2 + (y-1.7)**2)
                dist2 = np.sqrt((x2-1.7)**2 + (y2-1.7)**2)
            reward = (dist2-dist1)
            rewards.append(10*reward)

        if np.sqrt(x**2 + y**2) < 0.8:
            n_viols.append(1)
        else :
            n_viols.append(0)
        dist1 = np.sqrt((x+1.7)**2 + (y+1.7)**2)
        dist2 = np.sqrt((x-1.7)**2 + (y-1.7)**2)
        if dist1 < 0.3 and curr_status == 2:
            curr_status = 1
            inds.append(i)
        if dist2 < 0.3 and curr_status == 1:
            curr_status = 2
            inds.append(i)

    EP_LEN = 200
    rewards_ep = []
    n_viols_ep = []
    factor = 1
    if k == 2:
        factor = 1.
    for i in range(0,len(rewards),EP_LEN):
        rewards_ep.append(factor*np.sum(rewards[i:i+EP_LEN])/EP_LEN)
        n_viols_ep.append(np.sum(n_viols[i:i+EP_LEN])/EP_LEN)

    factor = 1
    if k == 0:
        factor = 2
    X = np.arange(len(rewards_ep))*EP_LEN*factor
    plt.plot(X[:-1], rewards_ep[:-1], label=names[k], color=colors[k])
plt.xlabel('n_steps')
plt.ylabel('rewards')
plt.ylim(-0.1,0.8)
plt.title('n_steps vs rewards')
plt.legend(loc='upper left')
plt.savefig('rewards.png')
plt.show()


for k in range(4) :
    thres_dist = 0.3
    filename = filenames[k]
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    data['traj'] = np.array(data['traj'])
    n_passes = 0
    # print("Avg lateral error: ",np.mean(data['lat_errs'][400:1500]))
    for i in range(len(data['traj'])-1):
        if data['traj'][i+1,0,0]*data['traj'][i,0,0] < 0.:
            n_passes += 1
    print(n_passes)
    curr_status = 1
    inds = []
    rewards = [0.]
    n_viols = []
    avg_viols = []
    avg_viol = 0
    for i in range(data['traj'].shape[0]):
        # plt.plot(data['traj'][i,0,0], data['traj'][i,0,1], 'o', markersize=2, markerfacecolor='blue')
        x,y = data['traj'][i,0,0], data['traj'][i,0,1]
        x2,y2 = data['traj'][i-1,0,0], data['traj'][i-1,0,1]
        if i!=0.:
            if curr_status == 2:
                dist1 = np.sqrt((x+1.7)**2 + (y+1.7)**2)
                dist2 = np.sqrt((x2+1.7)**2 + (y2+1.7)**2)
            if curr_status == 1:
                dist1 = np.sqrt((x-1.7)**2 + (y-1.7)**2)
                dist2 = np.sqrt((x2-1.7)**2 + (y2-1.7)**2)
            reward = (dist2-dist1)
            rewards.append(10*reward)

        if np.sqrt(x**2 + y**2) < 0.8:
            n_viols.append(1)
            avg_viol += 1
        else :
            n_viols.append(0)
        dist1 = np.sqrt((x+1.7)**2 + (y+1.7)**2)
        dist2 = np.sqrt((x-1.7)**2 + (y-1.7)**2)
        if dist1 < 0.3 and curr_status == 2:
            curr_status = 1
            avg_viols.append(avg_viol)
            avg_viol = 0
            inds.append(i)
        if dist2 < 0.3 and curr_status == 1:
            curr_status = 2
            inds.append(i)
            avg_viols.append(avg_viol)
            avg_viol = 0
    print("avg viols:",avg_viols, np.mean(avg_viols[:-1]))  
    EP_LEN = 200
    rewards_ep = []
    n_viols_ep = []
    for i in range(0,len(rewards),EP_LEN):
        rewards_ep.append(np.sum(rewards[i:i+EP_LEN])/EP_LEN)
        n_viols_ep.append(np.sum(n_viols[i:i+EP_LEN])/EP_LEN)

    factor = 1
    if k == 0:
        factor = 2
    X = np.arange(len(rewards_ep))*EP_LEN*factor
    plt.plot(X[:-1], n_viols_ep[:-1], label=names[k], color=colors[k])
plt.xlabel('n_steps')
plt.ylabel('costs')
plt.ylim(-0.1,1.2)
plt.legend(loc='upper left')
plt.title('n_steps vs costs')
plt.savefig('costs.png')
plt.show()

