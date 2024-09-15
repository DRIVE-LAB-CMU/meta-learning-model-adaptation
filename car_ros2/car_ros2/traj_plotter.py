import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# filename = 'data/circuit_bias.pickle'
# with open(filename, 'rb') as file:
#     data1 = pickle.load(file)

# filename = '/home/dvij/lecar-car/ref_trajs/berlin_2018-large_with_speeds.csv'
# ref_traj = np.loadtxt(filename, delimiter=',')[:,1:] + np.array([[200., 200., 0.]])

filename = 'safety_results/stats__07_16_2024_22_39_10.pkl'
# filename = 'safety_results/stats__07_16_2024_22_51_44.pkl'
# filename = 'safety_results/stats__07_16_2024_20_31_28.pkl'
# filename = 'safety_results/stats__07_16_2024_20_05_03.pkl'
# filename = 'data/smooth_tires_maml1.pickle'
thres_dist = 0.3
with open(filename, 'rb') as file:
    data = pickle.load(file)
# data['ref_traj'] = np.array(data['ref_traj'])
# data['ref_traj'] = np.array(ref_traj)
data['traj'] = np.array(data['traj'])
n_passes = 0
# print("Avg lateral error: ",np.mean(data['lat_errs'][400:1500]))
for i in range(300,1500):
    if data['traj'][i+1,0,0]*data['traj'][i,0,0] < 0.:
        n_passes += 1
print(n_passes)
# print(data['ref_traj'].shape)
model_used_after = data['online_transition']+100
# print(data['traj'].shape)

# Draw a filled circle at (-17,-17)
# plt.plot(-1.7,-1.7, 'o', markersize=10, markerfacecolor='green')
# plt.plot(1.7,1.7, 'o', markersize=10, markerfacecolor='green')

# Plot trajectory with a timescale based coloring scheme
segments = np.concatenate([data['traj'][:-1,:,:2], data['traj'][1:,:,:2]], axis=1)
print(segments.shape)
# Create a LineCollection from the segments and set the color according to the velocity
cmap = plt.get_cmap('viridis')
velocity = np.arange(data['traj'].shape[0]-1)*0.06
norm = plt.Normalize(velocity.min(), velocity.max())
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(velocity)
lc.set_linewidth(2)

fig, ax = plt.subplots()
ax.add_collection(lc)
curr_status = 1
inds = []
rewards = [0.]
for i in range(data['traj'].shape[0]):
    # plt.plot(data['traj'][i,0,0], data['traj'][i,0,1], 'o', markersize=2, markerfacecolor='blue')
    if i!=0.:
        x,y = data['traj'][i,0,0], data['traj'][i,0,1]
        x2,y2 = data['traj'][i-1,0,0], data['traj'][i-1,0,1]
        if curr_status == 2:
            dist1 = np.sqrt((x+1.7)**2 + (y+1.7)**2)
            dist2 = np.sqrt((x2+1.7)**2 + (y2+1.7)**2)
        if curr_status == 1:
            dist1 = np.sqrt((x-1.7)**2 + (y-1.7)**2)
            dist2 = np.sqrt((x2-1.7)**2 + (y2-1.7)**2)
        reward = (dist2-dist1)
        rewards.append(reward)

        
    x,y = data['traj'][i,0,0], data['traj'][i,0,1]
    dist1 = np.sqrt((x+1.7)**2 + (y+1.7)**2)
    dist2 = np.sqrt((x-1.7)**2 + (y-1.7)**2)
    if dist1 < 0.3 and curr_status == 2:
        curr_status = 1
        inds.append(i)
    if dist2 < 0.3 and curr_status == 1:
        curr_status = 2
        inds.append(i)

EP_LEN = 50
rewards_ep = []
for i in range(0,len(rewards),EP_LEN):
    rewards_ep.append(np.sum(rewards[i:i+EP_LEN])/EP_LEN)

# a = inds[-6]
# b = inds[-4]
# plt.plot(data['traj'][a:b,0,0], data['traj'][a:b,0,1], '--', markersize=2, markerfacecolor='red',color='red')    
# plt.plot(data['traj'][:,0,0], data['traj'][:,0,1], label='traj',color='blue')
# plt.plot(data['traj'][model_used_after-1:,0,0], data['traj'][model_used_after-1:,0,1], label='traj after adaptation')
# plt.plot(data['ref_traj'][:,0], data['ref_traj'][:,1], '--', label='ref')
plt.axis('equal')
plt.xlim(-2.3, 2.3)
plt.ylim(-2.3, 2.3)
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label('Time (in s)')
plt.legend()
# Draw a big red circle of radius 0.8 at center (0,0)
circle = plt.Circle((0, 0), 0.8, color='red', fill=True)
plt.gca().add_artist(circle)
circle = plt.Circle((-1.7, -1.7), 0.3, color='green', fill=True)
plt.gca().add_artist(circle)
circle = plt.Circle((1.7, 1.7), 0.3, color='green', fill=True)
plt.gca().add_artist(circle)

plt.xlabel('x')
plt.ylabel('y')
# plt.axis('equal')
plt.savefig(filename[:-7]+'.png')
plt.show()

# plt.plot(data['ref_traj'][:,0], data['ref_traj'][:,1], '--', label='ref')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
# plt.savefig('ref.png')
# plt.show()
