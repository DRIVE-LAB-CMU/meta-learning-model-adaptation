import numpy as np
import pickle
import matplotlib.pyplot as plt

# filename = 'data/circuit_bias.pickle'
# with open(filename, 'rb') as file:
#     data1 = pickle.load(file)

# filename = '/home/dvij/lecar-car/ref_trajs/berlin_2018-large_with_speeds.csv'
# ref_traj = np.loadtxt(filename, delimiter=',')[:,1:] + np.array([[200., 200., 0.]])

filename = 'data/mars.pickle'
with open(filename, 'rb') as file:
    data = pickle.load(file)
data['ref_traj'] = np.array(data['ref_traj'])
# data['ref_traj'] = np.array(ref_traj)
data['traj'] = np.array(data['traj'])
print(data['ref_traj'].shape)
model_used_after = data['online_transition']+100
print(data['traj'].shape)
plt.plot(data['traj'][:model_used_after,0,0], data['traj'][:model_used_after,0,1], label='traj before adaptation')
plt.plot(data['traj'][model_used_after-1:,0,0], data['traj'][model_used_after-1:,0,1], label='traj after adaptation')
plt.plot(data['ref_traj'][:,0], data['ref_traj'][:,1], '--', label='ref')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.savefig(filename[:-7]+'.png')
plt.show()

# plt.plot(data['ref_traj'][:,0], data['ref_traj'][:,1], '--', label='ref')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
# plt.savefig('ref.png')
# plt.show()
