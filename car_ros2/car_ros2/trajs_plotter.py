import numpy as np
import pickle
import matplotlib.pyplot as plt


filename = 'data/kart100.pickle'
with open(filename, 'rb') as file:
    data = pickle.load(file)

filename = 'data/circuit_kart.pickle'
with open(filename, 'rb') as file:
    data1 = pickle.load(file)

data['traj'] = np.array(data['traj'])
data['ref_traj'] = np.array(data['ref_traj'])
n_passes = 0

# model_used_after = data['online_transition']+100

# Draw a filled circle at (-17,-17)

plt.plot(data['traj'][:,0,0], data['traj'][:,0,1], label='traj (APACRace)',color='blue')
plt.plot(data['ref_traj'][:,0], data['ref_traj'][:,1],'--',label='ref trajectory',color='black')
plt.plot(data1['traj'][:,0,0], data1['traj'][:,0,1],label='traj (Ours)',color='green')
plt.axis('equal')
plt.legend()

plt.xlabel('x')
plt.ylabel('y')
# plt.axis('equal')
plt.savefig(filename[:-4]+'.png')
plt.show()
