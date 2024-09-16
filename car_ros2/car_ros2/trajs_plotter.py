import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['legend.fontsize'] = 50
plt.rcParams['axes.titlesize'] = 50
plt.rcParams['axes.labelsize'] = 50
plt.rcParams['xtick.labelsize'] = 50
plt.rcParams['ytick.labelsize'] = 50


filename = 'data/bias100.pickle'
with open(filename, 'rb') as file:
    data = pickle.load(file)

filename = 'data/circuit_bias.pickle'
with open(filename, 'rb') as file:
    data1 = pickle.load(file)

data['traj'] = np.array(data['traj'])
data['ref_traj'] = np.array(data['ref_traj'])
n_passes = 0

# model_used_after = data['online_transition']+100

# Create subplots ax_main, ax_zoom and ax_legend
fig, (ax_main, ax_zoom, ax_legend) = plt.subplots(1, 3, figsize=(30, 10))
ax_main.plot(data['ref_traj'][:,0], data['ref_traj'][:,1],'--',label='ref trajectory',color='black')
ax_main.plot(data['traj'][:,0,0], data['traj'][:,0,1], label='traj (APACRace)',color='red')
ax_main.plot(data1['traj'][:,0,0], data1['traj'][:,0,1],label='traj (Ours)',color='green')
ax_main.axis('equal')
ax_main.set_xlabel('x')
ax_main.set_ylabel('y')
# ax_main.legend()

ax_zoom.plot(data['ref_traj'][:,0], data['ref_traj'][:,1],'--',label='ref trajectory',color='black')
ax_zoom.plot(data['traj'][:,0,0], data['traj'][:,0,1], label='traj (APACRace)',color='red')
ax_zoom.plot(data1['traj'][:,0,0], data1['traj'][:,0,1],label='traj (Ours)',color='green')
ax_zoom.axis('equal')
ax_zoom.set_xlim(229,238)
ax_zoom.set_ylim(245,250)
ax_zoom.set_title('Zoomed-in View')
ax_zoom.set_xlabel('x')
# ax_zoom.set_ylabel('y')

rect = Rectangle((229, 245), 16, 12, linewidth=1, edgecolor='red', facecolor='gray',alpha=0.2)
ax_main.add_patch(rect)
plt.tight_layout()
# plt.axis('equal')
handles, labels = ax_main.get_legend_handles_labels()
ax_legend.legend(handles, labels, loc='center right')
ax_legend.axis('off')
plt.savefig(filename[:-7]+'.pdf', bbox_inches='tight')
plt.show()
