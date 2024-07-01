import numpy as np
import math

SPEED = 15.
f = np.loadtxt('traj_rr.csv',delimiter=',')
f_new = f[:,[3,0,1,2]]
f_new[:,-1] = SPEED
np.savetxt('../ref_trajs/traj_rr_with_speeds.csv',f_new,delimiter=',')