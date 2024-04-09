import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import os
import matplotlib.cm as cm
import pandas as pd
from utils import diff_angle
import argparse

parser = argparse.ArgumentParser(description='Arguments for offline slip angle vs Force plots')

# files = ['traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-10.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-15.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-20.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-25.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-30.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-35.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-40.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-45.csv',
#          'traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-50.csv']

files = ['traj_files/vicon-data-clean/vicon-circle-data-20240306-001713.csv',
        #  'traj_files/vicon-data-clean/vicon-circle-data-20240306-002357.csv',
        #  'traj_files/vicon-data-clean/vicon-circle-data-20240306-002727.csv',
         ]

# Add command-line arguments
# parser.add_argument('--file', dest='file_path', required=False, default='traj_files/vicon-data-clean/vicon-circle-data-20240225-215833.csv',help='Specify the file path')
# parser.add_argument('--file', dest='file_path', required=False, default='traj_files/vicon-data-clean/vicon-circle-data-20240302-143549-1.csv',help='Specify the file path')
# parser.add_argument('--file', dest='file_path', required=False, default='traj_files/vicon-data-clean/vicon-circle-data-20240305-002436.csv',help='Specify the file path')
# parser.add_argument('--file', dest='file_path', required=False, default='traj_files/vicon-data-clean/vicon-circle-data-20240305-010714.csv',help='Specify the file path')
# parser.add_argument('--file', dest='file_path', required=False, default='traj_files/vicon-data-clean/vicon-circle-data-20240305-011807.csv',help='Specify the file path')
parser.add_argument('--file', dest='file_path', required=False, default='none',help='Specify the file path')
# parser.add_argument('--file', dest='file_path', required=False, default='traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-30.csv',help='Specify the file path')
# parser.add_argument('--file', dest='file_path', required=False, default='traj_files/vicon-data-clean/vicon-circle-data-20240305-223404-45.csv',help='Specify the file path')
parser.add_argument('--dt', required=False, default=0.05,help='Specify the time step')
parser.add_argument('--lf', required=False, default=0.14,help='Specify the distance from COM to front axle')
parser.add_argument('--lr', required=False, default=0.22,help='Specify the distance from COM to rear axle')
parser.add_argument('--theta_bias', required=False, default=-0.08,help='Specify the theta bias')
parser.add_argument('--steer_bias', required=False, default=-0.03,help='Specify the steer bias')
parser.add_argument('--steer_factor', required=False, default=0.36,help='Specify the steer factor')
parser.add_argument('--filter_v', required=False, default=4,help='Exp filter for velocities')
parser.add_argument('--filter_theta', required=False, default=12,help='Exp filter for yaw')
parser.add_argument('--filter_a', required=False, default=3,help='Exp filter for accelerations')
parser.add_argument('--filter_wdot', required=False, default=3,help='Exp filter for wdot')
parser.add_argument('--filter_steer', required=False, default=1,help='Exp filter for steer')
parser.add_argument('--save', required=False, action='store_true',help='Whether to save')
parser.add_argument('--min_v', required=False, default=0.5,help='Min velocity for masking out')
parser.add_argument('--max_v', required=False, default=3.5,help='Max velocity for masking out')
parser.add_argument('--mask_padding', required=False, default=6,help='Mask padding')
parser.add_argument('--max_w_cmd', required=False, default=9.,help='Max w/cmd ratio')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
files.append(args.file_path)

lf = args.lf
lr = args.lr
dt = args.dt
mask_padding = args.mask_padding
max_w_cmd = args.max_w_cmd
theta_bias = args.theta_bias
steer_bias = args.steer_bias
steer_factor = args.steer_factor

filter_v = args.filter_v
filter_theta = args.filter_theta
filter_a = args.filter_a
filter_wdot = args.filter_wdot
filter_steer = args.filter_steer


PLOT_W_STEER = False
PLOT_VY_STEER = True
PLOT_SLIP_ANGLES = True
PLOT_ANGLE_DIFFS = True
PLOT_ACCS = True
PLOT_GG = True
RUN_NAME = files[0].split('/')[-1].split('.')[0]

path = []
mask_partition = []
for file_path in files:
    if file_path == 'none':
        continue
    path_file = pd.read_csv(file_path)
    mask_partition.append(1)
    path.append([path_file['pos_x'][0],path_file['pos_y'][0],path_file['yaw'][0],path_file['target_steer'][0],path_file['target_vel'][0]])
    for i in range(1,len(path_file)) :
        mask_partition.append(0)
        path.append([path_file['pos_x'][i],path_file['pos_y'][i],path_file['yaw'][i],path_file['target_steer'][i],path_file['target_vel'][i]])
path = np.array(path)
mask_partition = np.array(mask_partition)

batch_size = 500
learning_rate = 0.025
n_iters = 15

xyts = np.array(path)
xyts[:,2] += math.pi
xyts[:,2] -= theta_bias
xyts[:,2] = np.arctan2(np.sin(xyts[:,2]),np.cos(xyts[:,2]))
steers = xyts[:,-2]*steer_factor + steer_bias
throttles = xyts[:,-1]
steer = steer_bias
for i in range(len(steers)) :
    steer += (steers[i]-steer)/filter_steer
    steers[i] = steer

midts = (xyts[:-1,2] + xyts[:-1,2])/2.
vel_yaws = np.arctan2(xyts[1:,1]-xyts[:-1,1],xyts[1:,0]-xyts[:-1,0])
xy_dots_ = (xyts[1:,:2] - xyts[:-1,:2])/dt
xy_dots = np.array([xy_dots_[:,0]*np.cos(midts) + xy_dots_[:,1]*np.sin(midts),-xy_dots_[:,0]*np.sin(midts) + xy_dots_[:,1]*np.cos(midts)]).T
t_dots = (xyts[1:,2:] - xyts[:-1,2:])
t_dots = (t_dots>math.pi)*(t_dots-2*math.pi) + (t_dots<=-math.pi)*(t_dots+2*math.pi) \
    + (t_dots<=math.pi)*(t_dots>-math.pi)*t_dots
t_dots *= 1./dt
speeds = np.linalg.norm(xy_dots,axis=1)
mask1 = (speeds>args.min_v)*(speeds<args.max_v)
mask2 = (np.abs(throttles[1:])>0.15)
mask3 = (1 - ((t_dots[:,0]/np.abs(throttles[1:]))>-max_w_cmd)*((t_dots[:,0]/np.abs(throttles[1:]))<max_w_cmd)*((t_dots[:,0]*steers[1:])>0.))
mask = (1-mask1)*mask2 + mask3 + mask_partition[1:]
for i in range(mask_padding) :
    mask[1:] += mask[:-1]
    mask[:-1] += mask[1:]
mask = mask > 0
xy_dots[:,1] += t_dots[:,0]*lr
vx = 0.
vy = 0.
w = 0.
times = dt*np.arange(len(xyts))
for i in range(1,len(xy_dots)) :
    if mask[i] == 0 :
        vx = vx + (xy_dots[i,0]-vx)/filter_v
        vy = vy + (xy_dots[i,1]-vy)/filter_v
        w = w + (t_dots[i,0]-w)/filter_theta
    # if mask_partition[i] == 1 :
    #     vx = xy_dots[i,0]
    #     vy = xy_dots[i,1]
    #     w = t_dots[i,0]
    xy_dots[i,0] = vx
    xy_dots[i,1] = vy
    t_dots[i,0] = w


if PLOT_ANGLE_DIFFS:
    ang_diffs = xyts[:-1,2] - vel_yaws
    ang_diffs = (ang_diffs>math.pi)*(ang_diffs-2*math.pi) + (ang_diffs<-math.pi)*(ang_diffs+2*math.pi) +\
        (ang_diffs>-math.pi)*(ang_diffs<math.pi)*ang_diffs
    plt.plot(times[:-1],ang_diffs)
    # plt.plot(times[:-1],mask)
    plt.show()
    
if PLOT_W_STEER :
    plt.plot(times[:-1],t_dots[:,0],label='w actual')
    plt.plot(times[:-1],xy_dots[:,0]*np.tan(steers[:-1])/(lf+lr),label='w expected')
    plt.plot(times[:-1],mask)
    plt.legend()
    plt.show()

if PLOT_VY_STEER :
    plt.plot(times[:-1],xy_dots[:,0],label='vx')
    plt.plot(times[:-1],xy_dots[:,1]/np.maximum(xy_dots[:,0],0.5),label='vy/vx')
    plt.plot(times[:-1],t_dots[:,0],label='w')
    plt.plot(times[:-1],steers[:-1],label='steers')
    plt.plot(times[:-1],steer_factor*t_dots[:,0]/np.maximum(np.abs(xy_dots[:,0]),0.5),label='steers pred')
    plt.plot(times[:-1],mask)
    plt.legend()
    plt.show()

vx,vy,w = xy_dots[:,0], xy_dots[:,1], t_dots[:,0]
alpha_f = steers[:-1]-np.arctan2(vy+w*lf,np.maximum(0.5,vx))
alpha_r = np.arctan2(-vy+w*lr,np.maximum(0.5,vx))
if PLOT_SLIP_ANGLES :
    plt.plot(times[:-1],alpha_f,label='alpha_f')
    plt.plot(times[:-1],alpha_r,label='alpha_r')
    plt.plot(times[:-1],steers[:-1],label='steers')
    plt.legend()
    plt.show()

xy_dot_dots = (xy_dots[1:,:2] - xy_dots[:-1,:2])/dt
t_dot_dots = (t_dots[1:,:] - t_dots[:-1,:])/dt
ax = 0.
ay = 0.
wdot = 0.
for i in range(1,len(xy_dot_dots)) :
    ax = ax + (xy_dot_dots[i,0]-ax)/filter_a
    ay = ay + (xy_dot_dots[i,1]-ay)/filter_a
    wdot = wdot + (t_dot_dots[i,0]-wdot)/filter_wdot
    if mask_partition[i] == 1 :
        ax = xy_dot_dots[i,0]
        ay = xy_dot_dots[i,1]
        wdot = t_dot_dots[i,0]
    xy_dot_dots[i,0] = ax
    xy_dot_dots[i,1] = ay
    t_dot_dots[i,0] = wdot

if PLOT_ACCS :
    plt.plot(times[1:-1],xy_dot_dots[:,0],label='ax')
    plt.plot(times[1:-1],xy_dot_dots[:,1],label='ay')
    plt.plot(times[:-1],xy_dots[:,0]*t_dots[:,0],label='w_acc')
    plt.legend()
    plt.show()

def check_range(alpha, F, K_min = 3., K_max = 40.) :
    if alpha > 0. :
        if F > -alpha*K_min+5. :
            return False
        elif F < -alpha*K_max-5. :
            return False
        return True
    else :
        if F < -alpha*K_min-5. :
            return False
        elif F > -alpha*K_max + 5 :
            return False
        return True
    
Ffys = []
Frys = []
Frxs = []
alpha_fs = []
alpha_rs = []
curr_size = 0
mu_preds = []
for i in tqdm.tqdm(range(len(xy_dot_dots))) :
    delta = steers[i+1]
    A = np.array([[1,-math.sin(delta),0.,xy_dots[i,1]*t_dots[i,0]],\
                 [0.,math.cos(delta),1.,-xy_dots[i,0]*t_dots[i,0]],\
                 [0.,lf*math.cos(delta)*2./(0.15*0.15),-lr*2./(0.15*0.15),0.],\
                 [0.,0.,0.,1.]])
    B = np.array([[xy_dot_dots[i,0],xy_dot_dots[i,1],t_dot_dots[i,0],1.]]).T
    X = np.matmul(np.linalg.inv(A),B)
    # if abs(steers[i])>0.2 and mask[i]==0 and xy_dots[i,0]>1.5 and alpha_f[i]<0.4 and alpha_r[i]<0.4 and alpha_f[i]>-0.4 and alpha_r[i]>-0.4 and check_range(alpha_f[i],X[1,0]) and check_range(alpha_r[i],X[2,0]):
    if mask[i]==0 and abs(steers[i])>0.2 and xy_dots[i,0]>1.:
        Frxs.append(X[0,0])
        Ffys.append(X[1,0])
        Frys.append(X[2,0])
        alpha_fs.append(alpha_f[i])
        alpha_rs.append(alpha_r[i])
        curr_size += 1
        print(curr_size)

    if curr_size < batch_size + 2: 
        continue

if PLOT_GG :
    plt.scatter(np.array(alpha_fs),Ffys,s=1,label='front tire calculated Fy/m')
    plt.scatter(np.array(alpha_rs),Frys,s=1,label='rear tire calculated Fy/m')
    plt.scatter([0.],[0.],s=8,label='origin')
    plt.legend()
    plt.show()

states = np.array([xyts[1:-1,0],xyts[1:-1,1],xyts[1:-1,2],xy_dots[1:,0],xy_dots[1:,1],t_dots[1:,0],(1-mask[1:])*(abs(steers[1:-1])>0.2)*(xy_dots[1:,0]>1.)]).T
inputs = np.array([throttles[1:-2],steers[1:-2]]).T

if args.save:
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    np.savez('processed_data/'+RUN_NAME+'_data.npz',states=states,inputs=inputs,Ffys=Ffys,Frys=Frys,Frxs=Frxs,alpha_fs=alpha_fs,alpha_rs=alpha_rs)
