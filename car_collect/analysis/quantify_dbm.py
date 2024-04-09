import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interact
from car_dynamics.envs import DynamicParams, DynamicBicycleModel
from car_dynamics.modules.utils import denormalize_angle_sequence
from rich.progress import track
matplotlib.use('MacOSX')
# use GUI backend for plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# df = pd.read_csv('/Users/wenlixiao/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/data/dataset/vicon-data-clean/vicon-circle-data-20240302-151638-1.csv')
# df = pd.read_csv('/Users/wenlixiao/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/data/dataset/vicon-data-clean/vicon-circle-data-20240302-143549-3.csv')
df = pd.read_csv('/Users/wenlixiao/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/data/dataset/vicon-data-clean/vicon-circle-data-20240307-010142.csv')


H = 50

plot_type = 'traj'

model_params = DynamicParams(num_envs=1,Sa=0.36, Sb=0.03,K_FFY=20, K_RFY=20, Ta=7., Tb=-1.)
dynamics = DynamicBicycleModel(model_params, device=DEVICE)

err = []
dt_list = []
# for i in track(range(1, df.shape[0]-1-H)):
# N = 1200
N = 100
for i in range(N, N+1):
    ptr = i
    
    dt = df['time'][i+1] - df['time'][i]
    x = df['pos_x'][i]
    y = df['pos_y'][i]
    v_vec = np.array([df['pos_x'][i+1] - df['pos_x'][i], df['pos_y'][i+1] - df['pos_y'][i]]) / dt
    
    vx = v_vec[0] * np.cos(df['yaw'][i]) + v_vec[1] * np.sin(df['yaw'][i])
    vy = v_vec[1] * np.cos(df['yaw'][i]) - v_vec[0] * np.sin(df['yaw'][i])
    psi = df['yaw'][i]
    omega = (df['yaw'][i+1] - df['yaw'][i]) 
    omega = np.arctan2(np.sin(omega), np.cos(omega)) / dt
    omega = 2.
    
    print(vx, vy, omega)
    err.append([])
    pred_traj = []
    real_traj = []
    action_list = []
    
    pred_traj.append([x, y, psi, vx, vy, omega])
    real_traj.append([df['pos_x'][i], df['pos_y'][i], df['yaw'][i], vx, vy, omega])
    for h in range(H):
        ## calc next
        dt = df['time'][ptr + 1] - df['time'][ptr]
        v_vec_ip1 = np.array([df['pos_x'][ptr+1] - df['pos_x'][ptr], df['pos_y'][ptr+1] - df['pos_y'][ptr]]) / dt
        vx_ip1 = v_vec_ip1[0] * np.cos(df['yaw'][ptr+1]) + v_vec_ip1[1] * np.sin(df['yaw'][ptr+1])
        vy_ip1 = v_vec_ip1[1] * np.cos(df['yaw'][ptr+1]) - v_vec_ip1[0] * np.sin(df['yaw'][ptr+1])
        omega_ip1 = (df['yaw'][ptr+1] - df['yaw'][ptr])
        omega_ip1 = np.arctan2(np.sin(omega_ip1), np.cos(omega_ip1)) / dt
    
        x, y, psi, vx, vy, omega = dynamics.single_step_numpy(
            np.array([ x, y, psi, vx, vy, omega ]),
            np.array([ df['target_vel'][ptr], df['target_steer'][ptr] ])
        )
        # psi = np.arctan2(np.sin(psi), np.cos(psi))
        pred_traj.append([x, y, psi, vx, vy, omega])
        real_traj.append([df['pos_x'][ptr+1], df['pos_y'][ptr+1], df['yaw'][ptr+1], vx_ip1, vy_ip1, omega_ip1])
        action_list.append([ df['target_vel'][ptr], df['target_steer'][ptr] ])
        err[-1].append(
            np.abs(np.array([
                x - df['pos_x'][ptr+1],
                y - df['pos_y'][ptr+1],
                psi - df['yaw'][ptr+1],
                vx - vx_ip1,
                vy - vy_ip1,
                omega - omega_ip1,
            ]))
        )
        ptr += 1
    pred_traj = np.array(pred_traj)
    real_traj = np.array(real_traj)
    action_list = np.array(action_list)
    if plot_type == 'traj':
        fig, axs = plt.subplots(2, 3, figsize=(10,5))
        axs[0, 0].scatter(pred_traj[:, 0], pred_traj[:, 1], label='pred',s=1)
        axs[0, 0].scatter(real_traj[:, 0], real_traj[:, 1], label='real',s=1)
        axs[0, 0].axis('equal')
        axs[0, 0].set_title('Trajectory')
        axs[0, 0].legend()
        axs[0, 1].plot(action_list[:, 0], label='vel command')
        axs[0, 1].plot(action_list[:, 1], label='steer command')
        axs[0, 1].set_title('Commands')
        axs[0, 1].legend()
        axs[0, 2].plot(pred_traj[:, 2], label='pred')
        # import pdb; pdb.set_trace()
        axs[0, 2].plot(denormalize_angle_sequence(real_traj[:, 2]), label='real')
        axs[0, 2].legend()
        axs[0, 2].set_title('psi')
        axs[1, 0].plot(pred_traj[:, 3], label='pred')
        axs[1, 0].plot(real_traj[:, 3], label='real')
        axs[1, 0].set_title('vx')
        axs[1, 0].legend()
        axs[1, 1].plot(pred_traj[:, 4], label='pred')
        axs[1, 1].plot(real_traj[:, 4], label='real')
        axs[1, 1].set_title('vy')  
        axs[1, 1].legend()
        axs[1, 2].plot(pred_traj[:, 5], label='pred')
        axs[1, 2].plot(real_traj[:, 5], label='real')
        axs[1, 2].set_title('omega')
        axs[1, 2].legend()
        plt.show()
    
if plot_type == 'err':
    err = np.array(err)
    mean_err = np.mean(err, axis=1)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs[0, 0].plot(mean_err[:, 0])
    axs[0, 0].set_title('x')
    axs[0, 1].plot(mean_err[:, 1])
    axs[0, 1].set_title('y')
    axs[0, 2].plot(mean_err[:, 2])
    axs[0, 2].set_title('psi')
    axs[0, 3].plot(mean_err[:, 3])
    axs[0, 3].set_title('vx')
    axs[1, 0].plot(mean_err[:, 4])
    axs[1, 0].set_title('vy')
    axs[1, 1].plot(mean_err[:, 5])
    axs[1, 1].set_title('omega')
    axs[1, 2].plot(df['target_steer'])
    axs[1, 2].set_title('target_steer')
    axs[1, 3].plot(df['target_vel'])
    axs[1, 3].set_title('target_vel')
    plt.show()