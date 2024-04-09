import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from offroad.utils import load_state
import json
import matplotlib
from ipywidgets import interact
from offroad import OFFROAD_DATA_DIR
from car_dynamics.models_torch import MLP
from car_dynamics.envs import make_env, KinematicBicycleModel, KinematicParams, DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_torch import MPPIController, rollout_fn_select, reward_track_fn
import matplotlib.colors as colors
from car_dynamics.models_torch import MLP, parse_data_end2end_norm 

matplotlib.use('MacOSX')

SPEED = 1.0
LF = .16
LR = .15
L = LF+LR
N_ROLLOUTS = 10000
H = 10
DT = 0.05
SIGMA = 1.0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# log_dir = os.path.join(PROJ_DIR, 'data', 'data-20240131-195144') # drag
log_dir = os.path.join(OFFROAD_DATA_DIR, 'dataset', 'data-20240302-161221') # counter 
# log_dir = os.path.join(PROJ_DIR, 'data', 'data-20240205-000620') # circle
# log_dir = os.path.join(PROJ_DIR, 'data', 'sim-data-20240131-134617')
with open(os.path.join(log_dir, 'header.json')) as f:
    header_info = json.load(f)
# t_list, p_dict, yaw_dict, action_list, controller_info = load_state(log_dir, [0, 144], orientation_provider="ORIENTATION_PROVIDOER")
t_list, p_dict, yaw_dict, action_list, controller_info = load_state(log_dir, [10, 78], orientation_provider="ORIENTATION_PROVIDOER")
# t_list, p_dict, yaw_dict, action_list, controller_info = load_state(log_dir, [0, 303], orientation_provider="ORIENTATION_PROVIDOER")
# t_list, p_dict, yaw_dict, action_list, controller_info = load_state(log_dir, [0, 1], orientation_provider="ORIENTATION_PROVIDOER")
obs_np = p_dict['obs']
obs_np_1 = obs_np + .0
targets = controller_info['targets']
# targets = np.array([target[0] for target in controller_info['targets']])
is_recover = controller_info['is_recover']

len_history = 3
model_params = DynamicParams(
                    num_envs=N_ROLLOUTS, DT=DT,
)   


dynamics = DynamicBicycleModel(model_params, device=DEVICE)
rollout_fn = rollout_fn_select('dbm', dynamics, DT, model_params.LF+model_params.LR, model_params.LR)

def rollout_start_nn():
    ...

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
# a_cov_prev =  torch.full((H, 2, 2), 3.0**2) * torch.eye(2).unsqueeze(0).repeat(H, 1, 1)

# import pdb; pdb.set_trace()

mppi = MPPIController(
    gamma_mean=1.0,
    gamma_sigma=0.0,
    discount=1.,
    sample_sigma = 0.5,
    lam = 0.01,
    a_mean=torch.zeros(H, 2, device=DEVICE),
    a_cov = a_cov_init.to(DEVICE),
    n_rollouts=N_ROLLOUTS,
    H=H,
    device=DEVICE,
    rollout_fn=rollout_fn,
    a_min = [-1., -1],
    a_max = [1., 1.],
    a_mag = [0.1, 1.],
    a_shift= [0.3, 0.],
    delay=0,
    len_history=len_history,
    rollout_start_fn=rollout_start_nn,
    debug=False,
    fix_history=False,
    num_actions=2,
    num_obs=6,
)

N=386
target_N = np.array(targets[N])

obs_tensor = torch.tensor(obs_np, device=DEVICE)

v_vec = ( obs_tensor[1:, :2] - obs_tensor[:-1, :2] ) / DT
vx = v_vec[:, 0] * torch.cos(obs_tensor[1:, 2]) + v_vec[:, 1] * torch.sin(obs_tensor[1:, 2])
vy = v_vec[:, 1] * torch.cos(obs_tensor[1:, 2]) - v_vec[:, 0] * torch.sin(obs_tensor[1:, 2])

omega = obs_tensor[1:,2] - obs_tensor[:-1,2]
omega = torch.atan2(torch.sin(omega), torch.cos(omega)) / DT

for i in range(1, N):
    obs_t = torch.tensor([obs_tensor[i,0], obs_tensor[i,1],obs_tensor[i,2],vx[i-1],vy[i-1],omega[i-1]], device=DEVICE)
    mppi.feed_hist(obs_t, torch.tensor(action_list[i, :2], device=DEVICE))


target_pos_tensor = torch.Tensor(targets[N]).to(DEVICE).squeeze(dim=-1)
obs_N = torch.tensor([obs_tensor[N,0], obs_tensor[N,1],obs_tensor[N,2],vx[N-1],vy[N-1],omega[N-1]], device=DEVICE)
print("obs_N", obs_N)
action, mppi_info = mppi(obs_N, reward_track_fn(target_pos_tensor, SPEED), vis_all_traj=True, vis_optim_traj=True, use_nn=True)
all_traj = mppi_info['all_trajectory']
optim_traj = mppi_info['trajectory']
all_actions = mppi_info['action_candidate']
# plt.plot(obs_np[max(0,N-20):, 0], obs_np[max(0,N-20):, 1], alpha=0.5)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(obs_np[:, 0], obs_np[:, 1], alpha=0.5)
# print(all_traj)
for i in range(H+1):
    axs[0].scatter(all_traj[i][:, 0], all_traj[i][:, 1], s=1, alpha=0.2)
axs[0].plot(optim_traj[:, 0], optim_traj[:, 1], 'cyan', marker='o', )
axs[0].set_aspect('equal')

for i in range(H+1):
    axs[1].scatter(all_traj[i][:, 0], all_traj[i][:, 1], s=1, alpha=0.2,)
axs[1].plot(target_N[:, 0], target_N[:, 1], 'pink', marker='^', label='reference')
axs[1].plot(optim_traj[:, 0], optim_traj[:, 1], 'cyan', marker='o', label='optim traj')
axs[1].plot(obs_np[N:N+H+1, 0], obs_np[N:N+H+1, 1], alpha=1, marker='x', color='black', label='real traj')
axs[1].set_aspect('equal')
axs[1].legend()
# axs[2].hist2d(all_actions[:, 0,0], all_actions[:, 0, 1],norm=colors.LogNorm())
plt.suptitle("mppi-DBM")
# plt.tight_layout()
plt.show()