"""
MPPI for goal reaching ...
"""
import os
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import pickle
import time
from car_dynamics.envs import make_env, KinematicBicycleModel, KinematicParams
from car_dynamics.controllers_torch import MPPIController, rollout_fn_select, reward_track_fn
from car_dynamics.controllers_torch import PIDController
from car_dynamics.models_torch import MLP
from termcolor import colored

PROJ_DiR = '/Users/randyxiao/Library/CloudStorage/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/data'
timestr = time.strftime("%Y%m%d-%H%M%S")
logdir = 'sim-data-' + timestr
os.mkdir(os.path.join(PROJ_DiR, logdir))

DT = 0.05
VEL = 1.0
N_ROLLOUTS = 10000
H = 4
SIGMA = 1.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = make_env('car-base-single')
assert env.dt == DT
trajectory_type = "counter oval"

SPEED = 1.0
len_history = 2
LF = .16
LR = .15

model_struct = 'nn-end2end'

use_nn = False
random_walk = True

L = LF+LR



model_params = KinematicParams(
                    num_envs=N_ROLLOUTS,
                    last_diff_vel=torch.zeros([N_ROLLOUTS, 1]).to(DEVICE),
                    KP_VEL=7.,
                    KD_VEL=.02,
                    MAX_VEL=5.,
                    PROJ_STEER=.48,
                    SHIFT_STEER=-0.08,
                    DT=DT,
)   


dynamics = KinematicBicycleModel(model_params, device='cpu')


model_params_single = KinematicParams(
                    num_envs=1,
                    last_diff_vel=torch.zeros([1, 1]).to(DEVICE),
                    KP_VEL=7.,
                    KD_VEL=.02,
                    MAX_VEL=5.,
                    PROJ_STEER=.48,
                    SHIFT_STEER=-0.08,
                    DT=DT,
)   


dynamics_single = KinematicBicycleModel(model_params_single, device='cpu')


dynamics.reset()
dynamics_single.reset()



def rollout_start_kbm():
    dynamics.reset()
    
def rollout_start_nn():
    return


sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)

if model_struct == 'nn-heading':
    model_dir = f'/Users/randyxiao/Library/CloudStorage/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/tmp/20240127-202009/model.pt'
    dynamics_nn = MLP(input_size=6*len_history, hidden_size=512, output_size=3)
    dynamics_nn.load(model_dir)
    dynamics_nn.to(DEVICE)
    rollout_fn = rollout_fn_select(model_struct, dynamics_nn, DT, L, LR)
elif model_struct == 'nn-heading-psi':
    model_dir = f'/Users/randyxiao/Library/CloudStorage/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/tmp/20240127-224644/model.pt'
    dynamics_nn = MLP(input_size=6*len_history, hidden_size=128, output_size=5)
    dynamics_nn.load(model_dir)
    dynamics_nn.to(DEVICE)
    rollout_fn = rollout_fn_select(model_struct, dynamics_nn, DT, L, LR)
elif model_struct == 'nn-end2end':
    model_dir = f'/Users/randyxiao/Library/CloudStorage/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/tmp/20240128-114748/model.pt'
    dynamics_nn = MLP(input_size=7*len_history, hidden_size=256, output_size=5)
    dynamics_nn.load(model_dir)
    dynamics_nn.to(DEVICE)
    rollout_fn = rollout_fn_select(model_struct, dynamics_nn, DT, L, LR)
else:
    rollout_fn = rollout_fn_select('kbm', dynamics, DT, L, LR)


sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
# a_cov_prev =  torch.full((H, 2, 2), 3.0**2) * torch.eye(2).unsqueeze(0).repeat(H, 1, 1)

# import pdb; pdb.set_trace()

mppi = MPPIController(
    gamma_mean=1.0,
    gamma_sigma=0.0,
    discount=1.0,
    sample_sigma = 0.5,
    lam = 0.01,
    a_mean=torch.zeros(H, 2, device=DEVICE),
    a_cov = a_cov_init,
    n_rollouts=N_ROLLOUTS,
    H=H,
    device=DEVICE,
    rollout_fn=rollout_fn,
    a_min = [-1., -1],
    a_max = [1., 1.],
    a_mag = [.25, 1.],
    a_shift= [0.25, 0.],
    delay=0,
    len_history=len_history,
    rollout_start_fn=rollout_start_nn if use_nn else rollout_start_kbm,
)

done = False
frames = []


while not done:    
            

    target_pos_tensor = torch.Tensor(target_pos_list).to(DEVICE).squeeze(dim=-1)
    action, mppi_info = mppi(env.obs_state(), reward_track_fn(target_pos_tensor, SPEED), vis_optim_traj=False, use_nn=use_nn)
    # action += np.random.uniform(0, 0.1, size=(2))
    if random_walk:
        print(colored("random walk", "red"))
        action = np.zeros(2)
        action[0] += np.random.uniform(0, 0.2)
        action[1] += np.random.uniform(-1., 1.)
    # if t < 3 * DT:
    #     action *= 0.
    # pred_obs = dynamics_single.single_step_numpy(env.obs_state(), action.numpy())
    # print("pred", "input", env.obs_state(), action.numpy(), "\noutput", pred_obs)
    # if t < 5:
    #     action *= .0

    # mppi.feed_hist(env.obs_state(), action)
    
    obs_list.append(env.obs_state())
    
    car_state['obs'] = env.obs_state()
    car_state['throttle'] = action[0]
    car_state['steering'] = action[1]
    # car_state['mppi_sample_actions'] = mppi_info['action_candidate']
    car_state['targets'] = target_pos_list
    # car_state['mppi_actions'] = mppi_info['action']
    # car_state['mppi_traj'] = mppi_info['trajectory']
    dataset.append(car_state.copy())
    
    
    obs, reward, done, info = env.step(action)
    
    
    
    print("new obs", env.obs_state())
    # print("real", env.pos[0], env.pos[1], env.yaw, env.vel)
    # frames.append(env.render(mode='rgb_array'))
    # dynamics.params.MAX_VEL += np.random.uniform(-.5, .5)
    # dynamics.params.PROJ_STEER += np.random.uniform(-.2, .2)
    # dynamics.params.SHIFT_STEER += np.random.uniform(-.1, .1)
    # import pdb; pdb.set_trace()
    
    ### Gather Logging info
    # action_list.append(action.numpy())
    # mppi_action_list.append(mppi_info['action'])
    # mppi_traj_list.append(mppi_info['trajectory'])
    # mppi_sampled_action.append(mppi_info['action_candidate'])
    
    t += DT
    
    if t > DT * 8000:
        print("exit", t)
        break
    
print("len dataset", len(dataset))
obs_list = np.array(obs_list)
target_list = np.array(target_list)
action_list = np.array(action_list)
mppi_action_list = np.array(mppi_action_list)
mppi_traj_list = np.array(mppi_traj_list)
mppi_sampled_action = np.array(mppi_sampled_action)

with open(os.path.join(PROJ_DiR, logdir, f"log0.pkl"), "wb") as f:
    pickle.dump(dataset, f)
