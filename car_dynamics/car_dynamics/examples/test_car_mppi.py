"""
MPPI for goal reaching ...
"""

import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt


from car_dynamics.envs import make_env, KinematicBicycleModel, KinematicParams
from car_dynamics.controllers_torch import MPPIController

N_ROLLOUTS = 1000
H = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = make_env('car-base-single')



def reward_fn(goal):
    def reward(state, action):
        # import pdb; pdb.set_trace()
        return -torch.norm(state[:, :2] - goal[:2], dim=1) - 0.1 * torch.norm(action, dim=1)
    return reward

model_params = KinematicParams(
                    num_envs=N_ROLLOUTS,
                    last_diff_vel=torch.zeros([N_ROLLOUTS, 1]).to(DEVICE),
                    KP_VEL=7.,
                    KD_VEL=.02,
                    MAX_VEL=5.,
                    PROJ_STEER=.34,
                    SHIFT_STEER=0.,
)   


dynamics = KinematicBicycleModel(model_params, device='cpu')
dynamics.reset()
def rollout_fn(state, action):
    next_state = dynamics.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], 
                               action[:, 0], action[:, 1])
    # import pdb; pdb.set_trace()
    return torch.stack(next_state, dim=1)

mppi = MPPIController(
    gamma_mean=1.0,
    gamma_sigma=0.0,
    discount=0.99,
    sample_sigma = 0.5,
    lam = 0.01,
    a_mean=torch.zeros(H, 2, device=DEVICE),
    a_cov = torch.full((H, 2, 2), 1**2) * torch.eye(2).unsqueeze(0).repeat(H, 1, 1)
,
    n_rollouts=N_ROLLOUTS,
    H=H,
    device=DEVICE,
    rollout_fn=rollout_fn,
)
done = False
frames = []

obs_list = []

obs = env.reset()
# dynamics.reset()
trajectory_list = []
while not done:
    action, mppi_info = mppi(env.obs_state(), reward_fn(env.goal))
    trajectory_list.append(mppi_info['trajectory'])
    obs, reward, done, info = env.step(action)
    # print("real", env.pos[0], env.pos[1], env.yaw, env.vel)
    # frames.append(env.render(mode='rgb_array'))
    obs_list.append(obs)

    
obs_list = np.array(obs_list)

plt.plot(obs_list[:, 0], obs_list[:, 1])
plt.scatter(obs_list[0, 0], obs_list[0, 1], marker='o', label='start')
plt.scatter(env.goal[0], env.goal[1], marker='x', label='goal')
plt.legend()
plt.savefig(f'tmp/test_mppi.png')
    
# for i, trajectory in enumerate(trajectory_list):
#     print(i)
#     plt.clf()
#     for state_rollout in trajectory:
#         plt.plot(state_rollout[:, 0], state_rollout[:, 1], alpha=0.1)
#     plt.plot(obs_list[:, 0], obs_list[:, 1])
#     plt.scatter(obs_list[0, 0], obs_list[0, 1], marker='o', label='start')
#     plt.scatter(env.goal[0], env.goal[1], marker='x', label='goal')
#     plt.legend()
#     plt.savefig(f'tmp/traj/{i}.png')
    
