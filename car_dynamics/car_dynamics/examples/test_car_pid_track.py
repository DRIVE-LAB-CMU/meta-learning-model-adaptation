"""
MPPI for goal reaching ...
"""

import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt


from car_dynamics.envs import make_env, KinematicBicycleModel, KinematicParams
from car_dynamics.controllers_torch import PIDController

N_ROLLOUTS = 10000
H = 5
SIGMA = 1.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = make_env('car-base-single')

pid = PIDController(kp=5.0, ki=0.0, kd=0.1)

done = False
frames = []

obs_list = []

obs = env.reset()
# dynamics.reset()
trajectory_list = []

# goal_list = np.array([
#     # [1., 1.],
#     # [3, 3],
#     # [5, 5],
#     [7, 7],
#     [12, 12],
#     # [5, 7],
#     # [3, 7],
#     [5, 5],
#     # [7, 3],
#     # [4.5, 6],
#     # [5, 10],
#     # [3.5, 6],
#     # [3, 3],
# ])

center_circle = (6, 6)
circle_radius = 6
circle_slices = 32
goal_list = []
for i in range(circle_slices + 1):
    angle = 2 * np.pi * i / circle_slices - np.pi / 4
    goal_list.append([
        center_circle[0] + circle_radius * np.cos(angle),
        center_circle[1] + circle_radius * np.sin(angle),
    ])
goal_list = np.array(goal_list)

def get_steer_error(obs, goal):
    psi = obs[2]
    psi_des = np.arctan2(goal[1] - obs[1], goal[0] - obs[0])
    err = psi_des - psi
    # err = (err + np.pi) % (2 * np.pi) - np.pi
    while err > np.pi:
        err -= 2 * np.pi
    while err < -np.pi:
        err += 2 * np.pi
    # print(psi, psi_des, err)
    return err
    
    
goal_ptr = 0
error_list = []
steering_list = []
pos_err = []
idx = 0
while not done:
    steering_error = get_steer_error(env.obs_state(), goal_list[goal_ptr])
    steering = pid(steering_error, 0.05)
    # steering = 1.0
    error_list.append(steering_error)
    # print(steering_error)
    steering_list.append(steering)
    action = np.array([0.64, steering])
    obs, reward, done, info = env.step(action)
    # print("real", env.pos[0], env.pos[1], env.yaw, env.vel)
    # frames.append(env.render(mode='rgb_array'))
    # dynamics.params.MAX_VEL += np.random.uniform(-.5, .5)
    # dynamics.params.PROJ_STEER += np.random.uniform(-.2, .2)
    # dynamics.params.SHIFT_STEER += np.random.uniform(-.1, .1)
    obs_list.append(obs)
    pos_err.append(np.linalg.norm(env.pos - goal_list[goal_ptr]))
    if np.linalg.norm(env.pos - goal_list[goal_ptr]) < 0.1:
        print("goal reached")
        goal_ptr += 1
        if goal_ptr >= len(goal_list):
            break
        
    idx += 1
        
print(goal_ptr)
    
obs_list = np.array(obs_list)

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].plot(obs_list[:, 0], obs_list[:, 1], label='track trajectory')
# axs[0].scatter(obs_list[0, 0], obs_list[0, 1], marker='o', label='start')
# axs[0].scatter(goal_list[:, 0], goal_list[:, 1], marker='x', c='green', label='waypoints')
# axs[0].plot(goal_list[:, 0], goal_list[:, 1], label='reference trajectory')
# axs[0].legend()
# axs[0].axis('equal')
# axs[0].set_title("MPPI for trajectory tracking")

# axs[1].plot(error_list, label='steering error')
# axs[1].plot(steering_list, label='steering')
# axs[1].legend()

# axs[2].plot(pos_err, label='pos error')
# axs[2].legend()

plt.plot(obs_list[:, 0], obs_list[:, 1], label='track trajectory')
plt.scatter(obs_list[0, 0], obs_list[0, 1], marker='o', label='start')
plt.scatter(goal_list[:, 0], goal_list[:, 1], marker='x', c='green', label='waypoints')
plt.plot(goal_list[:, 0], goal_list[:, 1], label='reference trajectory')
plt.legend()
plt.axis('equal')
plt.title("PID for trajectory tracking")

plt.savefig(f'tmp/test_pid_track.png')
    
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
    
