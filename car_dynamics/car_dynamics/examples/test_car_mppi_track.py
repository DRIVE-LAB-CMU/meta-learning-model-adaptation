"""
MPPI for goal reaching ...
"""

# import imageio
import numpy as np
# import torch
import matplotlib.pyplot as plt
import json

from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_jax import MPPIController, MPPIParams, rollout_fn_select
from car_dynamics.envs.car3 import OffroadCar
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
print("DEVICE", jax.devices())


DT = .05
VEL = 1.0
N_ROLLOUTS = 10000
H = 40
SIGMA = 1.0
LF = .16
LR = .15
L = LF+LR

trajectory_type = "counter circle"

SPEED = 1.0


model_params =  DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=0.36, Sb=0.0, K_FFY=20, K_RFY=20, Ta=8., Tb=0.)


dynamics = DynamicBicycleModel(model_params)


model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.36, Sb=0.0, K_FFY=20, K_RFY=20, Ta=8., Tb=0.)

dynamics_single = DynamicBicycleModel(model_params_single)
env =OffroadCar({}, dynamics_single)

rollout_fn = rollout_fn_select('dbm', dynamics, DT, L, LR)

dynamics.reset()
dynamics_single.reset()



# a_cov_prev =  torch.full((H, 2, 2), 3.0**2) * torch.eye(2).unsqueeze(0).repeat(H, 1, 1)

# import pdb; pdb.set_trace()

def fn():
    ...
    
mppi_params = MPPIParams(
        sigma = 1.0,
        gamma_sigma=0.0,
        gamma_mean = 1.0,
        discount=.99,
        sample_sigma = 0.5,
        lam = 0.01,
        n_rollouts=N_ROLLOUTS,
        H=H,
        a_min = [-1., -1.],
        a_max = [1., 1.],
        a_mag = [.5, 1.], # 0.1, 0.35
        a_shift = [0.5, 0.],
        delay=0,
        len_history=2,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
)


mppi = MPPIController(
    mppi_params, rollout_fn, fn, key
)

done = False
frames = []


obs = env.reset()

def reference_traj(t):
    if trajectory_type == 'circle':
        
        # global total_angle
        center_circle = (.8, 1.2)
        circle_radius = 1.2
        angle = -np.pi/2  - circle_radius * SPEED * t
        return np.array([center_circle[0] + circle_radius * np.cos(angle),
                            center_circle[1] + circle_radius * np.sin(angle)])
        
    elif trajectory_type == 'counter circle':
        
        # global total_angle
        center_circle = (.9, 1.2)
        circle_radius = 1.2
        angle = -np.pi/2  + circle_radius * SPEED * t
        return np.array([center_circle[0] + circle_radius * np.cos(angle),
                            center_circle[1] + circle_radius * np.sin(angle)])
    elif trajectory_type == 'oval':
        
        center = (0.9, 1.0)
        x_radius = 1.2
        y_radius = 1.4

        # Assuming t varies from 0 to 2π to complete one loop around the oval
        angle = -np.pi/2  - x_radius * SPEED * t

        x = center[0] + x_radius * np.cos(angle)
        y = center[1] + y_radius * np.sin(angle)

        return np.array([x, y])

    elif trajectory_type == 'counter oval':
        center = (0.8, 1.0)
        x_radius = 1.2
        y_radius = 1.4

        # Assuming t varies from 0 to 2π to complete one loop around the oval
        angle = -np.pi/2  + x_radius * SPEED * t
mppi
goal_list = []
target_list = []
action_list = []
mppi_action_list = []
obs_list = []

waypoint_t_list = np.arange(-np.pi*2-DT, np.pi*2+DT, 0.01)
waypoint_list = np.array([reference_traj(t) for t in waypoint_t_list])

plt.plot(waypoint_list[:, 0], waypoint_list[:, 1])
plt.savefig('waypoints.png')
pos2d = []
target_list_all = []
for _ in range(1000):   
    distance_list = np.linalg.norm(waypoint_list - obs[:2], axis=-1)
    # import pdb; pdb.set_trace()
    t_idx = np.argmin(distance_list)
    t_closed = waypoint_t_list[t_idx]
    target_pos_list = [reference_traj(0. + t_closed + i*DT*.1) for i in range(H+0+1)]
    target_pos_tensor = jnp.array(target_pos_list)
    dynamics.reset()
    target_list_all += target_pos_list
    plt.figure()
    plt.scatter(target_pos_tensor[:, 0], target_pos_tensor[:, 1], color='red')
    plt.scatter(obs[0], obs[1], color='green', marker='x')
    plt.plot(waypoint_list[:, 0], waypoint_list[:, 1])
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.axis('equal')
    plt.savefig("target.png")
    # action, mppi_info = mppi(obs, reward_fn(target_pos_tensor))
    # print("obs", env.obs_state())
    action, mppi_info = mppi(env.obs_state(),target_pos_tensor,vis_optim_traj=True)
    action = np.array(action)
    # print(action)
    import pdb; pdb.set_trace()
    # action = np.zeros(2)
    # action *= 0.
    # action = np.array([.1, 1.])
    obs, reward, done, info = env.step(action)
    # print("new obs", env.obs_state())
    pos2d.append(env.obs_state()[:2])
    obs_list.append(env.obs_state())    

pos2d = np.array(pos2d)
obs_list = np.array(obs_list)
plt.plot(pos2d[:,0], pos2d[:,1])
# save to test.png
plt.scatter(pos2d[0,0], pos2d[0,1], color='r',s=10)
plt.scatter(pos2d[-1,0], pos2d[-1,1], color='g',s=10, marker='x')

target_list_all = np.array(target_list_all)
plt.scatter(target_list_all[:, 0], target_list_all[:, 1], color='blue')
# plt.plot(obs_list[:, 2])
plt.axis('equal')
plt.savefig('test.png')

    
plt.figure()
plt.plot(obs_list[:, 3])


# plt.plot(obs_list[:, 4])
# plt.show()
plt.savefig('test2.png')
