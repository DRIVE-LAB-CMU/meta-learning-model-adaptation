from mlagents_envs.environment import UnityEnvironment,ActionTuple

import numpy as np
import pickle
import os
import tqdm
import argparse
import yaml


DT = .05
N_ROLLOUTS = 100
N_episode_per_param = 10
var_params = True
H = 1000
SIGMA = 1.0

BASE_NAME = 'data_states_actions'
if var_params:
    BASE_NAME += '_var'

env = UnityEnvironment(file_name="ros2-env-v2/sim", seed=1,worker_id=0,log_folder='logs/',no_graphics=False)
env.reset()
datas_states = []
datas_actions = []
ts = np.arange(0.,2.,.2)


for i in range(N_ROLLOUTS):
    if (i) % N_episode_per_param == 0 and var_params:
        env.close()
        params = yaml.load(open('params_orig.yaml','r'), Loader=yaml.FullLoader)
        # Randomize params here
        params['vehicle_params']['K_cmd'] = np.random.uniform(1000., 3000.)
        params['vehicle_params']['K_brake'] = np.random.uniform(1000., 3000.)
        params['vehicle_params']['mu_f'] = np.random.uniform(.5, 1.)
        params['vehicle_params']['mu_r'] = np.random.uniform(.5, 1.)
        yaml.dump(params,open('params.yaml','w'))
        env = UnityEnvironment(file_name="ros2-env-v2/sim", seed=1,worker_id=0,log_folder='logs/',no_graphics=False)
    env.reset()
    amps_steer = np.random.random(ts.shape)
    amps_steer[0] *= 4.
    amps_steer = amps_steer/np.sum(amps_steer)
    if np.random.random() < .5:
        amps_steer = -amps_steer
    
    amps_throttle = np.random.random(ts.shape)
    amps_throttle[0] *= 5.
    amps_throttle = amps_throttle/np.sum(amps_throttle)
    
    behavior_name = list(env.behavior_specs)[0]
    done = False
    states = []
    actions = []
    for t in tqdm.tqdm(range(H)):
        steer = 0.
        throttle = 0.
        env_info = env.get_steps(behavior_name)
        x = float(env_info[0].obs[-1][0][0])
        y = float(env_info[0].obs[-1][0][1])
        z = float(env_info[0].obs[-1][0][2])
        roll = float(env_info[0].obs[-1][0][3])*np.pi/180.
        pitch = float(env_info[0].obs[-1][0][4])*np.pi/180.
        yaw = float(env_info[0].obs[-1][0][5])*np.pi/180.
        while pitch > np.pi:
            pitch -= 2.*np.pi
        while pitch < -np.pi:
            pitch += 2.*np.pi
        while roll > np.pi:
            roll -= 2.*np.pi
        while roll < -np.pi:
            roll += 2.*np.pi
        while yaw > np.pi:
            yaw -= 2.*np.pi
        while yaw < -np.pi:
            yaw += 2.*np.pi
        vx = float(env_info[0].obs[-1][0][6])
        vy = float(env_info[0].obs[-1][0][7])
        vz = float(env_info[0].obs[-1][0][8])
        wx = float(env_info[0].obs[-1][0][9])
        wy = float(env_info[0].obs[-1][0][10])
        wz = float(env_info[0].obs[-1][0][11])
        ax = float(env_info[0].obs[-1][0][12])
        ay = float(env_info[0].obs[-1][0][13])
        az = float(env_info[0].obs[-1][0][14])
        for j in range(len(ts)):
            if j == 0 :
                steer += amps_steer[j]
                throttle += amps_throttle[j]
            else :
                steer += amps_steer[j]*np.sin(2*np.pi*t*DT/ts[j])
                throttle += amps_throttle[j]*np.sin(2*np.pi*t*DT/ts[j])
        action = ActionTuple(np.array([[steer,throttle]]),None)
        env.set_actions(behavior_name, action)
        env.step()
        states.append([x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,ax,ay,az])
        actions.append([steer,throttle])
    # print(np.array(states),np.array(actions))
    datas_states.append(states)
    datas_actions.append(actions)
env.close()
data_states = np.array(datas_states)
data_actions = np.array(datas_actions)
pickle.dump((data_states,data_actions),open(BASE_NAME+'_.pkl','wb'))