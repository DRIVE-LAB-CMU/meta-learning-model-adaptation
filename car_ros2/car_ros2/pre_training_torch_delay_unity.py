from mlagents_envs.environment import UnityEnvironment,ActionTuple

import numpy as np
import pickle
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')
parser.add_argument('--var', action='store_true', help='Enable variable parameters')
parser.add_argument('--var_delay', action='store_true', help='Enable variable delay')
parser.add_argument('--const_delay', default=0, help='Use constant delay')
parser.add_argument('--use_gt', action='store_true', help='Use GT delay')
parser.add_argument('--lstm', action='store_true', help='Train LSTM model')
# parser.add_argument('--pre', action='store_true', help='Enable pre-training')

args = parser.parse_args()

filename = 'losses/'+args.exp_name+'.txt'
DT = .05
N_ROLLOUTS = 100
var_params = args.var
if not var_params :
    N_ROLLOUTS = 10000
H = 1000
SIGMA = 1.0

BASE_NAME = 'data_states_actions'
if args.var:
    BASE_NAME += '_var'

env = UnityEnvironment(file_name="ros2-env-v2/sim", seed=1,worker_id=0,log_folder='logs/')
datas_states = []
datas_actions = []
ts = np.arange(0.,2.,.2)

for i in range(N_ROLLOUTS):
    env.reset()
    amps_steer = np.random.random(ts.shape)
    amps_steer = amps_steer/np.sum(amps_steer)
    
    amps_throttle = np.random.random(ts.shape)
    amps_throttle = amps_throttle/np.sum(amps_throttle)
    behavior_name = list(env.behavior_specs)[0]
    done = False
    states = []
    actions = []
    for t in range(H):
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
        
    datas_states.append(states)
    datas_actions.append(actions)
if not os.path.exists(BASE_NAME+'.pkl'):
    data_states = np.array(datas_states)
    data_actions = np.array(datas_actions)
    pickle.dump((data_states,data_actions),open(BASE_NAME+'.pkl','wb'))