import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
import random

import numpy as np
import matplotlib.pyplot as plt

from tf_transformations import quaternion_from_euler, euler_matrix

from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_torch import rollout_fn_select as rollout_fn_select_torch
from car_dynamics.controllers_torch import reward_track_fn, MPPIController as MPPIControllerTorch
from car_dynamics.controllers_jax import MPPIController, MPPIParams, rollout_fn_select
from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.controllers_jax import WaypointGenerator
from std_msgs.msg import Float64
import torch.nn as nn
import torch.optim as optim
import torch
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
import tqdm

print("DEVICE", jax.devices())

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')
parser.add_argument('--var', action='store_true', help='Enable variable parameters')
parser.add_argument('--var_delay', action='store_true', help='Enable variable delay')
parser.add_argument('--const_delay', default=0, help='Use constant delay')
# parser.add_argument('--pre', action='store_true', help='Enable pre-training')

args = parser.parse_args()

filename = 'losses/'+args.exp_name+'.txt'
DT = .05
N_ROLLOUTS = 100
var_params = args.var
if not var_params :
    N_ROLLOUTS = 10000
H = 50
SIGMA = 1.0
LF = .16
LR = .15
L = LF+LR
learning_rate = 0.002
trajectory_type = "counter oval"
losses = []
SPEED = 2.2

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
DELAY = 5
MODEL = 'nn'
AUGMENT = False
HISTORY = 8
ART_DELAY = 0

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.fc2.bias.data = torch.tensor([0.,0.,0.])

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Define fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # Concatenate states and commands
        # inputs = torch.cat((states, commands), dim=2)
        inputs = X.view(X.shape[0], -1, 3+2)
        # print(inputs.shape)
        # LSTM layer
        lstm_out, _ = self.lstm(inputs)
        
        # Get the last output of the LSTM sequence
        lstm_out_last = lstm_out[:, -1, :]
        
        # Fully connected layer
        output = self.fc(lstm_out_last)
        
        return output

if MODEL == 'nn' :
    model = SimpleModel(5*HISTORY,300,3)
elif MODEL == 'nn-lstm' :
    model = LSTMModel(5,10,3)

# model.load_state_dict(torch.load('model_new.pth'))
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
model = model.double()
stats = {'losses': []}
rollout_fn_torch = rollout_fn_select_torch('nn', model, DT, L, LR)

def fn():
    return
 
def fn():
    ...


waypoint_generator = WaypointGenerator(trajectory_type, DT, H, SPEED)
done = False
frames = []


goal_list = []
target_list = []
action_list = []
mppi_action_list = []
obs_list = []


pos2d = []
target_list_all = []

def build_X(states,actions): 
    X_ = np.concatenate((np.array(states),np.array(actions)),axis=1)
    X = X_.reshape(-1)
    return torch.tensor(X).unsqueeze(0).double()

def rollout_nn(states,actions,state) :
    global model
    traj = []
    state = torch.tensor(state).double()
    # states = states[:,3:]
    actions = torch.tensor(actions).double()
    for i in range(len(actions)-len(states)+1) :
        # print(states.shape,actions[i:i+len(states)].shape)
        X = build_X(states,actions[i:i+len(states)])
        # print(X.shape)
        y = model(X).detach()
        # print(model(X))
        state[3] += y[0,0]*DT
        state[4] += y[0,1]*DT
        state[5] += y[0,2]*DT
        state[0] += state[3]*np.cos(state[2])*DT - state[4]*np.sin(state[2])*DT
        state[1] += state[3]*np.sin(state[2])*DT + state[4]*np.cos(state[2])*DT
        state[2] += state[5]*DT
        states[:-1] = states[1:]
        states[-1,:] = state[3:]
        # print(state)
        traj.append([float(state[0]),float(state[1]),float(state[2]),float(state[3]),float(state[4]),float(state[5])])
    # print(np.array(traj))
    return np.array(traj)

def train_step(states,cmds,augment=True,art_delay=ART_DELAY) :
    global model, stats
    if art_delay > 0 :
        states = states[:-art_delay]
        cmds = cmds[art_delay:]
    elif art_delay < 0 :
        cmds = cmds[:art_delay]
        states = states[-art_delay:]
    X_ = np.concatenate((np.array(states),np.array(cmds)),axis=1)[:-1]
    Y_ = (np.array(states)[1:] - np.array(states)[:-1])
    X = []
    for i in range(HISTORY-1) :
        X.append(X_[i:-HISTORY+i+1])
    X.append(X_[HISTORY-1:])
    X = np.concatenate(X,axis=1)
    Y = Y_[HISTORY-1:]
    # print(X,Y)
    
    # Augmentation
    if augment :
        X_ = X.copy()
        X_[:,1] = -X[:,1]
        X_[:,2] = -X[:,2]
        X_[:,4] = -X[:,4]
        Y_ = Y.copy()
        Y_[:,1] = -Y[:,1]
        Y_[:,2] = -Y[:,2]
        
        X = np.concatenate((X,X_),axis=0)    
        Y = np.concatenate((Y,Y_),axis=0)
    
        X_ = X.copy()
        X_[:,1] = 0.
        X_[:,2] = 0.
        X_[:,4] = 0.
        Y_ = Y.copy()
        Y_[:,1] = 0.
        Y_[:,2] = 0.
        
        X = np.concatenate((X,X_),axis=0)
        Y = np.concatenate((Y,Y_),axis=0)
    
    X = torch.tensor(X).double()
    # X[:,3] = 0.
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)*DT
    
    Y = torch.tensor(Y)
    # Compute the loss
    loss = criterion(outputs/DT, Y/DT)
    stats['losses'].append(loss.item())
    # print("Loss",loss.item())
    # Backward pass and optimization
    loss.backward()
    optimizer.step()    
    
    return

mppi_params = MPPIParams(
        sigma = 1.0,
        gamma_sigma=.0,
        gamma_mean = 1.0,
        discount=1.,
        sample_sigma = 1.,
        lam = 0.01,
        n_rollouts=N_ROLLOUTS,
        H=H,
        a_min = [-1., -1.],
        a_max = [1., 1.],
        a_mag = [1., 1.], # 0.1, 0.35
        a_shift = [.0, 0.],
        delay=0,
        len_history=2,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
        smooth_alpha=1.,
)

init_state = [0., 0., 0., 2., 0., 0.]
datas_states = []
datas_actions = []
if not var_params:
    n_data = 1
else :
    n_data = 100
# for i in range(100) :
for i in range(n_data) :
    print("Generating data: ", i, "/ 100")
    model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=random.uniform(0.1,0.75), Sb=random.uniform(-0.1,0.1),Ta=random.uniform(5.,45.), Tb=.0, mu=random.uniform(0.35,0.65),delay=1)
    
    # model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=0.36, Sb=0.0,Ta=20., Tb=.0, mu=0.5,delay=1)

    dynamics = DynamicBicycleModel(model_params)
    dynamics.reset()
    rollout_fn = rollout_fn_select('dbm', dynamics, DT, L, LR)
    mppi = MPPIController(
        mppi_params, rollout_fn, fn, key, nn_model=None
    )

    target_pos_tensor = waypoint_generator.generate(jnp.array(init_state[:5]))
    target_pos_list = np.array(target_pos_tensor)

    action, mppi_info = mppi(init_state,target_pos_tensor, vis_optim_traj=True, model_params=None)
    data_states = torch.tensor(np.array(mppi_info['all_traj'])).double()
    data_actions = torch.tensor(np.array(mppi_info['all_action'])).double()
    print(data_states.shape, data_actions.shape)
    datas_states.append(data_states)
    datas_actions.append(data_actions)

data_states = torch.concat(datas_states,dim=1)
data_actions = torch.concat(datas_actions,dim=0)
print(data_states.shape, data_actions.shape)
n_epochs = 100
for epoch in range(n_epochs):
    for i in tqdm.tqdm(range(data_states.shape[1])):
        if args.var_delay:
            delay = np.random.randint(0,5)
        else :
            delay = int(args.const_delay)
        # print(delay)
        train_step(data_states[:-1,i,3:],data_actions[i,:,:],augment = False,art_delay=delay)
    print("Epoch",epoch,"Loss",np.mean(stats['losses'][-data_states.shape[1]:]))
    losses.append(np.mean(stats['losses'][-data_states.shape[1]:]))

torch.save(model.state_dict(), 'model_fixed_params_var_delay.pth')
np.savetxt(filename, losses)