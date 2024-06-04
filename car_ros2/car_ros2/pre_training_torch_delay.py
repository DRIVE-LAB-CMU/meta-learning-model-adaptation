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
import time
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
import pickle
import tqdm
import os
print("DEVICE", jax.devices())

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
LF = .16
LR = .15
L = LF+LR
learning_rate = 0.002
trajectory_type = "counter oval"
losses = []
SPEED = 2.2
batch_size = 64
N_ensembles = 3
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
MAX_DELAY = 7
if args.lstm:
    MODEL = 'nn-lstm'
else :
    MODEL = 'nn'
AUGMENT = False
HISTORY = 8
ART_DELAY = 0
append_delay_type = "OneHot" # From "None", "Single", "OneHot"

def convert_delay_to_onehot(delay, max_delay=MAX_DELAY):
    onehot = np.zeros((delay.shape[0],max_delay))
    inds = np.arange(delay.shape[0])
    frac = delay - delay.astype(int)
    onehot[inds,delay.astype(int)] = 1 - frac
    onehot[inds,(delay.astype(int)+1)%max_delay] = frac
    return onehot

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = []
        curr_h = input_size
        for h in hidden_size:
            self.fc.append(nn.Linear(curr_h, h))
            self.fc.append(nn.ReLU())
            curr_h = h
        self.fc.append(nn.Linear(curr_h, output_size))
        self.fc = nn.Sequential(*self.fc)
        
    def forward(self, x):
        
        out = self.fc(x)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.num_layers = 1
        # Define LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # Define fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, X, h_0=None, c_0=None):
        # Concatenate states and commands
        # inputs = torch.cat((states, commands), dim=2)
        # Initialize hidden and cell states if not provided
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)
        
        inputs = X.view(X.shape[0], -1, 3+2)
        # print(inputs.shape)
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(inputs)
        
        
        # Fully connected layer
        output = self.fc(lstm_out)
        
        return output, (hn, cn)

models = []
if MODEL == 'nn' :
    if append_delay_type=='OneHot':
        for i in range(N_ensembles):
            model = SimpleModel(5*HISTORY+MAX_DELAY,[100,100],3)
            models.append(model)
        model_delay = SimpleModel(5*HISTORY,[100,100,100],MAX_DELAY)
    else :
        model = SimpleModel(5*HISTORY+1,[300],3)
        model_delay = SimpleModel(5*HISTORY,[150,150],2)
    # model = SimpleModel(5*HISTORY+MAX_DELAY,300,3)
elif MODEL == 'nn-lstm' :
    for i in range(N_ensembles):
        model = LSTMModel(5,300,3)
        models.append(model)
    model_delay = SimpleModel(5*HISTORY,[150,150],2)

# Check if file exist, if yes, then load the model
if os.path.exists(filename[:-4]+'.pth'):
    model.load_state_dict(torch.load(filename[:-4]+'.pth'))
    model_delay.load_state_dict(torch.load(filename[:-4]+'_delay.pth'))

for model in models:
    model = model.double()
# print(model.fc[0].weight.data)
# Define loss function and optimizer
criterion = nn.MSELoss()
criterion_delay = nn.CrossEntropyLoss()
print(model.fc)
optimizers = []
for i in range(N_ensembles):
    optimizer = optim.SGD(models[i].parameters(), lr=learning_rate)
    optimizers.append(optimizer)
optimizer_delay = optim.SGD(model_delay.parameters(), lr=1e-3)
model_delay = model_delay.double()
stats = {'losses': [], 'losses_delay': []}
for i in range(N_ensembles):
    stats['losses'+str(i)] = []

def fn():
    return
 
def fn():
    ...


waypoint_generator = WaypointGenerator(trajectory_type, DT, H, SPEED)

def acc_X(states,cmds,augment=False,art_delay=ART_DELAY,one_hot_delay=None,print_pred_delay=False, train=False) :
    global stats, model_delay
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
    # print(one_hot_delay)
    X = np.concatenate((X,one_hot_delay.repeat(X.shape[0],axis=0)),axis=1)
    Y = Y_[HISTORY-1:]
    # print(X,Y)
    
    
    X = torch.tensor(X).double()
    
    # Zero the parameter gradients
    optimizer_delay.zero_grad()

    # Forward pass
    outputs_delay = model_delay(X[:,:5*HISTORY])
    # outputs_delay = torch.sigmoid(outputs_delay)
    # print(one_hot_delay.shape,outputs_delay.shape)
    # print(torch.max(outputs_delay),torch.min(outputs_delay))
    loss_delay = criterion_delay(outputs_delay,torch.tensor([art_delay]*outputs_delay.shape[0]).long())
    stats['losses_delay'].append(loss_delay.item())
    loss_delay.backward()
    optimizer_delay.step()
    pred_delay = torch.argmax(outputs_delay,dim=1).double()
    # print(pred_delay)
    if print_pred_delay :
        print("Pred delay: ",pred_delay,art_delay)
    Y = torch.tensor(Y)
    
    return X, Y

def train_X(X, Y) :
    global models, stats, model_delay
    # Zero the parameter gradients
    
    # Forward pass
    Y = torch.tensor(Y)
    j = 0
    for model in models :
        optimizer = optimizers[j]
        optimizer.zero_grad()
        outputs = model(X)
        # Compute the loss
        loss = criterion(outputs, Y/DT)
        stats['losses' + str(j)].append(float(loss.item()))
        j += 1
        loss.backward()
        optimizer.step()
    return

def acc_X_lstm(states,cmds,art_delay=ART_DELAY) :
    global model, stats
    if art_delay > 0 :
        states = states[:-art_delay]
        cmds = cmds[art_delay:]
    elif art_delay < 0 :
        cmds = cmds[:art_delay]
        states = states[-art_delay:]
    X = np.concatenate((np.array(states),np.array(cmds)),axis=1)[:-1]
    Y = (np.array(states)[1:] - np.array(states)[:-1])
    X = torch.tensor(X).double().unsqueeze(0)
    Y = torch.tensor(Y).double().unsqueeze(0)
    
    return X, Y

def train_X_lstm(X, Y) :
    global model, stats
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    
    outputs, cn = model(X)
    Y = torch.tensor(Y)
    # Compute the loss
    loss = criterion(outputs, Y/DT)
    stats['losses'].append(float(loss.item()))
    loss.backward()
    optimizer.step()
    return

def train_step(states,cmds,augment=True,art_delay=ART_DELAY,one_hot_delay=None,predict_delay=False,print_pred_delay=False, train=False) :
    global model, stats, model_delay
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
    # print(one_hot_delay)
    X = np.concatenate((X,one_hot_delay.repeat(X.shape[0],axis=0)),axis=1)
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
    optimizer_delay.zero_grad()

    # Forward pass
    outputs = model(X)*DT
    outputs_delay = model_delay(X[:,:5*HISTORY])
    # outputs_delay = torch.sigmoid(outputs_delay)
    # print(one_hot_delay.shape,outputs_delay.shape)
    # print(torch.max(outputs_delay),torch.min(outputs_delay))
    loss_delay = criterion_delay(outputs_delay,torch.tensor([art_delay]*outputs_delay.shape[0]).long())
    stats['losses_delay'].append(loss_delay.item())
    loss_delay.backward()
    optimizer_delay.step()
    pred_delay = torch.argmax(outputs_delay,dim=1).double()
    # print(pred_delay)
    if print_pred_delay :
        print("Pred delay: ",pred_delay,art_delay)
    Y = torch.tensor(Y)
    # Compute the loss
    loss = criterion(outputs/DT, Y/DT)
    stats['losses'].append(float(loss.item()))
    # print(art_delay)
    # print("Before: ",stats['losses'][-1])
    # print("Loss",loss.item())
    # Backward pass and optimization
    loss.backward()
    # if train: 
    #     optimizer.step()
    # print("After: ",stats['losses'][-1])
    
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

BASE_NAME = 'data_states_actions'
if args.var:
    BASE_NAME += '_var'
if not os.path.exists(BASE_NAME+'.pkl'):
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
    print(data_states.shape)
    perm = np.random.permutation(data_states.shape[1])
    data_states = data_states[:, perm]
    print(data_states.shape)
    
    data_actions = torch.concat(datas_actions,dim=0)
    print(data_actions.shape)
    data_actions = data_actions[perm]
    print(data_actions.shape)
    
    pickle.dump((data_states,data_actions),open(BASE_NAME+'.pkl','wb'))
(data_states,data_actions) = pickle.load(open(BASE_NAME+'.pkl','rb'))
n_epochs = 100

for epoch in range(n_epochs):
    for j in tqdm.tqdm(range(0,data_states.shape[1]-batch_size,batch_size)):
        Xs = []
        Ys = []
        for i in range(j,j+batch_size):
            if args.var_delay:
                delay = np.random.randint(0,5)
            else :
                delay = int(args.const_delay)
            if append_delay_type=='OneHot':
                if args.use_gt:
                    one_hot_delay = convert_delay_to_onehot(np.array([delay]))
                else :
                    one_hot_delay = convert_delay_to_onehot(np.random.randint(0,MAX_DELAY,size=(1,)))
            elif append_delay_type=='Single' :
                one_hot_delay = np.array([[delay]])
            else :
                one_hot_delay = np.array([[0.]])
            # one_hot_delay = convert_delay_to_onehot(np.array([delay]))
            if i > data_states.shape[1] - 10 :
                print_pred_delay = True
            else :
                print_pred_delay = False
            # # if epoch > 1 :
            #     train_step(data_states[:-1,i,3:],data_actions[i,:,:],augment = False,art_delay=delay,one_hot_delay=one_hot_delay,print_pred_delay=print_pred_delay,train=False)
            # else :
            if MODEL == 'nn' :
                X, Y = acc_X(data_states[:-1,i,3:],data_actions[i,:,:],art_delay=delay,one_hot_delay=one_hot_delay,print_pred_delay=print_pred_delay,train=True)
            elif MODEL == 'nn-lstm' :
                X, Y = acc_X_lstm(data_states[:-1,i,3:],data_actions[i,:,:],art_delay=delay)
            # print(X.shape,Y.shape)
            if MODEL == 'nn-lstm' :
                Xs.append(X[:,:45])
                Ys.append(Y[:,:45])
            else :
                Xs.append(X)
                Ys.append(Y)
        X = torch.cat(Xs,dim=0)
        # print(X.shape,Xs[0].shape)
        Y = torch.cat(Ys,dim=0)
        # print(Y.shape,Ys[0].shape)
        if MODEL == 'nn' :
            train_X(X,Y)
        elif MODEL == 'nn-lstm' :
            train_X_lstm(X,Y)
        
        if (j//batch_size)%(1000//batch_size)==0 :
            for k in range(N_ensembles):
                print("Model",str(k),"Epoch",epoch,"Loss",np.mean(stats['losses'+str(k)][-1000:]))
            # print("Epoch",epoch,"Loss",np.mean(stats['losses'][-1000:]))
            # print("Epoch",epoch,"Losses delay",np.mean(stats['losses_delay'][-data_states.shape[1]//10:]))
    losses.append(np.mean(stats['losses'][-data_states.shape[1]:]))
    if epoch>2 and epoch%3==0 :
        # print(model.fc[0].weight.data)
        for k in range(N_ensembles):
            torch.save(models[k].state_dict(), filename[:-4]+str(k)+'.pth')
        # torch.save(model.state_dict(), filename[:-4]+'.pth')
        torch.save(model_delay.state_dict(), filename[:-4]+'_delay.pth')
np.savetxt(filename, losses)