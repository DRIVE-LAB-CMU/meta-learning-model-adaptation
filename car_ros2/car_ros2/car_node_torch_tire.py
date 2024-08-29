import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, TwistStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
import random
import yaml

import numpy as np
import matplotlib.pyplot as plt

from tf_transformations import quaternion_from_euler, euler_matrix

from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_torch import rollout_fn_select as rollout_fn_select_torch
from car_dynamics.controllers_torch import reward_track_fn, MPPIController as MPPIControllerTorch
from car_dynamics.controllers_jax import MPPIController, MPPIParams, rollout_fn_select
from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.controllers_jax import WaypointGenerator
from std_msgs.msg import Float64, Int8
import torch.nn as nn
import torch.optim as optim
import torch
import time
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
import pickle
import os
from ackermann_msgs.msg import AckermannDrive
import tf_transformations
import socket
import struct
import threading
print("DEVICE", jax.devices())

DT = 0.08
DT_torch = 0.08
DELAY = 3
N_ROLLOUTS = 10000
H = 8
SIGMA = 1.0
i_start = 30

####### VICON Callback ####################
vicon_loc = None
def update_vicon():
    global vicon_loc
    server_ip = "0.0.0.0"
    server_port = 12346
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((server_ip, server_port))

    print(f"UDP server listening on {server_ip}:{server_port}")

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
    t = time.time()
    while True :
        data, addr = server_socket.recvfrom(48)  # 3 doubles * 8 bytes each = 24 bytes
        unpacked_data = struct.unpack('dddddd', data)
        if np.isnan(unpacked_data[0]) or np.isnan(unpacked_data[1]) or np.isnan(unpacked_data[2]) or np.isnan(unpacked_data[3]) or np.isnan(unpacked_data[4]) or np.isnan(unpacked_data[5]):
            print("NAN received")
            exit(0)
            continue
        # print(f"Received pose from {addr}: {unpacked_data}", time.time()-t)
        t = time.time()
        vicon_loc = [unpacked_data[0],unpacked_data[1],unpacked_data[2],unpacked_data[3],unpacked_data[4],unpacked_data[5]]
        # time.sleep(0.005)

trajectory_type = "counter oval"
# trajectory_type = "berlin_2018"
SIM = 'numerical' # 'numerical' or 'unity' or 'vicon'

if SIM == 'vicon' :
    learning_rate = 0.04
    vicon_thread = threading.Thread(target=update_vicon)
    vicon_thread.start()
    LF = 0.16
    LR = 0.15
    L = LF+LR
    client_ip = "rcar.wifi.local.cmu.edu"
    client_port = 12347
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    

if SIM == 'numerical' :
    learning_rate = 0.04
    LF = 0.16
    LR = 0.15
    L = LF+LR

if SIM=='unity' :
    learning_rate = 0.01
    trajectory_type = "../../simulators/params.yaml"
    LF = 1.5
    LR = 1.6
    L = LF+LR

if SIM == 'unity' :
    SPEED = 10.0
else :
    SPEED = 1.95

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
    
AUGMENT = True
use_gt = True
HISTORY = 8
ART_DELAY = 0
MAX_DELAY = 7
new_delay_factor = 0.1
curr_delay = 0.
N_ensembles = 1
append_delay_type = 'OneHot' # 'OneHot' or 'Append'
LAST_LAYER_ADAPTATION = False
mass = 4.65
I = 0.1
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')
parser.add_argument('--pre', action='store_true', help='Enable pre-training')
parser.add_argument('--lstm', action='store_true', help='Enable lstm')

args = parser.parse_args()
if args.lstm: 
    MODEL = 'nn-lstm' # 'nn' or 'nn-lstm
else :
    MODEL = 'nn-tires' # 'nn' or 'nn-lstm

# model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=random.uniform(0.6,0.75), Sb=random.uniform(-0.1,0.1),Ta=random.uniform(5.,45.), Tb=.0, mu=random.uniform(0.35,0.65),delay=1)
model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=random.uniform(0.39,0.46), Sb=0.1,Ta=random.uniform(10.,15.), Tb=.0, mu=random.uniform(0.45,0.55),delay=1)
# model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5)#random.randint(1,5))

if SIM == 'unity' :
    yaml_contents = yaml.load(open(trajectory_type, 'r'), Loader=yaml.FullLoader)
    
    model_params.Bf = yaml_contents['vehicle_params']['Bf']
    model_params.Br = yaml_contents['vehicle_params']['Br']
    model_params.Cf = yaml_contents['vehicle_params']['Cf']
    model_params.Cr = yaml_contents['vehicle_params']['Cr']
    model_params.Iz = yaml_contents['vehicle_params']['I']
    model_params.LF = yaml_contents['vehicle_params']['Lf']
    model_params.LR = yaml_contents['vehicle_params']['Lr']
    model_params.MASS = yaml_contents['vehicle_params']['m']
    model_params.Ta = yaml_contents['vehicle_params']['K_cmd']
    model_params.Tb = yaml_contents['vehicle_params']['K_v']
    model_params.Sa = yaml_contents['vehicle_params']['max_steer']
    model_params.Sb = 0.
    model_params.mu = yaml_contents['vehicle_params']['mu_f']
    decay_start = yaml_contents['vehicle_params']['friction_decay_start']
    decay_rate = yaml_contents['vehicle_params']['friction_decay_rate']


model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY,LF=LF,LR=LR)#random.randint(1,5))
stats = {'lat_errs': [], 'ws_gt': [], 'ws_': [], 'ws': [], 'losses': [], 'date_time': time.strftime("%m/%d/%Y %H:%M:%S"),'buffer': 100, 'lr': learning_rate, 'online_transition': 300, 'delay': DELAY, 'model': MODEL, 'speed': SPEED, 'total_length': 2000, 'history': HISTORY, 'params': model_params, 'traj': [], 'ref_traj': []}
for i in range(3):
    stats['losses'+str(i)] = []
    
def convert_delay_to_onehot(delay, max_delay=MAX_DELAY):
    onehot = np.zeros((delay.shape[0],max_delay))
    inds = np.arange(delay.shape[0])
    frac = delay - delay.astype(int)
    onehot[inds,delay.astype(int)] = 1 - frac
    onehot[inds,(delay.astype(int)+1)%max_delay] = frac
    return torch.tensor(onehot)


if args.exp_name == 'none' :
    exp_name = MODEL + str(SPEED)
    if AUGMENT :
        exp_name += '_aug'
    exp_name += time.strftime("_%m_%d_%Y_%H_%M_%S")
else :
    exp_name = args.exp_name

# print(args.exp_name,exp_name)
dynamics = DynamicBicycleModel(model_params)

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
            h_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device).double()
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device).double()
        
        inputs = X.view(X.shape[0], -1, 3+2)
        # print(inputs.shape)
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(inputs, (h_0, c_0))
        
        
        # Fully connected layer
        output = self.fc(lstm_out)
        
        return output, (hn, cn)

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


models = []
if MODEL == 'nn-tires' :
    # models.append(SimpleModel(3,[40],1))
    models.append(SimpleModel(3,[],1))
    models[0].fc[-1].weight.data.fill_(0.)
    models[0].fc[-1].bias.data.fill_(0.)
    for i in range(2):
        model = SimpleModel(1,[40],1).double()
        models.append(model)
        
# Define loss function and optimizer
criterion = nn.MSELoss()
# criterion_delay = nn.CrossEntropyLoss()

optimizers = []
for i in range(3):
    if LAST_LAYER_ADAPTATION:
        optimizer = optim.SGD(models[i].fc[-1].parameters(), lr=learning_rate)
    else :
        if i==0 :
            optimizer = optim.SGD(models[i].parameters(), lr=0.001)
        else :
            optimizer = optim.SGD(models[i].parameters(), lr=learning_rate)
    optimizers.append(optimizer)
    models[i] = models[i].double()


if args.pre:
    post = ""
    if 'lstm' in MODEL :
        post = '_lstm'
    for i in range(N_ensembles):
        print('losses/none'+post+str(i)+'_.pth')
        models[i].load_state_dict(torch.load('losses/none'+post+str(i)+'.pth'))
    
    # model_delay.load_state_dict(torch.load('losses/exp1.pth'))
else :
    print("Didn't load pre-traned model")

rollout_fn_torch = rollout_fn_select_torch(MODEL, models, DT_torch, L, LR,mass/I)

def fn():
    return
 
mppi_torch = MPPIControllerTorch(
        gamma_mean=1.0,
        gamma_sigma=0.0,
        discount=1.,
        sample_sigma = 0.5,
        lam = 0.01,
        a_mean=torch.zeros(H, 2, device=DEVICE),
        a_cov = a_cov_init.to(DEVICE),
        n_rollouts=N_ROLLOUTS//30,
        H=H,
        device=DEVICE,
        rollout_fn=rollout_fn_torch,
        a_min = [-1., -1.],
        a_max = [1., 1.],
        a_mag = [1., 1.], # 0.1, 0.35
        a_shift = [0., 0.],
        # a_shift = [.06, 0.],
        delay=DELAY-1,
        len_history=HISTORY,
        rollout_start_fn=fn,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
        alpha=0.02,
        lstm=args.lstm
    )


dynamics_single = DynamicBicycleModel(model_params_single)

rollout_fn = rollout_fn_select('dbm', dynamics, DT, L, LR)

dynamics.reset()
dynamics_single.reset()

def fn():
    ...

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

mppi = MPPIController(
    mppi_params, rollout_fn, fn, key, nn_model=None
)

waypoint_generator = WaypointGenerator(trajectory_type, DT, H, SPEED)
done = False
frames = []

if SIM == 'numerical' :
    env = OffroadCar({}, dynamics_single)
    obs = env.reset(pose=(-1.,0.,-np.pi/2.))
elif SIM == 'unity' :
    obs = np.array([yaml_contents['respawn_loc']['z'], yaml_contents['respawn_loc']['x'],0.,0.,0.,0.,])
elif SIM == 'vicon' :
    obs = np.array([0.,0.,0.,0.,0.,0.])
    pose_x = 0.
    pose_y = 0.
    pose_yaw = 0.
    t_prev = time.time()

goal_list = []
target_list = []
action_list = []
mppi_action_list = []
obs_list = []


pos2d = []
target_list_all = []

def build_X(states,actions,one_hot_delay): 
    X_ = torch.cat((states,actions),dim=1)
    X = X_.reshape(-1)
    X = torch.tensor(X).unsqueeze(0).double()
    X = torch.cat((X,one_hot_delay),dim=1)
    return X

def rollout_nn_tires(states,actions,state,one_hot_delay,debug=False) :
    global models
    traj = []
    state = torch.tensor(state).double()
    actions = torch.tensor(actions).double()
    states = torch.tensor(states).double()
    for i in range(len(actions)-len(states)+1) :
        vx = torch.maximum(state[3],torch.tensor(1.))
        vy = state[4]
        w = state[5]
        
        action = actions[len(states)+i-1]
        steer = action[1]*0.34
        alpha_f = steer - torch.arctan2(vy+w*LF,vx)
        alpha_r = torch.arctan2(-vy+w*LR,vx)
        m_I = mass/I
    
        X1 = torch.tensor([[vx,vx**2,action[0]]]).double()
        Frx = models[0](X1).detach()[0,0]
        
        X2 = torch.tensor([[alpha_r]]).double()
        Fry = models[1](X2).detach()[0,0]
        
        X3 = torch.tensor([[alpha_f]]).double()
        Ffy = models[2](X3).detach()[0,0]
        
        A = torch.tensor([[1.,0.,-torch.sin(steer)],[0.,1.,torch.cos(steer)],[0.,-m_I*LR,m_I*LF*torch.cos(steer)]]).double()
        B = torch.tensor([[vy*w,-vx*w,0.]]).double().transpose(0,1)
        X = torch.tensor([[Frx,Fry,Ffy]]).double().transpose(0,1)
        y = torch.matmul(A,X) + B
        state[0] += state[3]*torch.cos(state[2])*DT - state[4]*torch.sin(state[2])*DT
        state[1] += state[3]*torch.sin(state[2])*DT + state[4]*torch.cos(state[2])*DT
        state[2] += state[5]*DT
        state[3] += y[0,0]*DT
        state[4] += y[1,0]*DT
        state[5] += y[2,0]*DT
        traj.append([float(state[0].cpu()),float(state[1].cpu()),float(state[2].cpu()),float(state[3].cpu()),float(state[4].cpu()),float(state[5].cpu())])

    return np.array(traj)

def train_step_tires(states,cmds,augment=True,art_delay=ART_DELAY,one_hot_delay=None,predict_delay=False,print_pred_delay=False) :
    global models, stats, model_delay, curr_delay
    if art_delay > 0 :
        states = states[:-art_delay]
        cmds = cmds[art_delay:]
    elif art_delay < 0 :
        cmds = cmds[:art_delay]
        states = states[-art_delay:]
    
    
    vxs = np.maximum(np.array(states)[:-1,0],np.array(states)[:-1,0]*0.+1.)
    
    vys = np.array(states)[:-1,1]
    ws = np.array(states)[:-1,2]
    throttles = np.array(cmds)[:-1,0]
    steers = np.array(cmds)[:-1,1]*0.34
    vx_dots = (np.array(states)[1:,0] - np.array(states)[:-1,0])/DT
    vy_dots = (np.array(states)[1:,1] - np.array(states)[:-1,1])/DT
    w_dots = (np.array(states)[1:,2] - np.array(states)[:-1,2])/DT
    alpha_fs = steers - np.arctan2(vys+ws*LF,vxs)
    alpha_rs = np.arctan2(-vys+ws*LR,vxs)
    m_I = mass/I
    ones = np.ones_like(vxs)
    A = np.array([[ones,0.*ones,-np.sin(steers)],[0.*ones,1.*ones,np.cos(steers)],[0.*ones,-m_I*LR*ones,m_I*LF*np.cos(steers)]])
    B = np.array([[-vys*ws+vx_dots,vxs*ws+vy_dots,w_dots]]).transpose((2,1,0))
    A_inv = np.linalg.inv(A.transpose((2,0,1)))
    soln = np.matmul(A_inv,B)
    Frx = soln[:,0,:]
    Fry = soln[:,1,:]
    Ffy = soln[:,2,:]
    
    X1 = np.stack((vxs,vxs**2,throttles),axis=1)
    # if augment :
    #     X1_ = np.stack((vxs,vxs**2,-throttles),axis=1)
    #     X1 = np.concatenate((X1,X1_),axis=0)
    #     Frx = np.concatenate((Frx,-Frx),axis=0)
    Frx_pred = models[0](torch.tensor(X1))
    optimizers[0].zero_grad()
    # print(Frx_pred.shape, Frx.shape)
    print(X1,Frx_pred,Frx)
    loss = criterion(Frx_pred, torch.tensor(Frx))
    stats['losses'+str(0)].append(loss.item())
    X_trial = np.zeros((100,3))
    X_trial[:,2] = np.linspace(-1.,1.,100)
    X_trial[:,0] = vxs[-1]
    X_trial[:,1] = vxs[-1]**2
    F_pred = models[0](torch.tensor(X_trial))
    print("Fx pred: ",models[0].fc[-1].weight)
    loss.backward()
    optimizers[0].step()

    X2 = alpha_rs.reshape(-1,1)
    if augment :
        X2 = np.concatenate((X2,-X2),axis=0)
        Fry = np.concatenate((Fry,-Fry),axis=0)
    Fry_pred = models[1](torch.tensor(X2))
    optimizers[1].zero_grad()
    X_trial = np.linspace(-0.5,0.5,2).reshape(-1,1)
    F_pred = models[1](torch.tensor(X_trial))
    print("rear", F_pred)
    
    loss = criterion(Fry_pred, torch.tensor(Fry))
    stats['losses'+str(1)].append(loss.item())
    loss.backward()
    optimizers[1].step()

    X3 = alpha_fs.reshape(-1,1)
    if augment :
        X3 = np.concatenate((X3,-X3),axis=0)
        Ffy = np.concatenate((Ffy,-Ffy),axis=0)
    Ffy_pred = models[2](torch.tensor(X3))
    X_trial = np.linspace(-0.5,0.5,2).reshape(-1,1)
    F_pred = models[2](torch.tensor(X_trial))
    print("front:", F_pred)
    
    optimizers[2].zero_grad()
    loss = criterion(Ffy_pred, torch.tensor(Ffy))
    stats['losses'+str(2)].append(loss.item())
    loss.backward()
    optimizers[2].step()
    
    return


curr_steer = 0.
class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')
        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        self.path_pub_nn = self.create_publisher(Path, 'path_nn', 1)
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        self.odom_pub_ = self.create_publisher(Odometry, 'odom', 1)
        self.timer_ = self.create_timer(DT, self.timer_callback)
        self.slow_timer_ = self.create_timer(2.0, self.slow_timer_callback)
        self.throttle_pub_ = self.create_publisher(Float64, 'throttle', 1)
        self.steer_pub_ = self.create_publisher(Float64, 'steer', 1)
        self.trajectory_array_pub_ = self.create_publisher(MarkerArray, 'trajectory_array', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
        self.status_pub_ = self.create_publisher(Int8, 'status', 1)
        if SIM == 'unity' :
            self.unity_publisher_ = self.create_publisher(AckermannDrive, '/cmd', 10)
            self.ackermann_msg = AckermannDrive()
            self.unity_subscriber_ = self.create_subscription(PoseStamped, 'car_pose', self.unity_callback, 10)
            self.unity_subscriber_twist_ = self.create_subscription(TwistStamped, 'car_twist', self.unity_twist_callback, 10)
            self.unity_state = [yaml_contents['respawn_loc']['z'], yaml_contents['respawn_loc']['x'],0.,0.,0.,0.]
            self.pose_received = True
            self.vel_received = True
            self.mu_factor_pub_ = self.create_publisher(Float64, 'mu_factor', 1)
            
        
        self.states = []
        self.cmds = []
        self.i = 0
        self.curr_t_counter = 0.
        self.vicon_loc = np.array(vicon_loc)
        self.unity_state_new = [0.,0.,0.,0.,0.,0.]
    
    def unity_callback(self, msg):
        px = msg.pose.position.z
        py = msg.pose.position.x
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        R = tf_transformations.quaternion_matrix(q)[:3, :3]
        t = np.array([0.0, 0.0, 1.0])
        Rt = np.dot(R, t)
        # print(Rt)
        # np.arctan2(Rt[0], Rt[2])
        psi = np.arctan2(Rt[0], Rt[2])
        
        # print("psi: ", psi)
        self.unity_state[0] = px
        self.unity_state[1] = py
        self.unity_state[2] = psi
        self.pose_received = True
    
    def unity_twist_callback(self, msg):
        vx = msg.twist.linear.z
        vy = msg.twist.linear.x
        omega = msg.twist.angular.y
        # print(vx,vy,omega)
        self.unity_state[3] = vx
        self.unity_state[4] = vy
        self.unity_state[5] = omega
        self.vel_received = True
    
    def obs_state(self):
        if SIM == 'unity' :
            if self.pose_received and self.vel_received :
                return np.array(self.unity_state_new)
            else :
                return np.array([yaml_contents['respawn_loc']['z'], yaml_contents['respawn_loc']['x'], 0., 0., 0., 0.])
        elif SIM == 'numerical':
            return env.obs_state()
        else :
            return self.vicon_loc
    
    def timer_callback(self):
        ti = time.time()
        print(self.i)
        global obs, target_list_all, stats, h_0s, c_0s, action, curr_steer
        if SIM == 'unity' and not self.pose_received :
            return
        if SIM == 'unity' and not self.vel_received :
            return
        
        self.i += 1
        mu_factor = 1.
        if SIM == 'unity' :
            if self.i*DT > decay_start :
                mu_factor = 1. - (self.i*DT - decay_start)*decay_rate
            mu_msg = Float64()
            mu_msg.data = mu_factor
            self.mu_factor_pub_.publish(mu_msg)
        # distance_list = np.linalg.norm(waypoint_list - obs[:2], axis=-1)
        # # import pdb; pdb.set_trace()
        # t_idx = np.argmin(distance_list)
        # t_closed = waypoint_t_list[t_idx]
        # target_pos_list = [reference_traj(0. + t_closed + i*DT*1.) for i in range(H+0+1)]
        # target_pos_tensor = jnp.array(target_pos_list)
        if SIM == 'unity':
            if self.i==1 :
                action = np.array([0.,0.])
                action[0] = -3.
                action[1] = -3.
            self.unity_state_new = self.unity_state.copy()
            obs = np.array(self.unity_state)
            self.ackermann_msg.acceleration = float(action[0])
            self.ackermann_msg.steering_angle = float(action[1])
            self.unity_publisher_.publish(self.ackermann_msg)
        ta = time.time()
        status = Int8()
        if self.i < stats['online_transition'] :
            target_pos_tensor, _ = waypoint_generator.generate(jnp.array(obs[:5]),mu_factor=mu_factor,dt=DT_torch)
        else :
            target_pos_tensor, _ = waypoint_generator.generate(jnp.array(obs[:5]),dt=DT_torch,mu_factor=mu_factor)
        # print("h: ", target_pos_tensor)
        stats['traj'].append([self.obs_state()])
        target_pos_list = np.array(target_pos_tensor)
        # print(target_pos_list.shape)
        # print("Target pos: ",target_pos_list[:,3])
        dynamics.reset()
        tb = time.time()
        # print("Waypoint generation time: ", tb-ta)
        # print(self.obs_state())
        
        # target_list_all += target_pos_list
        # action, mppi_info = mppi(obs, reward_fn(target_pos_tensor))
        # print("obs", self.obs_state())
        target_pos_tensor_torch = torch.Tensor(target_pos_list).to(DEVICE).squeeze(dim=-1)
        # print(self.obs_state())
        if use_gt :
            one_hot_delay = convert_delay_to_onehot(np.array([(DELAY-1)*0]))
        else :
            one_hot_delay = convert_delay_to_onehot(np.array([curr_delay]))
        # print(self.obs_state())
        if self.i < stats['online_transition'] and self.i > 3:
            status.data = 1
            print("State: ", self.obs_state())
            action, mppi_info = mppi(self.obs_state(),target_pos_tensor, vis_optim_traj=True, model_params=None)
        elif self.i > stats['total_length'] :
            filename = 'data/'+exp_name + '.pickle'
            with open(filename, 'wb') as file:
                stats['ref_traj'] = np.array(waypoint_generator.waypoint_list_np)
                stats['traj'] = np.array(stats['traj'])
                print(stats['traj'].shape)
                pickle.dump(stats, file)
            exit(0)
        else :
            status.data = 2
            t1 = time.time()
            for model in models :
                model = model.to(DEVICE)
            t2 = time.time()
            if not args.lstm :
                print("State: ", self.obs_state(), target_pos_tensor_torch.shape)
                action, mppi_info = mppi_torch(np.array(self.obs_state()), reward_track_fn(target_pos_tensor_torch, SPEED, SIM), vis_optim_traj=True, one_hot_delay=one_hot_delay.to(DEVICE))
            else :
                action, mppi_info = mppi_torch(np.array(self.obs_state()), reward_track_fn(target_pos_tensor_torch, SPEED, SIM), vis_optim_traj=True, one_hot_delay=one_hot_delay.to(DEVICE), h_0s=h_0s, c_0s=c_0s)
            t3 = time.time()
            for model in models :
                model = model.to('cpu')
            t4 = time.time()
            # print("Model to gpu time: ", t2-t1)
            print("MPPI time: ", t3-t2)
            # print("Model to cpu time: ", t4-t3)
        action = np.array(action)
        if self.curr_t_counter + 1e-3 > DT_torch :
            self.curr_t_counter = 0.
            mppi_torch.feed_hist(torch.tensor(self.obs_state()).double().to(DEVICE),torch.tensor(action).double().to(DEVICE))
        self.curr_t_counter += DT
        sampled_traj = np.array(mppi_info['trajectory'][:, :2])
        
        px, py, psi, vx, vy, omega = self.obs_state().tolist()
        lat_err = np.sqrt((target_pos_list[0,0]-px)**2 + (target_pos_list[0,1]-py)**2)
        stats['lat_errs'].append(lat_err)
        if self.i > i_start :
            if np.isnan(vx) or np.isnan(vy) or np.isnan(omega) :
                print("State received a nan value")
                exit(0) 
            self.states.append([vx,vy,omega])
            if np.isnan(action[0]) or np.isnan(action[1]) :
                print("Action received a nan value")
                exit(0)
            self.cmds.append([action[0], action[1]])
        
        q = quaternion_from_euler(0, 0, psi)
        now = self.get_clock().now().to_msg()
        
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = now
        pose.pose.pose.position.x = px
        pose.pose.pose.position.y = py
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]
        self.pose_pub_.publish(pose)
        
        
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = now
        for i in range(target_pos_list.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(target_pos_list[i][0])
            pose.pose.position.y = float(target_pos_list[i][1])
            path.poses.append(pose)
        self.ref_trajectory_pub_.publish(path)
        
        mppi_path = Path()
        mppi_path.header.frame_id = 'map'
        mppi_path.header.stamp = now
        for i in range(len(sampled_traj)):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(sampled_traj[i, 0])
            pose.pose.position.y = float(sampled_traj[i, 1])
            mppi_path.poses.append(pose)
        self.path_pub_.publish(mppi_path)
        actions = np.array(mppi_info['action'])
        # print("Actions: ", self.cmds[-HISTORY:-1], actions)
        traj_pred = np.zeros((H+1, 2))
        # actions[:,1] = 1
        if len(self.states) > HISTORY :
            traj_pred = rollout_nn_tires(np.array(self.states[-HISTORY:]),np.concatenate((self.cmds[-HISTORY:-1],actions),axis=0),self.obs_state().tolist(),one_hot_delay,debug=True)
        else :
            status.data = 0
        # print(self.cmds[-HISTORY:-1],actions)
        self.status_pub_.publish(status)
        nn_path = Path()
        nn_path.header.frame_id = 'map'
        nn_path.header.stamp = now
        for i in range(len(traj_pred)):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(traj_pred[i, 0])
            pose.pose.position.y = float(traj_pred[i, 1])
            nn_path.poses.append(pose)
        self.path_pub_nn.publish(nn_path)
        
        if self.i < 6 :
            action[0] = 0.
            action[1] = 0.
        
        if SIM == 'numerical':
            obs, reward, done, info = env.step(action)
        elif SIM == 'vicon' :
            obs = np.array(vicon_loc)
            self.vicon_loc = np.array(vicon_loc)
            client_socket.sendto(struct.pack('dd',vx*0.13+action[0]*0.13,action[1]),(client_ip,client_port))
            
        w_pred_ = 0.
        _w_pred = 0.
        print(models[0].fc[-1].weight)
        if len(self.states) > stats['buffer']  :
            
            self.states = self.states[-stats['buffer']:]
            self.cmds = self.cmds[-stats['buffer']:]
            
            if not args.lstm :
                train_step_tires(self.states,self.cmds,augment = AUGMENT,one_hot_delay=one_hot_delay,print_pred_delay=False,art_delay=-(DELAY-1))
            else :
                h_0s, c_0s = train_step(self.states,self.cmds,augment = AUGMENT,one_hot_delay=one_hot_delay,print_pred_delay=False)
                
            
            acts = np.concatenate((np.array(self.cmds[-HISTORY:]),np.repeat(np.array(self.cmds)[-1:,:],5,axis=0)),axis=0)
            traj = rollout_nn_tires(np.array(self.states[-HISTORY:]),acts,self.obs_state().tolist(),one_hot_delay)
            w_pred_ = float(traj[-1,5])
            
            acts = np.concatenate((np.array(self.cmds[-HISTORY:]),np.repeat(-np.array(self.cmds)[-1:,:],20,axis=0)),axis=0)
            traj = rollout_nn_tires(np.array(self.states[-HISTORY:]),acts,self.obs_state().tolist(),one_hot_delay)
            _w_pred = float(traj[-1,5])

            
        w_pred_ = np.clip(w_pred_,-4.,4.)
        _w_pred = np.clip(_w_pred,-4.,4.)
        stats['ws_'].append(w_pred_)
        stats['ws'].append(_w_pred)
        stats['ws_gt'].append(self.obs_state()[5])
        odom = Odometry()
        odom.header.frame_id = 'map'
        odom.header.stamp = now
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = omega
        odom.twist.twist.angular.x = w_pred_
        odom.twist.twist.angular.y = _w_pred
        self.odom_pub_.publish(odom)
        
        # print(np.array(mppi_info['action']).shape)
        
        throttle = Float64()
        throttle.data = float(action[0])
        self.throttle_pub_.publish(throttle)
        steer = Float64()
        curr_steer += 0.1*(float(action[1]) - curr_steer) 
        steer.data = curr_steer #float(action[1])
        self.steer_pub_.publish(steer)
        
        # trajectory array
        # all_trajectory is of shape horizon, num_rollout, 3
        # trajectory_array = MarkerArray()
        # for i in range(all_trajectory.shape[1]):
            # marker = Marker()
            # marker.header.frame_id = 'map'
            # marker.header.stamp = now
            # marker.type = Marker.LINE_STRIP
            # marker.action = Marker.ADD
            # marker.id = i
            # marker.scale.x = 0.05
            # marker.color.a = 1.0
            # marker.color.r = 1.0
            # marker.color.g = 0.0
            # marker.color.b = 0.0
            # for j in range(all_trajectory.shape[0]):
            #     point = all_trajectory[j, i]
            #     p = Point()
            #     p.x = float(point[0])
            #     p.y = float(point[1])
            #     p.z = 0.
            #     marker.points.append(p)
            # trajectory_array.markers.append(marker)
        # self.trajectory_array_pub_.publish(trajectory_array)
        
        # body polygon
        pts = np.array([
            [LF, L/3],
            [LF, -L/3],
            [-LR, -L/3],
            [-LR, L/3],
        ])
        # transform to world frame
        R = euler_matrix(0, 0, psi)[:2, :2]
        pts = np.dot(R, pts.T).T
        pts += np.array([px, py])
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = now
        for i in range(pts.shape[0]):
            p = Point32()
            p.x = float(pts[i, 0])
            p.y = float(pts[i, 1])
            p.z = 0.
            body.polygon.points.append(p)
        self.body_pub_.publish(body)
        tf = time.time()
        print("Time taken", tf-ti)
        if SIM == 'unity' :
            self.pose_received = False
            self.vel_received = False

        
    def slow_timer_callback(self):
        # publish waypoint_list as path
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(waypoint_generator.waypoint_list_np.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.waypoint_list_np[i][0])
            pose.pose.position.y = float(waypoint_generator.waypoint_list_np[i][1])
            path.poses.append(pose)
        self.waypoint_list_pub_.publish(path)

def main():
    rclpy.init()
    car_node = CarNode()
    rclpy.spin(car_node)
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
