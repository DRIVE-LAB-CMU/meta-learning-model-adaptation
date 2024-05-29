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
import os
print("DEVICE", jax.devices())

DT = .05
N_ROLLOUTS = 10000
H = 8
SIGMA = 1.0
LF = .16
LR = .15
L = LF+LR
learning_rate = 0.0003
trajectory_type = "counter oval"
# trajectory_type = "berlin_2018"

SPEED = 2.2

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
DELAY = 5
MODEL = 'nn'
AUGMENT = False
use_gt = True
HISTORY = 8
ART_DELAY = 0
MAX_DELAY = 7
new_delay_factor = 0.1
curr_delay = 0.
N_ensembles = 3
append_delay_type = 'OneHot' # 'OneHot' or 'Append'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')
parser.add_argument('--pre', action='store_true', help='Enable pre-training')
# parser.add_argument('--pre', action='store_true', help='Enable pre-training')

args = parser.parse_args()

# model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=random.uniform(0.6,0.75), Sb=random.uniform(-0.1,0.1),Ta=random.uniform(5.,45.), Tb=.0, mu=random.uniform(0.35,0.65),delay=1)
model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=random.uniform(0.36,0.38), Sb=random.uniform(-0.01,0.01),Ta=random.uniform(15.,25.), Tb=.0, mu=random.uniform(0.45,0.55),delay=1)
model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.36, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)#random.randint(1,5))
stats = {'lat_errs': [], 'ws_gt': [], 'ws_': [], 'ws': [], 'losses': [], 'date_time': time.strftime("%m/%d/%Y %H:%M:%S"),'buffer': 100, 'lr': learning_rate, 'online_transition': 300, 'delay': DELAY, 'model': MODEL, 'speed': SPEED, 'total_length': 1000, 'history': HISTORY, 'params': model_params}
for i in range(N_ensembles):
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

print(args.exp_name,exp_name)
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

# Define loss function and optimizer
criterion = nn.MSELoss()
# criterion_delay = nn.CrossEntropyLoss()

optimizers = []
for i in range(N_ensembles):
    optimizer = optim.SGD(models[i].parameters(), lr=learning_rate)
    optimizers.append(optimizer)
optimizer_delay = optim.SGD(model_delay.parameters(), lr=1e-3)
for i in range(N_ensembles):
    models[i] = models[i].double()
model_delay = model_delay.double()
print(model.fc)

if args.pre:
    for i in range(N_ensembles):
        models[i].load_state_dict(torch.load('losses/exp3'+str(i)+'.pth'))
    
    # model_delay.load_state_dict(torch.load('losses/exp1.pth'))
else :
    print("Didn't load pre-traned model")

rollout_fn_torch = rollout_fn_select_torch('nn', models, DT, L, LR)

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
        n_rollouts=N_ROLLOUTS,
        H=H,
        device=DEVICE,
        rollout_fn=rollout_fn_torch,
        a_min = [-1., -1.],
        a_max = [1., 1.],
        a_mag = [1., 1.], # 0.1, 0.35
        a_shift = [0., 0.],
        # a_shift = [.06, 0.],
        delay=DELAY*0,
        len_history=HISTORY,
        rollout_start_fn=fn,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
        alpha=0.02,
    )


dynamics_single = DynamicBicycleModel(model_params_single)
env =OffroadCar({}, dynamics_single)

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

obs = env.reset()

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

def rollout_nn(states,actions,state,one_hot_delay,debug=False) :
    global models
    traj = []
    state = torch.tensor(state).double()
    # states = states[:,3:]
    actions = torch.tensor(actions).double()
    states = torch.tensor(states).double()
    one_hot_delay = one_hot_delay
    for i in range(len(actions)-len(states)+1) :
        # print(states.shape,actions[i:i+len(states)].shape)
        X = build_X(states,actions[i:i+len(states)],one_hot_delay)
        # if i==4 and debug:
        #     print("X for pred: ",X)
        # print(X.shape)
        y = 0.
        for j in range(N_ensembles):
            model = models[j]
            y += model(X).detach()
        y /= N_ensembles
        # print(model(X))
        state[0] += state[3]*torch.cos(state[2])*DT - state[4]*torch.sin(state[2])*DT
        state[1] += state[3]*torch.sin(state[2])*DT + state[4]*torch.cos(state[2])*DT
        state[2] += state[5]*DT
        state[3] += y[0,0]*DT
        state[4] += y[0,1]*DT
        state[5] += y[0,2]*DT
        states[:-1] = states.clone()[1:]
        states[-1,:] = state.clone()[3:]
        # print(state)
        traj.append([float(state[0].cpu()),float(state[1].cpu()),float(state[2].cpu()),float(state[3].cpu()),float(state[4].cpu()),float(state[5].cpu())])
    # print(np.array(traj))
    return np.array(traj)

def train_step(states,cmds,augment=True,art_delay=ART_DELAY,one_hot_delay=None,predict_delay=False,print_pred_delay=False) :
    global models, stats, model_delay, curr_delay
    # print("art_delay: ",art_delay)
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
    X = np.concatenate((X,one_hot_delay.repeat((X.shape[0],1))),axis=1)
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
    
    # outputs_delay = model_delay(X[:,:5*HISTORY])
    # pred_delay = torch.argmax(outputs_delay,dim=1).double()
    # curr_delay += (torch.mean(pred_delay).item().cpu()-curr_delay)*new_delay_factor
    # print("Predicted delay: ", curr_delay)
    
    Y = torch.tensor(Y)
    for i in range(N_ensembles):
        optimizer = optimizers[i]
        model = models[i]
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X)
        # print(outputs.shape,Y.shape)
        # Compute the loss
        loss = criterion(outputs, Y/DT)
        stats['losses'+str(i)].append(loss.item())
        # print("Loss",loss.item())
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    return

class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')
        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        self.path_pub_nn = self.create_publisher(Path, 'path_nn', 1)
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        self.odom_pub_ = self.create_publisher(Odometry, 'odom', 1)
        self.timer_ = self.create_timer(0.05, self.timer_callback)
        self.slow_timer_ = self.create_timer(1.0, self.slow_timer_callback)
        self.throttle_pub_ = self.create_publisher(Float64, 'throttle', 1)
        self.steer_pub_ = self.create_publisher(Float64, 'steer', 1)
        self.trajectory_array_pub_ = self.create_publisher(MarkerArray, 'trajectory_array', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
        self.states = []
        self.cmds = []
        self.i = 0
        
    def timer_callback(self):
        print(self.i)
        global obs, target_list_all, stats
        self.i += 1
        # distance_list = np.linalg.norm(waypoint_list - obs[:2], axis=-1)
        # # import pdb; pdb.set_trace()
        # t_idx = np.argmin(distance_list)
        # t_closed = waypoint_t_list[t_idx]
        # target_pos_list = [reference_traj(0. + t_closed + i*DT*1.) for i in range(H+0+1)]
        # target_pos_tensor = jnp.array(target_pos_list)
        target_pos_tensor = waypoint_generator.generate(jnp.array(obs[:5]))
        target_pos_list = np.array(target_pos_tensor)
        # print("Target pos: ",target_pos_list[:,3])
        dynamics.reset()
        # print(env.obs_state())
        
        # target_list_all += target_pos_list
        # action, mppi_info = mppi(obs, reward_fn(target_pos_tensor))
        # print("obs", env.obs_state())
        t1 = time.time()
        target_pos_tensor_torch = torch.Tensor(target_pos_list).to(DEVICE).squeeze(dim=-1)
        # print(env.obs_state())
        if use_gt :
            one_hot_delay = convert_delay_to_onehot(np.array([DELAY-1]))
        else :
            one_hot_delay = convert_delay_to_onehot(np.array([curr_delay]))
        
        if self.i < stats['online_transition'] :
            action, mppi_info = mppi(env.obs_state(),target_pos_tensor, vis_optim_traj=True, model_params=None)
        elif self.i > stats['total_length'] :
            filename = 'data/'+exp_name + '.pickle'
            with open(filename, 'wb') as file:
                pickle.dump(stats, file)
            exit(0)
        else :
            for model in models :
                model = model.to(DEVICE)
            action, mppi_info = mppi_torch(np.array(env.obs_state()), reward_track_fn(target_pos_tensor_torch, SPEED), vis_optim_traj=True, one_hot_delay=one_hot_delay.to(DEVICE))
            for model in models :
                model = model.to('cpu')
        action = np.array(action)
        mppi_torch.feed_hist(torch.tensor(env.obs_state()).double().to(DEVICE),torch.tensor(action).double().to(DEVICE))
        sampled_traj = np.array(mppi_info['trajectory'][:, :2])
        
        px, py, psi, vx, vy, omega = env.obs_state().tolist()
        lat_err = np.sqrt((target_pos_list[0,0]-px)**2 + (target_pos_list[0,1]-py)**2)
        stats['lat_errs'].append(lat_err)
        self.states.append([vx,vy,omega])
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
        # print("Actions: ", actions)
        traj_pred = np.zeros((H+1, 2))
        actions[:,1] = 1
        if len(self.states) > HISTORY :
            traj_pred = rollout_nn(np.array(self.states[-HISTORY:]),np.concatenate((self.cmds[-HISTORY:-1],actions),axis=0),env.obs_state().tolist(),one_hot_delay,debug=True)
        
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
        
        obs, reward, done, info = env.step(action)
        
        w_pred_ = 0.
        _w_pred = 0.
        if len(self.states) > stats['buffer']  :
            
            self.states = self.states[-stats['buffer']:]
            self.cmds = self.cmds[-stats['buffer']:]
            
            train_step(self.states,self.cmds,augment = AUGMENT,one_hot_delay=one_hot_delay,print_pred_delay=False)
            
            
            acts = np.concatenate((np.array(self.cmds[-HISTORY:]),np.repeat(np.array(self.cmds)[-1:,:],5,axis=0)),axis=0)
            traj = rollout_nn(np.array(self.states[-HISTORY:]),acts,env.obs_state().tolist(),one_hot_delay)
            w_pred_ = float(traj[-1,5])
            
            acts = np.concatenate((np.array(self.cmds[-HISTORY:]),np.repeat(-np.array(self.cmds)[-1:,:],20,axis=0)),axis=0)
            traj = rollout_nn(np.array(self.states[-HISTORY:]),acts,env.obs_state().tolist(),one_hot_delay)
            _w_pred = float(traj[-1,5])

            
        w_pred_ = np.clip(w_pred_,-4.,4.)
        _w_pred = np.clip(_w_pred,-4.,4.)
        stats['ws_'].append(w_pred_)
        stats['ws'].append(_w_pred)
        stats['ws_gt'].append(env.obs_state()[5])
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
        steer.data = float(action[1])
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
        t2 = time.time()
        print("Time taken", t2-t1)
        

        
        
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
