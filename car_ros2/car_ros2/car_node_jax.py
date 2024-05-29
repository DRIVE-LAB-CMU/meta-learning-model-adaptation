import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker


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
from jax import grad, jit, vmap


# from jax.experimental.optimizers import optimizer
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
learning_rate = 0.01
trajectory_type = "counter oval"

SPEED = 2.2

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
DELAY = 1
MODEL = 'nn'
AUGMENT = False
HISTORY = 1
ART_DELAY = 0
model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=0.72, Sb=0.0,Ta=20., Tb=.0, mu=0.5,delay=1)
model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.36, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)
stats = {'lat_errs': [], 'ws_gt': [], 'ws_': [], 'ws': [], 'losses': [], 'date_time': time.strftime("%m/%d/%Y %H:%M:%S"),'buffer': 100, 'lr': learning_rate, 'online_transition': 500, 'delay': DELAY, 'model': MODEL, 'speed': SPEED, 'total_length': 1000, 'history': HISTORY}
exp_name = MODEL + str(SPEED)
if AUGMENT :
    exp_name += '_aug'
exp_name += time.strftime("_%m_%d_%Y_%H_%M_%S")
exp_name = 'without_lstm_pre'
dynamics = DynamicBicycleModel(model_params)

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


@jax.jit
def forward(params, x):
    # print(params)
    (w1, b1), (w2, b2) = params
    hidden = jnp.dot(x, w1) + b1
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, w2) + b2
    return output

class SimpleModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = self.init_params(input_size, hidden_size, output_size)
        # print(self.params)

    

    def init_params(self,input_size, hidden_size, output_size):
        keys = jax.random.split(jax.random.key(0), 4)
        w1 = jax.random.uniform(keys[0],(input_size, hidden_size),minval=-1.,maxval=1.)*jnp.sqrt(2/(input_size+hidden_size))
        b1 = jnp.zeros(hidden_size)
        w2 = jax.random.uniform(keys[1],(hidden_size, output_size),minval=-1.,maxval=1.)*jnp.sqrt(2/(output_size+hidden_size))
        b2 = jnp.zeros(output_size)
        return [(w1, b1), (w2, b2)]

@jax.jit
def update(params, inputs, targets,lr=0.1):
    grads = jax.grad(loss)(params, inputs, targets)
    return [(w - lr * dw, b - lr * db)
            for (w, b), (dw, db) in zip(params, grads)]
    
@jax.jit
def loss(params, inputs, targets):
    preds = forward(params, inputs)
    return jnp.mean((preds - targets) ** 2)

def build_X(states, actions):
    X_ = jnp.concatenate((states, actions), axis=1)
    X = X_.reshape(-1)
    return jnp.expand_dims(X, axis=0).astype(jnp.float64)

# Convert the PyTorch function to a JAX one
@jax.jit
def rollout_nn(states, actions, state, params):
    global model
    traj = []
    t1 = time.time()
    state = jnp.array(state)
    # print(state)
    states = jnp.array(states)
    actions = jnp.array(actions)
    t2 = time.time()
    # print("Time taken for conversion", t2-t1)
    for i in range(len(actions) - len(states) + 1):
        t1 = time.time()
        X = build_X(states, actions[i:i + len(states)])
        y = forward(params, X)
        t2 = time.time()
        # print("Time taken for one forward", t2-t1)
        t1 = time.time()
        state = state.at[3].add(y[0, 0] * DT)
        state = state.at[4].add(y[0, 1] * DT)
        state = state.at[5].add(y[0, 2] * DT)
        t2 = time.time()
        # print("Time taken for one step", t2-t1)
        t1 = time.time()
        state = state.at[0].add(state[3] * jnp.cos(state[2]) * DT - state[4] * jnp.sin(state[2]) * DT)
        state = state.at[1].add(state[3] * jnp.sin(state[2]) * DT + state[4] * jnp.cos(state[2]) * DT)
        state = state.at[2].add(state[5] * DT)
        t2 = time.time()
        # print("Time taken for one step", t2-t1)
        states = jnp.roll(states, -1, axis=0)
        states = states.at[-1].set(state[3:])
        traj.append([state[0], state[1], state[2], state[3], state[4], state[5]])
        
    return jnp.array(traj)

def train_step(states, cmds, augment=True, art_delay=ART_DELAY):
    global model, stats
    if art_delay > 0:
        states = states[:-art_delay]
        cmds = cmds[art_delay:]
    elif art_delay < 0:
        cmds = cmds[:art_delay]
        states = states[-art_delay:]

    X_ = jnp.concatenate((jnp.array(states), jnp.array(cmds)), axis=1)[:-1]
    Y_ = jnp.array(states)[1:] - jnp.array(states)[:-1]

    X = [X_[i:-HISTORY+i+1] for i in range(HISTORY-1)]
    X.append(X_[HISTORY-1:])
    X = jnp.concatenate(X, axis=1)
    Y = Y_[HISTORY-1:]
    
    # Augmentation
    if augment:
        X_ = X.copy()
        X_ = jax.ops.index_update(X_, jax.ops.index[:, 1], -X[:, 1])
        X_ = jax.ops.index_update(X_, jax.ops.index[:, 2], -X[:, 2])
        X_ = jax.ops.index_update(X_, jax.ops.index[:, 4], -X[:, 4])
        Y_ = Y.copy()
        Y_ = jax.ops.index_update(Y_, jax.ops.index[:, 1], -Y[:, 1])
        Y_ = jax.ops.index_update(Y_, jax.ops.index[:, 2], -Y[:, 2])

        X = jnp.concatenate((X, X_), axis=0)
        Y = jnp.concatenate((Y, Y_), axis=0)

        X_ = X.copy()
        X_ = jax.ops.index_update(X_, jax.ops.index[:, 1], 0.)
        X_ = jax.ops.index_update(X_, jax.ops.index[:, 2], 0.)
        X_ = jax.ops.index_update(X_, jax.ops.index[:, 4], 0.)
        Y_ = Y.copy()
        Y_ = jax.ops.index_update(Y_, jax.ops.index[:, 1], 0.)
        Y_ = jax.ops.index_update(Y_, jax.ops.index[:, 2], 0.)

        X = jnp.concatenate((X, X_), axis=0)
        Y = jnp.concatenate((Y, Y_), axis=0)

    X = X.astype(jnp.float64)
    Y = Y.astype(jnp.float64)

    

    # Forward pass
    outputs = forward(model.params, X) 

    model.params = update(model.params, X, Y / DT, lr=learning_rate)
    # Compute the loss
    loss = jnp.mean((outputs - Y / DT) ** 2)
    stats['losses'].append(loss)
    # print(loss)
    

    return

if MODEL == 'nn' :
    model = SimpleModel(5*HISTORY,300,3)
elif MODEL == 'nn-lstm' :
    model = LSTMModel(5,10,3)
# Define loss function and optimizer

# model.load_state_dict(torch.load('model.pth'))
rollout_fn_torch = rollout_fn_select_torch('nn', model, DT, L, LR)

def fn():
    return
 




dynamics_single = DynamicBicycleModel(model_params_single)
env =OffroadCar({}, dynamics_single)

rollout_fn = rollout_fn_select('dbm', dynamics, DT, L, LR)
rollout_fn_nn = rollout_fn_select('nn', dynamics, DT, L, LR)

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
        len_history=HISTORY,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
        smooth_alpha=1.,
)

mppi = MPPIController(
    mppi_params, rollout_fn, fn, key, nn_model=model, rollout_fn_nn=rollout_fn_nn
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
        
        dynamics.reset()
        # print(env.obs_state())
        
        # target_list_all += target_pos_list
        # action, mppi_info = mppi(obs, reward_fn(target_pos_tensor))
        # print("obs", env.obs_state())
        t1 = time.time()
        target_pos_tensor_torch = torch.Tensor(target_pos_list).to(DEVICE).squeeze(dim=-1)
        # print(env.obs_state())
        
        if self.i > stats['online_transition'] :
            print("Transitioned?")
            use_nn = True
        else :
            use_nn = False
            # mppi.rollout_fn = rollout_fn_select('nn', dynamics, DT, L, LR)
        print(use_nn)
        action, mppi_info = mppi(env.obs_state(), target_pos_tensor, model.params, vis_optim_traj=True, use_nn=use_nn)
        # else :
        #     print("Hehh?", self.i)
        # if self.i < stats['online_transition']//3 :
        #     action, mppi_info = mppi(env.obs_state(),target_pos_tensor, vis_optim_traj=True)
        # else :#if self.i > stats['total_length'] :
        #     filename = 'data/'+exp_name + '.pickle'
        #     with open(filename, 'wb') as file:
        #         pickle.dump(stats, file)
        #     exit(0)
        # else :
        #     action, mppi_info = mppi_torch(np.array(env.obs_state()), reward_track_fn(target_pos_tensor_torch, SPEED), vis_optim_traj=True)
        action = np.array(action)
        mppi.feed_hist(env.obs_state(),action)
        # print(np.array(mppi.state_hist))
            # print(mppi_info['action'])
        t2 = time.time()
        print("Time taken", t2-t1)
        # action[1] = action[1] * 1.5
        sampled_traj = np.array(mppi_info['trajectory'][:, :2])
        # sampled_traj = np.zeros((H+1, 2))
        # print(action)
        # import pdb; pdb.set_trace()
        # all_trajectory = np.array(mppi_info['all_traj'])[:, :, :2]
        # all_trajectory = np.zeros((H+1, 1000, 2))
        # plt.figure()
        # for i in range(10):
        #     plt.plot(all_trajectory[:, i, 0], all_trajectory[:, i, 1])
        # plt.savefig('all_trajectory.png')
        # print(all_trajectory.shape)
        # action = np.zeros(2)
        # action *= 0.
        # action = np.array([.1, 1.])
        px, py, psi, vx, vy, omega = env.obs_state().tolist()
        lat_err = np.sqrt((target_pos_list[0,0]-px)**2 + (target_pos_list[0,1]-py)**2)
        stats['lat_errs'].append(lat_err)
        self.states.append([vx,vy,omega])
        # print(action*1000)
        obs, reward, done, info = env.step(action)
        
        
        
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
        
        w_pred_ = 0.
        _w_pred = 0.
        self.cmds.append([action[0], action[1]])
        if len(self.states) > stats['buffer']  :
            self.states = self.states[-stats['buffer']:]
            self.cmds = self.cmds[-stats['buffer']:]
            t1 = time.time()
            train_step(self.states,self.cmds,augment = AUGMENT)
            t2 = time.time()
            print("Time taken for training", t2-t1)
            
            t1 = time.time()
            acts = np.concatenate((np.array(self.cmds[-HISTORY:]),np.repeat(np.array(self.cmds)[-1:,:],5,axis=0)),axis=0)
            traj = rollout_nn(np.array(self.states[-HISTORY:]),acts,env.obs_state().tolist(),model.params)
            # X = np.concatenate((np.array(self.states[-1:]),np.array(self.cmds[-1:])),axis=1)
            # X = torch.tensor(X).double()
            # # X[:,-1] *= 1.5
            # for j in range(5) :
            #     X[:,:3] += model(X)*DT
            # print(traj[:,5])
            w_pred_ = float(traj[-1,5])
            t2 = time.time()
            print("Time taken for prediction", t2-t1)
            acts = np.concatenate((np.array(self.cmds[-HISTORY:]),np.repeat(-np.array(self.cmds)[-1:,:],20,axis=0)),axis=0)
            traj = rollout_nn(np.array(self.states[-HISTORY:]),acts,env.obs_state().tolist(),model.params)
            
            # X = np.concatenate((np.array(self.states[-1:]),np.array(self.cmds[-1:])),axis=1)
            # X = torch.tensor(X).double()
            # X[:,-1] *= -1.
            # print(X)
            # X[:,1] *= -1.
            # X[:,2] *= -1.
            # for j in range(20) :
            #     # print(j,X[:,2])
            #     X[:,:3] += model(X)*DT
            # print("haha: ", traj[:,5])
            
            _w_pred = float(traj[-1,5])

            # X = np.concatenate((np.array(self.states[-10:]),np.array(self.cmds[-10:])),axis=1)
            # X[:,0] = X[-1,0]
            # X[:,1] = X[-1,1]
            # X[:,2] = X[-1,2]
            # X[:,3] = X[-1,3]
            # X[:,-1] = np.arange(-1.,0.99,1./5.)
            # X = torch.tensor(X).double()
            # for j in range(5) :
            #     X[:,:3] += model(X)*DT
            # print(X[:,2])
        
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
        actions = np.array(mppi_info['action'])
        traj_pred = np.zeros((H+1, 2))
        if len(self.states) > HISTORY :
            traj_pred = rollout_nn(np.array(self.states[-HISTORY:]),np.concatenate((self.cmds[-HISTORY:],actions),axis=0),env.obs_state().tolist(),model.params)
        
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
