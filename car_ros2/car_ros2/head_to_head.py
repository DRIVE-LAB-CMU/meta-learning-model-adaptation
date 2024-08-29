import scipy.interpolate
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
import argparse
import scipy

print("DEVICE", jax.devices())

DT = 0.1
DT_torch = 0.1
DELAY = 2
N_ROLLOUTS = 10000
H = 8
SIGMA = 1.0
i_start = 30
N_lat_divs = 5
dist_long = 20
curv_cost = 10.
coll_cost = 100.
track_width = 1.
LON_THRES = 3.
EP_LEN = 500

# trajectory_type = "counter oval"
trajectory_type = "berlin_2018"
SIM = 'numerical' # 'numerical' or 'unity' or 'vicon'


if SIM == 'numerical' :
    trajectory_type = "../../simulators/params-num.yaml"
    LF = 0.12
    LR = 0.24
    L = LF+LR

if SIM=='unity' :
    trajectory_type = "../../simulators/params.yaml"
    LF = 1.6
    LR = 1.5
    L = LF+LR

if SIM == 'unity' :
    SPEED = 10.0
else :
    SPEED = 2.

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if SIM == 'unity' :
    yaml_contents = yaml.load(open(trajectory_type, 'r'), Loader=yaml.FullLoader)
    
    decay_start = yaml_contents['vehicle_params']['friction_decay_start']
    decay_rate = yaml_contents['vehicle_params']['friction_decay_rate']

AUGMENT = False
use_gt = True
HISTORY = 8
ART_DELAY = 0
MAX_DELAY = 7
new_delay_factor = 0.1
curr_delay = 0.
N_ensembles = 1
append_delay_type = 'OneHot' # 'OneHot' or 'Append'
LAST_LAYER_ADAPTATION = False
mass = 1.
I = 1.

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')
parser.add_argument('--pre', action='store_true', help='Enable pre-training')
parser.add_argument('--lstm', action='store_true', help='Enable lstm')

args = parser.parse_args()
if args.lstm: 
    MODEL = 'nn-lstm' # 'nn' or 'nn-lstm
else :
    MODEL = 'nn' # 'nn' or 'nn-lstm


model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)#random.randint(1,5))
model_params_single_opp = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)#random.randint(1,5))

stats = {'lat_errs': [], 'ws_gt': [], 'ws_': [], 'ws': [], 'losses': [], 'date_time': time.strftime("%m/%d/%Y %H:%M:%S"),'buffer': 100, 'delay': DELAY, 'model': MODEL, 'speed': SPEED, 'total_length': 2000, 'history': HISTORY, 'traj': [], 'ref_traj': []}
for i in range(N_ensembles):
    stats['losses'+str(i)] = []
    
if args.exp_name == 'none' :
    exp_name = MODEL + str(SPEED)
    if AUGMENT :
        exp_name += '_aug'
    exp_name += time.strftime("_%m_%d_%Y_%H_%M_%S")
else :
    exp_name = args.exp_name


dynamics_single = DynamicBicycleModel(model_params_single)
dynamics_single_opp = DynamicBicycleModel(model_params_single_opp)

dynamics_single.reset()
dynamics_single_opp.reset()


waypoint_generator = WaypointGenerator(trajectory_type, DT, H, 2.)
waypoint_generator_opp = WaypointGenerator(trajectory_type, DT, H, 1.)

state_lattice = {'arr': [], 's_list': []}
for i in range(N_lat_divs):
    traj_ = waypoint_generator.left_boundary + (waypoint_generator.right_boundary - waypoint_generator.left_boundary)*(i+1)/(N_lat_divs+1)
    traj = traj_[::dist_long,:]
    state_lattice['arr'].append(traj)
state_lattice['s_list'] = waypoint_generator.path[::dist_long,0]
state_lattice['arr'] = np.array(state_lattice['arr']).transpose(1,0,2)

done = False
frames = []

if SIM == 'numerical' :
    env = OffroadCar({}, dynamics_single)
    env_opp = OffroadCar({}, dynamics_single_opp)
    obs = env.reset(pose=[5.,7.,-np.pi/2.-0.7])
    obs_opp = env_opp.reset(pose=[0.,0.,-np.pi/2.-0.5])

goal_list = []
target_list = []
action_list = []
mppi_action_list = []
obs_list = []


pos2d = []
target_list_all = []


curr_steer = 0.
class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')
        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        self.path_pub_nn = self.create_publisher(Path, 'path_nn', 1)
        self.path_pub_nn_opp = self.create_publisher(Path, 'path_nn_opp', 1)
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.left_boundary_pub_ = self.create_publisher(Path, 'left_boundary', 1)
        self.right_boundary_pub_ = self.create_publisher(Path, 'right_boundary', 1)
        self.raceline_pub_ = self.create_publisher(Path, 'raceline', 1)
        self.state_lattice_pub_ = self.create_publisher(Path, 'state_lattice', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        self.odom_pub_ = self.create_publisher(Odometry, 'odom', 1)
        self.odom_opp_pub_ = self.create_publisher(Odometry, 'odom_opp', 1)
        self.timer_ = self.create_timer(DT/3., self.timer_callback)
        self.slow_timer_ = self.create_timer(10.0, self.slow_timer_callback)
        self.throttle_pub_ = self.create_publisher(Float64, 'throttle', 1)
        self.steer_pub_ = self.create_publisher(Float64, 'steer', 1)
        self.trajectory_array_pub_ = self.create_publisher(MarkerArray, 'trajectory_array', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
        self.body_opp_pub_ = self.create_publisher(PolygonStamped, 'body_opp', 1)
        self.status_pub_ = self.create_publisher(Int8, 'status', 1)
        self.raceline = waypoint_generator.raceline
        self.ep_no = 0
        
        self.curr_speed_factor = 0.6
        self.curr_lookahead_factor = 0.5
        self.curr_sf1 = 0.3
        self.curr_sf2 = 2.
        
        self.L = LF+LR
        
        self.curr_speed_factor_opp = 0.55
        self.curr_lookahead_factor_opp = 0.5
        self.curr_sf1_opp = 0.3
        self.curr_sf2_opp = 2.
        
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
        self.unity_state_new = [0.,0.,0.,0.,0.,0.]
        self.dataset = []
        self.buffer = []
    
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
    
    def cbf_filter(self,s,s_opp,vs,vs_opp,sf1=0.3,sf2=0.3,lookahead_factor=1.0) :
        eff_s = s_opp-s + (vs_opp-vs)*lookahead_factor
        factor = sf1*np.exp(-sf2*np.abs(eff_s))
        return factor
    
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
    
    def obs_state_opp(self):
        if SIM == 'unity' :
            if self.pose_received and self.vel_received :
                return np.array(self.unity_state_new)
            else :
                return np.array([yaml_contents['respawn_loc']['z'], yaml_contents['respawn_loc']['x'], 0., 0., 0., 0.])
        elif SIM == 'numerical':
            return env_opp.obs_state()
    
    def timer_callback(self):
        global obs, obs_opp, target_list_all, stats, h_0s, c_0s, action, curr_steer
        ti = time.time()
        # print(self.i)
        if SIM == 'unity' and not self.pose_received :
            return
        if SIM == 'unity' and not self.vel_received :
            return
        
        self.i += 1
        print("iter:", self.i)
        # RESTART_PARAMS
        if self.i > EP_LEN :
            self.ep_no += 1
            self.i = 0
            waypoint_generator.last_i = -1
            waypoint_generator_opp.last_i = -1
            obs = env.reset(pose=[5.,7.,-np.pi/2.-0.7])
            obs_opp = env_opp.reset(pose=[0.,0.,-np.pi/2.-0.5])
            self.curr_sf1 = np.random.uniform(0.1,0.5)
            self.curr_sf2 = np.random.uniform(0.1,0.5)
            self.curr_lookahead_factor = np.random.uniform(0.3,0.7)
            self.curr_speed_factor = np.random.uniform(0.5,0.9)
            self.curr_sf1_opp = np.random.uniform(0.1,0.5)
            self.curr_sf2_opp = np.random.uniform(0.1,0.5)
            self.curr_lookahead_factor_opp = np.random.uniform(0.3,0.7)
            self.curr_speed_factor_opp = np.random.uniform(0.4,0.7)
            self.dataset.append(np.array(self.buffer))
            self.buffer = []
            print("Saving dataset")
            pickle.dump(self.dataset, open('dataset.pkl','wb'))
            
        if self.ep_no > 100 :
            print("Saving dataset")
            pickle.dump(self.dataset, open('dataset.pkl','wb'))
            exit(0)
        mu_factor = 1.
        if SIM == 'unity' :
            if self.i*DT > decay_start :
                mu_factor = 1. - (self.i*DT - decay_start)*decay_rate
            mu_msg = Float64()
            mu_msg.data = mu_factor
            self.mu_factor_pub_.publish(mu_msg)
        
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
        status = Int8()
        target_pos_tensor, _, s, e = waypoint_generator.generate(jnp.array(obs[:5]),dt=DT_torch,mu_factor=mu_factor)
        target_pos_tensor_opp, _, s_opp, e_opp = waypoint_generator_opp.generate(jnp.array(obs_opp[:5]),dt=DT_torch,mu_factor=mu_factor)
        # print("h: ", target_pos_tensor)
        
        
        
        print("lat_err: ",e,e_opp)
        stats['traj'].append([self.obs_state()])
        target_pos_list = np.array(target_pos_tensor)
        
        action = np.array([0.,0.])
        px, py, psi, vx, vy, omega = self.obs_state().tolist()
        px_opp, py_opp, psi_opp, vx_opp, vy_opp, omega_opp = self.obs_state_opp().tolist()
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
        self.status_pub_.publish(status)
        
        
        if self.i < 6 :
            action[0] = 0.
            action[1] = 0.
        
        if SIM == 'numerical':
            closest_idx = np.sum(state_lattice['s_list']<(s+0.4))
            extended_path = np.concatenate((state_lattice['arr'][closest_idx:,:,:],state_lattice['arr'][:5,:,:]),axis=0)
            ego_plan = self.mcts_planner([px,py,psi,vx,vy,omega,e,s],extended_path[:5,:,:],[px_opp,py_opp,psi_opp,vx_opp,vy_opp,omega_opp,e_opp,s_opp])
            # action, ref_path = self.pure_pursuit_controller(ego_plan,self.obs_state(),speed=2.) 
            # print("Executing action: ", action)
            steer, throttle, curv, curv_lookahead = self.pure_pursuit((px,py,psi),(s,e,vx),(s_opp,e_opp,vx_opp),self.curr_sf1,self.curr_sf2,self.curr_lookahead_factor,self.curr_speed_factor)
            obs, reward, done, info = env.step(np.array([throttle,steer]))
            
            closest_idx = np.sum(state_lattice['s_list']<(s_opp+0.4))
            extended_path = np.concatenate((state_lattice['arr'][closest_idx:,:,:],state_lattice['arr'][:5,:,:]),axis=0)
            # action_opp, ref_path_opp = self.pure_pursuit_controller(extended_path[:5,1,:],self.obs_state_opp(),speed=1.) 
            # action_opp[0] = 1.
            # print("Executing action: ", action_opp)
            print("Lat error: ", e, e_opp)
            steer, throttle, curv_opp, curv_opp_lookahead = self.pure_pursuit((px_opp,py_opp,psi_opp),(s_opp,e_opp,vx_opp),(s,e,vx),self.curr_sf1_opp,self.curr_sf2_opp,self.curr_lookahead_factor_opp,self.curr_speed_factor_opp)
            
            action_opp = np.array([throttle,steer])
            obs_opp, reward, done, info = env_opp.step(action_opp)
            
            # State is s_opp-s, e_opp-e, theta, vx, vy, omega, theta_opp, vx_opp, vy_opp, omega_opp
            state_obs = [s,s_opp, e,e_opp, obs[2], obs[3], obs[4], obs[5], obs_opp[2], obs_opp[3], obs_opp[4], obs_opp[5],curv,curv_opp,curv_lookahead,curv_opp_lookahead,self.curr_sf1,self.curr_sf2,self.curr_lookahead_factor,self.curr_speed_factor,self.curr_sf1_opp,self.curr_sf2_opp,self.curr_lookahead_factor_opp,self.curr_speed_factor_opp]

            self.buffer.append(state_obs)
        w_pred_ = 0.
        _w_pred = 0.
        if len(self.states) > stats['buffer']  :
            
            self.states = self.states[-stats['buffer']:]
            self.cmds = self.cmds[-stats['buffer']:]
            
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
        
        # Odom for opponent
        q_opp = quaternion_from_euler(0, 0, psi_opp)
        
        odom = Odometry()
        odom.header.frame_id = 'map'
        odom.header.stamp = now
        odom.pose.pose.position.x = px_opp
        odom.pose.pose.position.y = py_opp
        odom.pose.pose.orientation.x = q_opp[0]
        odom.pose.pose.orientation.y = q_opp[1]
        odom.pose.pose.orientation.z = q_opp[2]
        odom.pose.pose.orientation.w = q_opp[3]
        odom.twist.twist.linear.x = vx_opp
        odom.twist.twist.linear.y = vy_opp
        odom.twist.twist.angular.z = omega_opp
        self.odom_opp_pub_.publish(odom)
        
        
        # print(np.array(mppi_info['action']).shape)
        
        throttle = Float64()
        throttle.data = float(action_opp[0])
        self.throttle_pub_.publish(throttle)
        steer = Float64()
        curr_steer += 1.0*(float(action_opp[1]) - curr_steer) 
        steer.data = curr_steer #float(action[1])
        self.steer_pub_.publish(steer)
        
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
        
        # body polygon
        pts = np.array([
            [LF, L/3],
            [LF, -L/3],
            [-LR, -L/3],
            [-LR, L/3],
        ])
        # transform to world frame
        R = euler_matrix(0, 0, psi_opp)[:2, :2]
        pts = np.dot(R, pts.T).T
        pts += np.array([px_opp, py_opp])
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = now
        for i in range(pts.shape[0]):
            p = Point32()
            p.x = float(pts[i, 0])
            p.y = float(pts[i, 1])
            p.z = 0.
            body.polygon.points.append(p)
        self.body_opp_pub_.publish(body)
        
        # path = Path()
        # path.header.frame_id = 'map'
        # path.header.stamp = self.get_clock().now().to_msg()
        # for i in range(ref_path.shape[0]):
        #     pose = PoseStamped()
        #     pose.header.frame_id = 'map'
        #     pose.pose.position.x = ref_path[i][0]
        #     pose.pose.position.y = ref_path[i][1]
        #     path.poses.append(pose)
        # self.path_pub_nn.publish(path)
        
        # path = Path()
        # path.header.frame_id = 'map'
        # path.header.stamp = self.get_clock().now().to_msg()
        # for i in range(ref_path.shape[0]):
        #     pose = PoseStamped()
        #     pose.header.frame_id = 'map'
        #     pose.pose.position.x = ref_path_opp[i][0]
        #     pose.pose.position.y = ref_path_opp[i][1]
        #     path.poses.append(pose)
        # self.path_pub_nn_opp.publish(path)
        
        tf = time.time()
        print("Time taken", tf-ti)
        if SIM == 'unity' :
            self.pose_received = False
            self.vel_received = False

    def get_curvature(self, x1, y1, x2, y2, x3, y3):
        a = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        b = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        c = np.sqrt((x3-x1)**2 + (y3-y1)**2)
        s = (a+b+c)/2
        return 4*np.sqrt(s*(s-a)*(s-b)*(s-c))/(a*b*c)
    
    def mcts_planner(self, state, lattice, opp_state):
        lat, lon = state[-2:]
        lat_opp, lon_opp = opp_state[-2:]
        print("long:", lon,lon_opp)
        v_opp = opp_state[3]
        mincost = 1000.
        minwaypoints = None
        ta = time.time()
        print(N_lat_divs**len(lattice))
        waypoints = np.zeros((N_lat_divs**len(lattice),len(lattice),3))
        lat_errs = np.zeros((N_lat_divs**len(lattice),len(lattice)))
        t1 = 0.
        for i in range(N_lat_divs**len(lattice)):
            curr_i = i
            tc = time.time()
            for j in range(len(lattice)) :
                waypoints[i,j,:] = lattice[j][curr_i%N_lat_divs]
                lat_errs[i,j] = track_width/2. - ((curr_i%N_lat_divs) + 1)*track_width/(N_lat_divs+1)
                curr_i = curr_i//N_lat_divs
        waypoints = np.array(waypoints)
        xs = waypoints[:, :, 0]
        ys = waypoints[:, :, 1]
        vs = waypoints[:, :, 2]
        nw = waypoints.shape[0]
        # print(xs)
        _xs = np.concatenate((np.array([[state[0]]*nw]).T,xs),axis=1)
        _ys = np.concatenate((np.array([[state[1]]*nw]).T,ys),axis=1)
        _vs = np.concatenate((np.array([[state[3]]*nw]).T,vs),axis=1)
        dists = np.sqrt((_xs[:,1:] - _xs[:,:-1])**2 + (_ys[:,1:] - _ys[:,:-1])**2)
        for j in range(1,dists.shape[1]) :
            dists[:,j] += dists[:,j-1]
        ts = dists/vs
        _ts = np.concatenate((np.array([[0.]*nw]).T,ts),axis=1)
        cost = 0.
        td = time.time()
        t1 += td-tc
        for j in range(_xs.shape[1]-2) :
            cost += curv_cost*np.sqrt(self.get_curvature(_xs[:,j],_ys[:,j],_xs[:,j+1],_ys[:,j+1],_xs[:,j+2],_ys[:,j+2]))
        curr_lon_opp = lon_opp + v_opp*ts
        curr_lon = lon + vs*ts
        cost += coll_cost*np.sum((np.abs(curr_lon_opp-curr_lon)<LON_THRES)*(track_width-np.abs(lat_opp-lat_errs)),axis=1)
        mini = np.argmin(cost)
        # print(curr_lon_opp[mini])
        # print(curr_lon[mini])
        # print("lat_errs:", lat_errs[mini])
        # print(curr_lon_opp[:5])
        # print(curr_lon[:5])
        # print(dists[mini])
        # print(dists[:5])
        # print(_xs[mini],_ys[mini])
        # print(_xs[:5],_ys[:5])
        # print(cost)
        minwaypoints = waypoints[mini]
        tb = time.time()
        print("time diff: ", tb-ta, t1)
        return minwaypoints

    def pure_pursuit_controller(self, waypoints, state, P=1., speed=None):
        xs = waypoints[:, 0]
        ys = waypoints[:, 1]
        vs = waypoints[:, 2]
        # print(xs)
        _xs = np.concatenate((np.array([state[0]]),xs))
        _ys = np.concatenate((np.array([state[1]]),ys))
        _vs = np.concatenate((np.array([state[3]]),vs))
        if speed is not None :
            _vs[:] = speed
        
        dists = np.sqrt((_xs[1:] - _xs[:-1])**2 + (_ys[1:] - _ys[:-1])**2)
        for i in range(1,len(dists)) :
            dists[i] += dists[i-1]
        ts = dists/vs
        _ts = np.concatenate((np.array([0.]),ts))
        
        # Represent waypoints which is a numpy array of 2d points to as an interpolated curve using scipy.interpolate
        interp_x = scipy.interpolate.interp1d(_ts, _xs, kind='quadratic', fill_value='extrapolate')
        interp_y = scipy.interpolate.interp1d(_ts, _ys, kind='quadratic', fill_value='extrapolate')
        interp_v = scipy.interpolate.interp1d(_ts, _vs, kind='quadratic', fill_value='extrapolate')
        lookahead_x = interp_x(0.2)
        lookahead_y = interp_y(0.2)
        lookahead_v = interp_v(0.2)
        dist = np.sqrt((lookahead_x - state[0])**2 + (lookahead_y - state[1])**2)
        # print("Distance: ", dist)
        
        times = np.linspace(0, ts[-1], 100)
        x = interp_x(times)
        y = interp_y(times)
        p = np.array([x, y]).T
        # Calculate the steer command from the lookahead point
        # Calculate the angle of the line connecting the lookahead point and the current state
        alpha = np.sin(np.arctan2(lookahead_y - state[1], lookahead_x - state[0]) - state[2])
        steer = np.arctan(2*L*alpha/dist)
        action = np.zeros(2)
        action[0] = P*(lookahead_v-state[3])
        action[1] = steer
        return action, p
    
    def pure_pursuit(self,xyt,pose,pose_opp,sf1,sf2,lookahead_factor,v_factor,gap=0.06) :
        s,e,v = pose
        
        x,y,theta = xyt
        s_opp,e_opp,v_opp = pose_opp
        shift = self.cbf_filter(s,s_opp,v,v_opp,sf1,sf2,lookahead_factor)
        if e>e_opp :
            shift = np.abs(shift)
        else :
            shift = -np.abs(shift)
        shift = max(min(max(0.5-e,0.),shift),min(-0.5-e,0.))
        # Find the closest point on raceline from x,y
        dists = np.sqrt((self.raceline[:,0]-x)**2 + (self.raceline[:,1]-y)**2)
        closest_idx = np.argmin(dists)
        curv = self.get_curvature(self.raceline[closest_idx-1,0],self.raceline[closest_idx-1,1],self.raceline[closest_idx,0],self.raceline[closest_idx,1],self.raceline[(closest_idx+1)%len(self.raceline),0],self.raceline[(closest_idx+1)%len(self.raceline),1])
        closest_point = self.raceline[closest_idx]
        lookahead_distance = lookahead_factor*v
        N = len(self.raceline)
        lookahead_idx = int(closest_idx+1+lookahead_distance//gap)%N
        lookahead_point = self.raceline[lookahead_idx]
        curv_lookahead = self.get_curvature(self.raceline[lookahead_idx-1,0],self.raceline[lookahead_idx-1,1],self.raceline[lookahead_idx,0],self.raceline[lookahead_idx,1],self.raceline[(lookahead_idx+1)%N,0],self.raceline[(lookahead_idx+1)%N,1])
        theta_traj = np.arctan2(self.raceline[(lookahead_idx+1)%N,1]-self.raceline[lookahead_idx,1],self.raceline[(lookahead_idx+1)%N,0]-self.raceline[lookahead_idx,0]) + np.pi/2.
        print(closest_idx,lookahead_idx,lookahead_distance,gap,shift)
        shifted_point = lookahead_point + shift*np.array([np.cos(theta_traj),np.sin(theta_traj),0.])
        
        v_target = v_factor*lookahead_point[2]
        throttle = (v_target-v)
        # Pure pursuit controller
        _dx = shifted_point[0]-x
        _dy = shifted_point[1]-y
        
        dx = _dx*np.cos(theta) + _dy*np.sin(theta)
        dy = _dy*np.cos(theta) - _dx*np.sin(theta)
        alpha = np.arctan2(dy,dx)
        steer = 2*self.L*dy/(dx**2 + dy**2)
        if np.abs(alpha) > np.pi/4 :
            steer = np.sign(dy) 
        return steer, throttle, curv, curv_lookahead
    
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

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(waypoint_generator.left_boundary.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.left_boundary[i][0])
            pose.pose.position.y = float(waypoint_generator.left_boundary[i][1])
            path.poses.append(pose)
        self.left_boundary_pub_.publish(path)

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(waypoint_generator.right_boundary.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.right_boundary[i][0])
            pose.pose.position.y = float(waypoint_generator.right_boundary[i][1])
            path.poses.append(pose)
        self.right_boundary_pub_.publish(path)

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(waypoint_generator.raceline.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.raceline[i][0])
            pose.pose.position.y = float(waypoint_generator.raceline[i][1])
            path.poses.append(pose)
        self.raceline_pub_.publish(path)

        # path = Path()
        # path.header.frame_id = 'map'
        # path.header.stamp = self.get_clock().now().to_msg()
        # N = state_lattice['arr'].shape[0]
        # for i in range(state_lattice['arr'].shape[0]):
        #     for j in range(state_lattice['arr'].shape[1]):
        #         pose = PoseStamped()
        #         pose.header.frame_id = 'map'
        #         pose.pose.position.x = float(state_lattice['arr'][i,j,0])
        #         pose.pose.position.y = float(state_lattice['arr'][i,j,1])
        #         yaw = np.arctan2(state_lattice['arr'][(i+1)%N,j,1]-state_lattice['arr'][i-1,j,1],state_lattice['arr'][(i+1)%N,j,0]-state_lattice['arr'][i-1,j,0])
                
        #         q = quaternion_from_euler(0, 0, yaw)
        #         # print(q)
        #         pose.pose.orientation.x = q[0]
        #         pose.pose.orientation.y = q[1]
        #         pose.pose.orientation.z = q[2]
        #         pose.pose.orientation.w = q[3]
        #         path.poses.append(pose)
                
        # self.state_lattice_pub_.publish(path)

def main():
    rclpy.init()
    car_node = CarNode()
    rclpy.spin(car_node)
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
