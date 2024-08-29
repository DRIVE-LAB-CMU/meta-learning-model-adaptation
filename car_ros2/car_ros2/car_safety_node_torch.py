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

res_model_learning = False
# learning_rate = 0.001

DT = 0.05
DELAY = 1
L = 0.3
Ka = 1.0
Kv = 6.
wall_dist = 1.95
steer_factor = 0.1
goal_poses = np.array([[1.7,1.7],[-1.7,-1.7]])
thres = 0.4
vmax = 1.5
Kd = 1.0
controller_mul = 3.
cbf_dist = 0.6
lamb = 0.1

vmax_ = 0.2
Kd_ = 0.02
cbf_dist_ = 0.03
cbf_speed = 1.
lamb_ = 0.01

_vmax = 2.5
_Kd = 1.5
_cbf_dist = 0.8
lamb_max = 0.3
lamb_min = 0.1
vmin = 1.5

SIM = 'vicon' # 'numerical' or 'unity' or 'vicon'

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


if SIM == 'vicon' :
    learning_rate = 0.0004
    vicon_thread = threading.Thread(target=update_vicon)
    vicon_thread.start()
    LF = 0.16
    LR = 0.15
    L = LF+LR
    client_ip = "rcar.wifi.local.cmu.edu"
    client_port = 12347
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    

if SIM == 'numerical' :
    learning_rate = 0.001
    LF = 0.16
    LR = 0.15
    L = LF+LR

if SIM=='unity' :
    learning_rate = 0.00001
    trajectory_type = "../../simulators/params.yaml"
    LF = 1.6
    LR = 1.5
    L = LF+LR

if SIM == 'unity' :
    SPEED = 10.0
else :
    SPEED = 2.


cmd_buffer = []

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

model = SimpleModel(4, [64,64], 2)
model = model.double()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(data) :
    global model, optimizer
    if len(data) < 2 :
        return
    X = data[:-1,:4]
    print(data.shape)
    Y = torch.tensor(np.stack((data[1:,4],data[1:,0]-data[:-1,0]),axis=1))
    Y_nom = torch.zeros_like(Y)
    Y_nom[:,0] = torch.tensor(X[:,0])*torch.tan(steer_factor*torch.tensor(X[:,2]))/L
    if SIM == 'numerical' :
        Y_nom[:,1] = torch.tensor(Ka*X[:,3])
    else :
        Y_nom[:,1] = torch.tensor(Ka*(Kv*X[:,3] - X[:,0]))
    X[:,2] = X[:,0]*X[:,2]
    model = model.double()
    Y_pred = model(torch.tensor(X).double())
    loss = nn.MSELoss()(Y_pred+Y_nom,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
    
    
def cbf_filter(cmd_,pose,delay=0.) :
    global cbf_dist, cmd_buffer, model
    cmd_buffer.append(cmd_)
    if len(cmd_buffer) > DELAY :
        cmd_buffer = cmd_buffer[1:]
    cmd = cmd_buffer[0]
    x,y,theta,vx,vy,w = pose
    x = x + delay*vx*np.cos(theta)
    y = y + delay*vx*np.sin(theta)
    theta = theta + delay*w
    betas = [0.,np.pi/2.,np.pi,-np.pi/2.]
    # betas = []
    # print(cbf_dist)
    h = np.sqrt(x**2 + y**2) - cbf_dist
    alpha = theta - np.arctan2(-y,-x)
    hd = -vx*np.cos(alpha)
    D = np.sqrt(x**2 + y**2)
    if SIM == 'numerical' :
        ax = Ka*cmd[0]#(Kv*cmd[0] - vx)
    else :
        ax = Ka*(Kv*cmd[0] - vx)
    w_cmds = np.linspace(-1.,1.,100)
    ws = vx*np.tan(steer_factor*w_cmds)/L
    
    X = torch.zeros((100,4))
    X[:,0] = float(vx)
    X[:,1] = float(vy)
    X[:,2] = torch.tensor(vx*w_cmds)
    X[:,3] = float(cmd[1])
    model_pred = model(X.double()).detach().numpy()
    
    if res_model_learning :
        ws += model_pred[:,0]
        ax = ax + model_pred[:,1]
    hdds = vx*np.sin(alpha)*(ws-vx*np.sin(alpha)/D) - ax*np.cos(alpha)
    val = lamb**2*hdds + 2*lamb*hd + h
    costs = 1000.*(val<0.)*val**2 + (w_cmds-cmd[1])**2
    for i in range(len(betas)) :
        h_ = wall_dist - x*np.cos(betas[i]) - y*np.sin(betas[i])
        hd_ = -vx*np.cos(betas[i])*np.cos(theta) - vx*np.sin(betas[i])*np.sin(theta)
        hdds_ = -ax*np.cos(betas[i])*np.cos(theta) - ax*np.sin(betas[i])*np.sin(theta) + \
            ws*(-vx*np.sin(betas[i])*np.cos(theta) + vx*np.cos(betas[i])*np.sin(theta))
        val = lamb**2*hdds_ + 2*lamb*hd_ + h_
        costs += 1000.*(val<0.)*val**2
    w_cmd = w_cmds[np.argmin(costs)]
    
    a_cmds = np.linspace(-1.,1.,100)
    if SIM == 'numerical' :
        axs = Ka*a_cmds#(Kv*a_cmds - vx)
    else :
        axs = Ka*(Kv*a_cmds - vx)
    
    w = vx*np.tan(steer_factor*w_cmd)/L
    
    X = torch.zeros((100,4))
    X[:,0] = float(vx)
    X[:,1] = float(vy)
    X[:,2] = float(vx*w_cmd)
    X[:,3] = torch.tensor(a_cmds)
    model_pred = model(X.double()).detach().numpy()
    
    if res_model_learning :
        w = w + model_pred[:,0]
        axs = axs + model_pred[:,1]
    
    hdds = vx*np.sin(alpha)*(w-vx*np.sin(alpha)/D) - axs*np.cos(alpha)
    val = lamb**2*hdds + 2*lamb*hd + h
    costs = cbf_speed*(val<0.)*val**2 + (a_cmds-cmd[0])**2
    # print(costs, 2*lamb,hd , h)
    for i in range(len(betas)) :
        h_ = wall_dist - x*np.cos(betas[i]) - y*np.sin(betas[i])
        hd_ = -vx*np.cos(betas[i])*np.cos(theta) - vx*np.sin(betas[i])*np.sin(theta)
        hdds_ = -axs*np.cos(betas[i])*np.cos(theta) - axs*np.sin(betas[i])*np.sin(theta) + \
            w*(-vx*np.sin(betas[i])*np.cos(theta) + vx*np.cos(betas[i])*np.sin(theta))
        val = lamb**2*hdds_ + 2*lamb*hd_ + h_
        costs += 1000*(val<0.)*val**2
    # print(costs)
    a_cmd = a_cmds[np.argmin(costs)]
    print(a_cmd,w_cmd,cmd)
    return a_cmd, w_cmd
    
def controller(target,pose,ulta=False,delay=0.0) :
    xt,yt = target
    x0,y0,theta0,v0,_,w0 = pose
    x0 = x0 + delay*v0*np.cos(theta0)
    y0 = y0 + delay*v0*np.sin(theta0)
    theta0 = theta0 + delay*w0
    Dt = np.sqrt((xt-x0)**2 + (yt-y0)**2)
    vt = max(-vmax,min(vmax,Kd*Dt))
    # Pure pursuit controller to reach target
    alpha = np.arctan2(yt-y0,xt-x0) - theta0
    alpha = np.arctan2(np.sin(alpha),np.cos(alpha))
    # print(alpha)
    
    delta = controller_mul*2*L*np.sin(alpha)/(0.34*np.sqrt(xt**2 + yt**2))
    if alpha < np.pi/2 and alpha > -np.pi/2 :
        # if v0 < -0.1 :
        if ulta :
            if vt < -0 :
                delta = -delta
        else :
            if v0 < -0.1 :
                delta = -delta
        if SIM == 'numerical' :
            return cbf_filter(((vt-v0)/Kv, delta),pose)
        else :
            return cbf_filter((vt/Kv, delta),pose)
    else :
        # if v0 > 0.1 :
        if ulta :
            if vt > 0. :
                delta = -delta
        else :
            if v0 > 0.1 :
                delta = -delta
        if SIM == 'numerical' :
            return cbf_filter(((-vt-v0)/Kv, delta),pose)
        else :
            return cbf_filter((-vt/Kv, delta),pose)



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')
parser.add_argument('--pre', action='store_true', help='Enable pre-training')
parser.add_argument('--lstm', action='store_true', help='Enable lstm')

args = parser.parse_args()
model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=1.5,delay=1)#random.randint(1,5))
stats = {'lat_errs': [], 'ws_gt': [], 'ws_': [], 'ws': [], 'losses': [], \
    'date_time': time.strftime("%m/%d/%Y %H:%M:%S"),'buffer': 100, 'lr': learning_rate, \
    'online_transition': 300, 'speed': SPEED, 'total_length': 10000, 'traj': []}
    

exp_name = ""
if args.exp_name == 'none' :
    exp_name += time.strftime("_%m_%d_%Y_%H_%M_%S")
else :
    exp_name = args.exp_name

dynamics_single = DynamicBicycleModel(model_params_single)
dynamics_single.reset()

if SIM == 'numerical' :
    env = OffroadCar({}, dynamics_single)
    obs = env.reset()
elif SIM == 'vicon' :
    obs = np.array([0.,0.,0.,0.,0.,0.])
    pose_x = 0.
    pose_y = 0.
    pose_yaw = 0.
    t_prev = time.time()


curr_steer = 0.
class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        self.odom_pub_ = self.create_publisher(Odometry, 'odom', 1)
        self.timer_ = self.create_timer(DT, self.timer_callback)
        self.throttle_pub_ = self.create_publisher(Float64, 'throttle', 1)
        self.steer_pub_ = self.create_publisher(Float64, 'steer', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
        self.states = []
        self.cmds = []
        self.data = []
        self.i = 0
        self.curr_t_counter = 0.
        self.vicon_loc = np.array(vicon_loc)
        self.curr_pose_i = 0
        self.violated = False
        self.violated_boundary = False
        
    def obs_state(self):
        if SIM == 'numerical':
            return env.obs_state()
        else :
            return self.vicon_loc
    
    def timer_callback(self):
        global cbf_dist, Kd, vmax, curr_steer, lamb
        print("Lamb is ", lamb)
        ti = time.time()
        print(self.i,self.curr_pose_i)
        global obs, target_list_all, stats, h_0s, c_0s, action, curr_steer
        
        self.i += 1
        stats['traj'].append([self.obs_state()])
        ulta = False
        if self.i > 3000 :
            ulta = False
        action = controller(goal_poses[self.curr_pose_i],self.obs_state(),ulta)
        action = np.array(action)
        # print(action)
        self.curr_t_counter += DT
        px, py, psi, vx, vy, omega = self.obs_state().tolist()
        self.data.append([vx,vy,vx*action[0],action[1],omega])
        self.data = self.data[-100:]
        train(np.array(self.data))
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
        target_pos_list = np.array([[px,py],[goal_poses[self.curr_pose_i][0],goal_poses[self.curr_pose_i][1]]])
        for i in range(target_pos_list.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(target_pos_list[i][0])
            pose.pose.position.y = float(target_pos_list[i][1])
            path.poses.append(pose)
        self.ref_trajectory_pub_.publish(path)
        
        
        if self.i > stats['total_length'] :
            self.get_logger().info('Saving stats')
            with open(f'safety_results/stats_{exp_name}.pkl', 'wb') as f:
                pickle.dump(stats, f)
            exit(0)
        if SIM == 'numerical':
            # print(action)
            # action[0] = -1.
            # action[1] = -1.
            obs, reward, done, info = env.step(action)
        elif SIM == 'vicon' :
            obs = np.array(vicon_loc)
            self.vicon_loc = np.array(vicon_loc)
            # client_socket.sendto(struct.pack('dd',vx*0.13+action[0]*0.4,action[1]),(client_ip,client_port))
            if abs(action[0]) < 0.14 :
                action[0] = 0.14 * np.sign(action[0])
            print("Throttle", action[0])
            client_socket.sendto(struct.pack('dd',action[0],action[1]),(client_ip,client_port))
            
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
        odom.twist.twist.angular.x = np.sqrt(px**2 + py**2)
        if np.sqrt((px)**2 + (py)**2) < 0.8 :#cbf_dist :
            self.violated = True
        if px < -1.9 or px > 1.9 or py < -1.9 or py > 1.9 :
            self.violated_boundary = True
        self.odom_pub_.publish(odom)
        print("vmax: ", vmax)
        throttle = Float64()
        throttle.data = float(action[0])
        self.throttle_pub_.publish(throttle)
        steer = Float64()
        curr_steer += 0.1*(float(action[1]) - curr_steer) 
        steer.data = curr_steer #float(action[1])
        self.steer_pub_.publish(steer)
        
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
        
        if np.sqrt((px-goal_poses[self.curr_pose_i][0])**2 + (py-goal_poses[self.curr_pose_i][1])**2) < thres :
            self.curr_pose_i = (self.curr_pose_i + 1) % 2
            cbf_dist += cbf_dist_
            Kd += Kd_
            Kd = min(_Kd,Kd)
            cbf_dist = min(_cbf_dist,cbf_dist)
            if self.violated :
                lamb += lamb_
                lamb = min(lamb_max,lamb)
                
            else :
                
                lamb -= lamb_/2.
                lamb = max(lamb_min,lamb)
            
            if self.violated_boundary :
                vmax -= vmax_/2.
                vmax = max(vmin,min(_vmax,vmax))
            else :
                vmax += vmax_
                vmax = max(vmin,min(_vmax,vmax))
            self.violated = False
            self.violated_boundary = False
        tf = time.time()
        print("Time taken", tf-ti)
        
        
    

def main():
    rclpy.init()
    car_node = CarNode()
    rclpy.spin(car_node)
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
