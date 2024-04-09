import rospy
import socket
from sensor_msgs.msg import NavSatFix, Joy
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import Odometry
from scipy.optimize import minimize

import threading
import time
import json
import os
import struct
from datetime import datetime
import numpy as np
import ros_numpy
from copy import deepcopy
import pickle
import math
from termcolor import colored

from fusion_engine_client.parsers import FusionEngineDecoder
from fusion_engine_client.messages import PoseMessage

from car_dynamics.analysis import rotate_point
from car_dynamics.models_torch import MLP
from car_dynamics.controllers_torch import PIDController
# from ..utils import keyboard_server

import warnings
width = 336
height = 188
SAVE_FREQ = 10
BUFFER_SIZE = 8
num_log = 0
DT = 0.05

LEARNING_BUFFER = 100
learning_rate = 0.01
adaptible_params = ['Sb']
N_ROLLOUTS = 1000
H = 8
SIGMA = 1.0
VEL_ALPHA = 0.8
velocity = 0.0
prev_x = 0.0
prev_y = 0.0
prev_psi = 0.0
# controller_type = 'mppi-dbm' # mppi-nn, mppi-kbm, pid, debug
# controller_type = 'mppi-kbm' # mppi-nn, mppi-kbm, pid, debug
# controller_type = 'pid' # mppi-nn, mppi-kbm, pid, debug
controller_type = 'teleop-dbm' # mppi-nn, mppi-kbm, pid, debug, random_walk
# controller_type = 'mppi-nn-end2end-trunk' # mppi-nn, mppi-kbm, pid, debug, random_walk
# trajectory_type = 'circle' # (counter-)circle/oval, straight, center
trajectory_type = 'counter oval' # (counter-)circle/oval, straight, center
# trajectory_type = 'center' # (counter-)circle/oval, straight, center
SPEED = 2.0
time_complement = 0.0
MAX_VEL = 8.
PID_KP = 5.0
PID_KI = 0.0
PID_KD = 0.05
MPPI_STEER_PROJ = .43
# 。45 for clockwise, .48 for counter
MPPI_STEER_SHIFT = 0.17
DELAY=2
waypoint_projection_shift_time = DT * .0
len_history = 10
LF = .16
LR = .15
L = LF+LR
RECOVOER_TIME = 2
RECOVER_COUNTDOWN = 2
WITHOUT_VICON = False
# For data collection
class Collector:
    center = np.array([1, 1.68])
    x_radius = 1.3
    y_radius = 1.62
    
    def in_boundary(self, x, y):
        return np.abs(x-self.center[0]) <= self.x_radius and \
                np.abs(y-self.center[1]) <= self.y_radius
                
    def find_nearest_vertex(self, x, y, psi):
        vertex_pos = np.array([
            [self.x_radius, self.y_radius],
            [self.x_radius, -self.y_radius],
            [-self.x_radius, self.y_radius],
            [-self.x_radius, -self.y_radius],
        ]) * .1 + self.center
        
        min_angle = np.pi*2
        min_idx = -1
        for i, pos in enumerate(vertex_pos):
            delta_pos = pos - np.array([x, y])
            delta_angle = np.arctan2(delta_pos[1], delta_pos[0]) - psi
            delta_angle = np.abs(np.arctan2(np.sin(delta_angle), np.cos(delta_angle)))
            if delta_angle < min_angle:
                min_idx = i
                
        return vertex_pos[i]


def steering_to_ros_command(command: float, calibrate=False):
    # check calibrate.ipynb
    #   , we found the maximum symmetric range to be [-0.75, 0.635]
    if calibrate:
        a = 0.7225
        b = -0.0375
        return a * command + b
    else:
        return command

mppi_config = dict()
# if 'teleop' in controller_type:
#     import torch
#     from car_dynamics.envs import make_env, KinematicBicycleModel, KinematicParams, DynamicParams, DynamicBicycleModel
#     from car_dynamics.controllers_torch import MPPIController, rollout_fn_select, reward_track_fn
    
#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # DEVICE = torch.device("cpu")
    
if 'mppi' in controller_type or 'teleop' in controller_type :
    import torch
    from car_dynamics.envs import make_env, KinematicBicycleModel, KinematicParams, DynamicParams, DynamicBicycleModel
    from car_dynamics.controllers_torch import MPPIController, rollout_fn_select, reward_track_fn
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    

    print("DEVICE", DEVICE)
    if 'nn-phyx-kbm' in controller_type:
        model_dir = f'/home/cmu/catkin_ws/offroad/tmp/20240204-233419/model.pt'
        dynamics = MLP(input_size=7*len_history-2, hidden_size=256, output_size=2)
        dynamics.load(model_dir)
        dynamics.to(DEVICE)
        rollout_fn = rollout_fn_select('nn-phyx-kbm', dynamics, DT, L, LR)
    elif 'nn-heading-psi' in controller_type :
        model_dir = f'/Users/randyxiao/Library/CloudStorage/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/tmp/20240127-224644/model.pt'
        dynamics_nn = MLP(input_size=6*len_history, hidden_size=128, output_size=5)
        dynamics_nn.load(model_dir)
        dynamics_nn.to(DEVICE)
        rollout_fn = rollout_fn_select('nn-heading-psi', dynamics_nn, DT, L, LR)
    elif 'nn-heading' in controller_type:
        model_dir = f'/Users/randyxiao/Library/CloudStorage/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/tmp/20240127-202009/model.pt'
        dynamics_nn = MLP(input_size=6*len_history, hidden_size=512, output_size=3)
        dynamics_nn.load(model_dir)
        dynamics_nn.to(DEVICE)
        rollout_fn = rollout_fn_select("nn-heading", dynamics_nn, DT, L, LR)
    elif 'nn-end2end-trunk' in controller_type:
        model_dir = f'/home/cmu/catkin_ws/offroad/tmp/20240203-135537/model.pt'
        dynamics_nn = MLP(input_size=7*len_history, hidden_size=32, output_size=4)
        dynamics_nn.load(model_dir)
        dynamics_nn.to(DEVICE)
        rollout_fn = rollout_fn_select('nn-end2end-trunk', dynamics_nn, DT, L, LR)
    elif 'nn-end2end' in controller_type:
        # model_dir = f'/Users/randyxiao/Library/CloudStorage/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/tmp/20240128-190950/model.pt'
        model_dir = f'/home/cmu/catkin_ws/offroad/tmp/20240202-011311/model.pt'
        dynamics_nn = MLP(input_size=6*len_history, hidden_size=64, output_size=4)
        dynamics_nn.load(model_dir)
        dynamics_nn.to(DEVICE)
        rollout_fn = rollout_fn_select('nn-end2end', dynamics_nn, DT, L, LR)
    elif 'kbm' in controller_type:
        model_params = KinematicParams(
                        num_envs=N_ROLLOUTS,
                        last_diff_vel=torch.zeros([N_ROLLOUTS, 1]).to(DEVICE),
                        KP_VEL=5.96,
                        KD_VEL=.02,
                        MAX_VEL=MAX_VEL,
                        PROJ_STEER=MPPI_STEER_PROJ,
                        SHIFT_STEER=MPPI_STEER_SHIFT,
                        DT=DT,
        )   
        dynamics = KinematicBicycleModel(model_params, device=DEVICE)
        rollout_fn = rollout_fn_select('kbm', dynamics, DT, L, LR)
    elif 'dbm' in controller_type:
        model_params = DynamicParams(num_envs=N_ROLLOUTS, DT=DT,Sa=0.45, Sb=0.0, K_FFY=15, K_RFY=15, Ta=8.)
        mppi_config.update(model_params.to_dict())
        dynamics = DynamicBicycleModel(model_params, device=DEVICE)
        rollout_fn = rollout_fn_select('dbm', dynamics, DT, L, LR)
    else:
        raise NotImplementedError
    
    print(rollout_fn)
        
    sigmas = torch.tensor([SIGMA] * 2)
    a_cov_per_step = torch.diag(sigmas**2)
    a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
    # a_cov_prev =  torch.full((H, 2, 2), 3.0**2) * torch.eye(2).unsqueeze(0).repeat(H, 1, 1)

    def fn():
        return 
    mppi = MPPIController(
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
        rollout_fn=rollout_fn,
        a_min = [-1., -1.],
        a_max = [1., 1.],
        # a_mag = [0.06, 1.],
        # a_mag = [0.02, 1.],
        # a_shift = [.08, 0.],
        a_mag = [0.1, 1.], # 0.1, 0.35
        a_shift = [0.3, 0.],
        # a_shift = [.06, 0.],
        delay=DELAY,
        len_history=len_history,
        rollout_start_fn=fn,
        debug=False,
        fix_history='phyx-kbm' in controller_type,
        num_obs=6,
        num_actions=2,
    )

pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD)


###################### Logging dir ##################
DATA_DIR = '/home/cmu/catkin_ws/offroad/data/dataset'
timestr = time.strftime("%Y%m%d-%H%M%S")
logdir = 'data-' + timestr
os.mkdir(os.path.join(DATA_DIR, logdir))

#####################################################

car_state = dict(
                    obs=None,
                    date=None,
                    time=None, 
                 vicon_loc=None,
                #  gps_vel=None,
                 left_rgb=None,
                 right_rgb=None,
                 lidar=None,
                 imu=None,
                 zed2_odom=None,
                 vesc_odom=None,
                 throttle=0.0,
                 steering=0.0,
                 velocity = 0.0,
                 recover=False,
            )

teleop_state = dict( throttle = 0., steering = 0. )

dataset = []

print("init")
####### VICON Callback ####################
def update_vicon_placeholder():
    
    while True :
        pose_x = np.random.uniform(-1, 3)
        pose_y = np.random.uniform(-1, 3)
        pose_yaw = np.random.uniform(-np.pi, np.pi)
        car_state['vicon_loc'] = [pose_x, pose_y, pose_yaw]
        time.sleep(0.001)
        # print(colored(f"received - {car_state['vicon_loc']}", "green"))


def update_vicon():
    server_ip = "0.0.0.0"
    server_port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((server_ip, server_port))

    print(f"UDP server listening on {server_ip}:{server_port}")

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
    while True :
        data, addr = server_socket.recvfrom(24)  # 3 doubles * 8 bytes each = 24 bytes
        unpacked_data = struct.unpack('ddd', data)
        # print(f"Received pose from {addr}: {unpacked_data}")
        pose_x = unpacked_data[0]/1000.
        pose_y = unpacked_data[1]/1000.
        pose_yaw = unpacked_data[2] - np.pi/2
        pose_yaw = np.arctan2(np.sin(pose_yaw), np.cos(pose_yaw))
        pose_x, pose_y = rotate_point(pose_x, pose_y, 90)
        pose_x += 1.757
        pose_y += 1.414
        car_state['vicon_loc'] = [pose_x, pose_y, pose_yaw]
        # print(colored(f"received - {car_state['vicon_loc']}", "green"))


########### Logging Fn ######################

def dumpe_file(name, data):
    def fn():
        with open(name, "wb") as f:
            # json.dump(data, f)
            pickle.dump(data, f)
        # print(f"[INFO] {data[0]['time']}, {data[0]['date']}")
    return fn

def logging():
    global dataset, num_log
    # print(len(dataset))
    if len(dataset) > SAVE_FREQ:
        # print(os.path.join(DATA_DIR, logdir, f"log{num_log}.json"))
        dataset2 = dataset[:SAVE_FREQ]
        dataset = dataset[SAVE_FREQ:]
        # print([d['time'] for d in dataset])
        assert id(dataset2) != id(dataset)
        # print(dataset)
        print("start to log")
        save_threading = threading.Thread(target=dumpe_file(os.path.join(DATA_DIR, logdir, f"log{num_log}.pkl"), dataset2))
        save_threading.start()
        save_threading.join()
        # with open(os.path.join(DATA_DIR, logdir, f"log{num_log}.json"), "w") as f:
        #     json.dump({'test':dataset2}, f)
            # json.dump(dict(data=dataset[:10]), f)
        # print("after save")
        # dataset = dataset[10:]
        num_log += 1
        print("saved", num_log, "logs")

if 'teleop' in controller_type or 'random_walk' in controller_type:
    TCP_IP = "0.0.0.0"  # Replace with the IP address of your laptop
    TCP_PORT = 15214  # Replace with the desired port number
    KeyboardConnect = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    KeyboardConnect.bind((TCP_IP, TCP_PORT))
    
def keyboard_server(teleop_dict):
    BUFFER_SIZE = 8
    while True:
        data, addr = KeyboardConnect.recvfrom(BUFFER_SIZE)
        if not data:
            break
        unpacked_data = struct.unpack('ff', data)
        teleop_dict['throttle'] = unpacked_data[0]
        teleop_dict['steering'] = unpacked_data[1]
    while True:
        print("BREAK")

#############################################

    
def reference_vel(t):
    if controller_type == 'pid' and t < 0.5:
        return 0
    return SPEED

def reference_traj(t):
    if trajectory_type == 'circle':
        if controller_type == 'debug':
            if t < .4:
                t = .1
            else:
                t -= .2
        # global total_angle
        center_circle = (1., 1.2)
        circle_radius = 1.3
        angle = -np.pi/2  - circle_radius * SPEED * t
        return np.array([center_circle[0] + circle_radius * np.cos(angle),
                            center_circle[1] + circle_radius * np.sin(angle)])
        
    elif trajectory_type == 'counter circle':
        if controller_type == 'debug':
            if t < .4:
                t = .1
            else:
                t -= .2
        # global total_angle
        center_circle = (1., 1.2)
        circle_radius = 1.3
        angle = -np.pi/2  + circle_radius * SPEED * t
        return np.array([center_circle[0] + circle_radius * np.cos(angle),
                            center_circle[1] + circle_radius * np.sin(angle)])
    elif trajectory_type == 'oval':
        if controller_type == 'debug':
            if t < .4:
                t = .1
            else:
                t -= .2
        center = (0.9, 1.0)
        x_radius = 1.2
        y_radius = 1.4

        # Assuming t varies from 0 to 2π to complete one loop around the oval
        angle = -np.pi/2  - x_radius * SPEED * t

        x = center[0] + x_radius * np.cos(angle)
        y = center[1] + y_radius * np.sin(angle)

        return np.array([x, y])

    elif trajectory_type == 'counter oval':
        if controller_type == 'debug':
            if t < .4:
                t = .1
            else:
                t -= .2
        center = (0.8, 1.0)
        x_radius = 1.2
        y_radius = 1.4

        # Assuming t varies from 0 to 2π to complete one loop around the oval
        angle = -np.pi/2  + x_radius * SPEED * t

        x = center[0] + x_radius * np.cos(angle)
        y = center[1] + y_radius * np.sin(angle)

        return np.array([x, y])

    elif trajectory_type == 'straight':
        start = np.array([-0.7, -0.2])
        end = np.array([2., 3.5])
        angle_ = np.arctan2(end[1] - start[1], end[0] - start[0])
        time_length = np.linalg.norm(end-start) / SPEED
        if t < 0:
            t = 0
        if t > time_length:
            t = time_length
        x = start[0] + SPEED * t * np.cos(angle_)
        y = start[1] + SPEED * t * np.sin(angle_)
        return np.array([x, y])
    elif trajectory_type == 'center':
        return np.array([1., 1.2])
    else:
        raise NotImplementedError


def distance_to_trajectory(t, p0, trajectory_func):
    x, y = trajectory_func(t)
    return np.sqrt((x - p0[0])**2 + (y - p0[1])**2)

def find_projection(p0, trajectory_func, initial_guess=0):
    # t_list = np.arange(-np.pi-DT, np.pi+DT, 0.01)
    result = minimize(distance_to_trajectory, initial_guess, args=(p0, trajectory_func))
    if result.success:
        # import pdb; pdb.set_trace()
        return result.x
    else:
        raise ValueError("Optimization failed")

def get_steer_error(obs, goal):
    # psi = obs[2]
    psi = obs[4]
    psi_des = np.arctan2(goal[1] - obs[1], goal[0] - obs[0])
    err = psi_des - psi
    # err = (err + np.pi) % (2 * np.pi) - np.pi
    while err > np.pi:
        err -= 2 * np.pi
    while err < -np.pi:
        err += 2 * np.pi
    # print(psi, psi_des, err)
    return err


def joystick_controller():
    
    if 'mppi' in controller_type \
        or 'pid' in controller_type:
        waypoint_t_list = np.arange(-np.pi*3-DT, np.pi*4+DT, 0.01)
        waypoint_list = np.array([reference_traj(t) for t in waypoint_t_list])
    
    if controller_type in [ 'random_walk' ]:
        data_collecter = Collector()
        
    goal_ptr = 0
    header_info = dict(goal_list=[],controller=controller_type, trajectory=trajectory_type,
                       pid=dict(kp=PID_KP,ki=PID_KI,kd=PID_KD, max_vel=MAX_VEL),
                       mppi=dict(proj=MPPI_STEER_PROJ, shift=MPPI_STEER_SHIFT, max_vel=MAX_VEL,delay=DELAY, config=mppi_config),
                       tag='identify vicon discount error',)
    with open(os.path.join(DATA_DIR, logdir, f"header.json"), "w") as f:
        json.dump(header_info, f)
        
    delta_t = DT
    t_ = 0.
    goal_list = []
    targets = []
    pid_recover = False
    vicon_recover = False
    lost_count = 0
    boundary_count = 0
    buffer = []
    while True:
        global pub, car_state, prev_x, prev_y, prev_psi, velocity, time_complement, RECOVOER_TIME
        now = rospy.Time.now()
        cmd = Joy()
        cmd.header.stamp = now
        cmd.buttons = [0,0,0,0,0,0,1,0,0,0,0]
        # car_state['time'] = datetime.now()
        car_state['time'] =time.time()
        car_state['date'] = datetime.now()
        
        # Calculate time difference
        
        # print(f"car state: {car_state['vicon]}")
        if car_state['vicon_loc'] is None:
            print(colored("DISCOUNNECT", "red"))
            continue
        
        # print("before", car_state['vicon_loc'])
        st_control = time.time()
        # Calculate position differences
        delta_x = car_state['vicon_loc'][0] - prev_x
        delta_y = car_state['vicon_loc'][1] - prev_y
        omega = car_state['vicon_loc'][2] - prev_psi
        omega = np.arctan2(np.sin(omega), np.cos(omega)) / delta_t
        
        # Check direction
        vec_pos = np.array([delta_x, delta_y])
        
        v_vec = vec_pos / delta_t
        
      
        vx = v_vec[0] * np.cos(car_state['vicon_loc'][2]) + v_vec[1] * np.sin(car_state['vicon_loc'][2])
        vy = v_vec[1] * np.cos(car_state['vicon_loc'][2]) - v_vec[0] * np.sin(car_state['vicon_loc'][2])
        
        if 'Sb' in adaptible_params:
            if abs(vx) > 0.4 and abs(car_state['throttle']) > 0.05 and abs(vx/car_state['throttle']) < 10.:
                buffer.append([vx,omega,car_state['steering']])
            if len(buffer) > LEARNING_BUFFER:
                buffer = buffer[-LEARNING_BUFFER:]
                buffer = np.array(buffer)
                vxs = buffer[:,0]
                omegas = buffer[:,1]
                steers = buffer[:,2]
                wheel_angles = dynamics.params.Sa * steers + dynamics.params.Sb
                omegas_expected = vxs * np.tan(wheel_angles) / (dynamics.params.LF + dynamics.params.LR)
                error = np.mean((omegas - omegas_expected)**2)
                print("error: ", error)
                grad = np.mean(2*(omegas - omegas_expected) * (-vxs / (np.cos(wheel_angles)**2*(dynamics.params.LF + model_params.LR))))
                dynamics.params.Sb -= learning_rate*grad
                print(colored(f"Adaptive Sb: {dynamics.params.Sb}", "green"))
            else :
                print(len(buffer))
            # dynamics.params.Sb = 0.0
        
        vel_angle = np.arctan2(delta_y, delta_x)
        vec_car = np.array([np.cos(car_state['vicon_loc'][2]), np.sin(car_state['vicon_loc'][2])])
        idc = np.dot(vec_pos, vec_car)

        if idc >= 0:
            idc = 1.0
        else:
            idc = -1.0

        # Calculate velocity
        velocity = velocity \
            + (idc * math.sqrt(delta_x**2 + delta_y**2) / delta_t - velocity) * VEL_ALPHA
        
        prev_x = car_state['vicon_loc'][0]
        prev_y = car_state['vicon_loc'][1]
        prev_psi = car_state['vicon_loc'][2]
        
        velocity_norm = 0. if velocity < 0. else velocity
        
        
        obs = np.array([
            car_state['vicon_loc'][0],
            car_state['vicon_loc'][1],
            car_state['vicon_loc'][2],
            velocity_norm,
            vel_angle,
        ])
        
        obs_mppi = np.array([
            car_state['vicon_loc'][0],
            car_state['vicon_loc'][1],
            car_state['vicon_loc'][2],
            vx,
            vy,
            omega,
        ])
        
        start_to_control = time.time()
        target_pos = reference_traj(t_)
        
        car_state['recover'] = False
        if 'random_walk' in controller_type:
            if not data_collecter.in_boundary(obs[0], obs[1]):
                pid_recover = True
                boundary_count = RECOVER_COUNTDOWN
            else:
                boundary_count -= 1
                if boundary_count < 0:
                    pid_recover = False
        
        if 'random_walk' in controller_type:
            if pid_recover or vicon_recover:
                print(colored("switch to teleop", "red"))
                car_state['recover'] = True
                action = np.zeros(2)
                action[0] = teleop_state['throttle']
                action[1] = teleop_state['steering']
            else:
                car_state['recover'] = False
                action = np.random.uniform(-1, 1, size=2)
                action[0] = np.random.uniform(0.05, 0.2)
            
        elif 'teleop' in controller_type:
            if vicon_recover:
                car_state['recover'] = True
            
            action_0 = teleop_state['throttle']
            action_1 = teleop_state['steering'] - dynamics.params.Sb/dynamics.params.Sa
            # action_0 = np.random.normal(loc=teleop_state['throttle'], scale=0.05)
            # if teleop_state['steering'] not in [-1., 1.]:
            #     action_1 = np.random.normal(loc=teleop_state['steering'], scale=0.5)
            # else:
            #     action_1 = teleop_state['steering']
            # action_0 = np.random.uniform(0., 0.1)
            # action_1 = np.random.uniform(-1., 1.)
            action_0 = np.clip(action_0, -.6, .6)
            action_1 = np.clip(action_1, -1., 1.)
            # print(colored(f"teleop, {teleop_state}, {action_0}, {action_1}", "red"))
            action = np.array([action_0, action_1])
            
        elif ('debug-nn' in controller_type and t_ > 1 and t_ < RECOVOER_TIME) \
                or vicon_recover \
                or pid_recover \
                or 'pid' in controller_type:
                    
            if 'pid' not in controller_type:
                print(colored("[INFO] temporarily using PID", "red"))
            
            if pid_recover:
                # target_pos = data_collecter.center
                target_pos = data_collecter.find_nearest_vertex(obs[0], obs[1], obs[2])
            
            else:
                distance_list = np.linalg.norm(waypoint_list - obs[:2], axis=-1)
                # import pdb; pdb.set_trace()
                t_idx = np.argmin(distance_list)
                t_closed = waypoint_t_list[t_idx]
                target_pos_list = [reference_traj(t_closed + i*DT) for i in range(H+DELAY)]
                target_pos = target_pos_list[1] # for pid, need 1 waypoint ahead
                car_state['targets'] = np.array(target_pos_list).tolist()
            
            # dist = np.linalg.norm(obs[:2] - target_pos) / (delta_t * 10)
            dist = np.linalg.norm(obs[:2] - target_pos) / (delta_t * 2.5)
            if 'mppi' in controller_type:
                car_state['recover'] = True
            if pid_recover:
                throttle = np.clip(dist / MAX_VEL, .06, .1)
                car_state['recover'] = True
            elif vicon_recover:
                throttle = np.clip(dist / MAX_VEL, .1, .2)
                car_state['recover'] = True
            else:
                throttle = np.clip(dist / MAX_VEL, .1, .35)
            steering_error = get_steer_error(obs, target_pos)
            # print(colored(f"err: {steering_error}", "red"))
            steering = pid(steering_error, DT)
            # print("sterring error", steering_error, steering)
            
            action = np.array([throttle, steering])
            
        elif 'mppi' in controller_type:
            ## TODO: Project waypoints based on current position
            # t_closed = find_projection(obs[:2], reference_traj)[0]
            distance_list = np.linalg.norm(waypoint_list - obs[:2], axis=-1)
            # import pdb; pdb.set_trace()
            t_idx = np.argmin(distance_list)
            t_closed = waypoint_t_list[t_idx]
            target_pos_list = [reference_traj(waypoint_projection_shift_time + t_closed + i*DT*.75) for i in range(H+DELAY+1)]
            target_pos = target_pos_list[0]
            # import pdb; pdb.set_trace()
            target_pos_tensor = torch.Tensor(target_pos_list).to(DEVICE).squeeze(dim=-1)
            # if 'kbm' in controller_type:
            #     dynamics_kbm.reset()
            st_mppi = time.time()
            # action, mppi_info = mppi(obs[:4], reward_track_fn(target_pos_tensor, SPEED))
            action, mppi_info = mppi(obs_mppi, reward_track_fn(target_pos_tensor, SPEED))
            print(colored(f"mppi time - {time.time() - st_mppi}", "green"))
            action = action.cpu().numpy()
            # import pdb; pdb.set_trace()
            car_state['mppi_actions'] = mppi_info['action'].tolist()
            # print(car_state['mppi_actions'])
            car_state['targets'] = np.array(target_pos_list).tolist()
            
        elif 'debug' in controller_type:
            action = np.array([.1, .0])
        
        

            
        
        print(action)  
        # print(colored(f"target: {target_pos}", "red"))
        # print(colored(f"pos: {obs[:2]}, yaw: {obs[2]}, vel: {obs[3]}", 'blue'))
        if time.time() - start_to_control > DT:
            # action *= .0
            print("time out", "control time", time.time() - start_to_control)
        # action *= 0.
        
        if 'mppi-nn' in controller_type:
            mppi.feed_hist(obs, action)
            
        car_state['obs'] = np.concatenate([obs, target_pos])
        
        if t_>RECOVOER_TIME and delta_x == 0 and delta_y == 0 and action[0] != 0:
            lost_count = RECOVER_COUNTDOWN
            vicon_recover = True
            print(colored("\n\n!!!!!!!! Vicon get lost !!!!!\n\n", "red"))
        else:
            lost_count -= 1
            if lost_count <= 0:
                vicon_recover = False
      
        
        car_state['throttle'] = action[0]
        car_state['steering'] = action[1]
        
        
        ################### Publish Command to ROS #############################
        ros_command_steering = steering_to_ros_command(car_state['steering'], calibrate=True)
        # ros_command_steering = car_state['steering']
        # print("ROS Command", car_state['throttle'], ros_command_steering)
        print(car_state['throttle'],ros_command_steering)
        cmd.axes = [0.,car_state['throttle'],ros_command_steering,0.,0.,0.]
        # cmd.axes = [0.,forward_speed,(steering/max_steering),steering/max_steering,0.,0.]
        # print("REAL COMMAND", cmd)
        pub.publish(cmd)
        ########################################################################
        
        dataset.append(car_state.copy()) 
        goal_list.append(target_pos)
            
            
        # # print("pub", throttle, steering)
        # if car_state['imu'] is not None and \
        #         car_state['zed2_odom'] is not None:
        #     # print("start to collect", car_state['time'])
        #     # print(f"yaw, imu:{quaternion_to_euler(car_state['imu']['orientation'])[2]}, zed:{quaternion_to_euler(car_state['zed2_odom']['orientation'])[2]}, vesc:{quaternion_to_euler(car_state['vesc_odom']['orientation'])[2]}")
        #     # print("after", car_state['vicon_loc'])
        #     dataset.append(car_state.copy()) 
        #     goal_list.append(target_pos)
            
        #     # if len(dataset) > 1:
        #         # print(dataset[-2]['time'], dataset[-1]['time'])
        # # print("before logging!")
        logging()
        time_complement = time.time() - st_control
        # print(time_complement)
        if DT > time_complement:
            time.sleep(DT - time_complement)
        else:
            ...
        # print("time complement", time_complement)
        # assert DT > time_complement
        # time.sleep(DT)
        
        t_ += time.time() - st_control
        
        if trajectory_type == 'straight' and t_ > 24:
            break
        if t_ > 1000:
            break
            
        delta_t = time.time() - st_control
        # print(colored(f"[INFO] delta time: {delta_t}", "red"))
    
    goal_list = np.array(goal_list)
    header_info = dict(goal_list=goal_list.tolist(),controller=controller_type,
                       pid=dict(kp=PID_KP,ki=PID_KI,kd=PID_KD, max_vel=MAX_VEL),
                       mppi=dict(proj=MPPI_STEER_PROJ, shift=MPPI_STEER_SHIFT, max_vel=MAX_VEL,delay=DELAY),)
    with open(os.path.join(DATA_DIR, logdir, f"header.json"), "w") as f:
        json.dump(header_info, f)
###############################################

############## ROS Callback functions ##########
def left_rgb_callback(data):
    np_arr = np.frombuffer(data.data, dtype=np.uint8)
    image = np_arr.reshape((height, width, -1))
    rgb_image = image[:, :, :3]
    data = rgb_image.tolist()
    global car_state
    car_state['left_rgb'] = data

def right_rgb_callback(data):
    np_arr = np.frombuffer(data.data, dtype=np.uint8)
    image = np_arr.reshape((height, width, -1))
    rgb_image = image[:, :, :3]
    data = rgb_image.tolist()
    global car_state
    car_state['right_rgb'] = data

def lidar_callback(data):
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
    data = xyz_array.tolist()
    global car_state
    car_state['lidar'] = data

def imu_callback(data):
    global car_state
    orientation = np.array([data.orientation.x,data.orientation.y,data.orientation.z,data.orientation.w]).tolist()
    angular_velocity = np.array([data.angular_velocity.x,data.angular_velocity.y,data.angular_velocity.z]).tolist()
    linear_acceleration = np.array([data.linear_acceleration.x,data.linear_acceleration.y,data.linear_acceleration.z]).tolist()
    car_state['imu']=dict(orientation=orientation,angular_vel=angular_velocity,linear_acc=linear_acceleration)
    
def imu_rpy_callback(data):
    ...
    
def imu_all_callback(data):
    ...

def vesc_odom_callback(data):
    global car_state
    position = np.array([data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z]).tolist()
    orientation = np.array([data.pose.pose.orientation.x,data.pose.pose.orientation.y,
                            data.pose.pose.orientation.z,data.pose.pose.orientation.w]).tolist()
    linear_velocity = np.array([data.twist.twist.linear.x,data.twist.twist.linear.y,data.twist.twist.linear.z]).tolist()
    angular_velocity = np.array([data.twist.twist.angular.x,data.twist.twist.angular.y,data.twist.twist.angular.z]).tolist()
    car_state['vesc_odom']=dict(position=position,orientation=orientation,linear_vel=linear_velocity,angular_vel=angular_velocity)

def zed2_odom_callback(data):
    global car_state
    position = np.array([data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z]).tolist()
    orientation = np.array([data.pose.pose.orientation.x,data.pose.pose.orientation.y,
                            data.pose.pose.orientation.z,data.pose.pose.orientation.w]).tolist()
    linear_velocity = np.array([data.twist.twist.linear.x,data.twist.twist.linear.y,data.twist.twist.linear.z]).tolist()
    angular_velocity = np.array([data.twist.twist.angular.x,data.twist.twist.angular.y,data.twist.twist.angular.z]).tolist()
    car_state['zed2_odom']=dict(position=position,orientation=orientation,linear_vel=linear_velocity,angular_vel=angular_velocity)


################################################
        
if __name__ == '__main__':
    print("start")
    rospy.init_node('controller', anonymous=True, disable_signals=True)
    global pub
    pub = rospy.Publisher('/vesc/joy',Joy)
    
    if 'teleop' in controller_type or 'random_walk' in controller_type:
        keyboard_thread = threading.Thread(target=keyboard_server, args=(teleop_state, ))
        keyboard_thread.start()
    
    if WITHOUT_VICON:
        vicon_fn = update_vicon_placeholder
    else:
        vicon_fn = update_vicon
    vicon_thread = threading.Thread(target=vicon_fn)
    vicon_thread.start()

    joystick_thread = threading.Thread(target=joystick_controller)
    joystick_thread.start()


    ## Camera node
    # rospy.Subscriber('/front_camera/zed2/zed_node/left/image_rect_color', Image, left_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/right/image_rect_color', Image, right_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/velodyne_points', PointCloud2, lidar_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/imu/data', Imu, imu_callback, queue_size = 1)
    rospy.Subscriber('/vesc/odom', Odometry, vesc_odom_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/odom', Odometry, zed2_odom_callback, queue_size = 1)

    rospy.spin()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        vicon_thread.join()
        if 'teleop' in controller_type or 'random_walk' in controller_type:
            keyboard_thread.join()
        joystick_thread.join()
        vicon_thread.join()