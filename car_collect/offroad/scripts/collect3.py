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
import numpy as np
from scipy.signal import butter, lfilter, freqz
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
# DT = 0.05
N_ROLLOUTS = 1000
H = 4
SIGMA = 1.0
VEL_ALPHA = 0.8
velocity = 0.0
prev_x = 0.0
prev_y = 0.0
controller_type = 'teleop' # mppi-nn, mppi-kbm, pid, debug
# controller_type = 'mppi-kbm' # mppi-nn, mppi-kbm, pid, debug
# controller_type = 'pid' # mppi-nn, mppi-kbm, pid, debug
# controller_type = 'mppi-nn-phyx-kbm' # mppi-nn, mppi-kbm, pid, debug, random_walk
# controller_type = 'mppi-nn-end2end-trunk' # mppi-nn, mppi-kbm, pid, debug, random_walk
trajectory_type = 'center' # (counter-)circle/oval, straight, center
# trajectory_type = 'center' # (counter-)circle/oval, straight, center
SPEED = 4.0
time_complement = 0.0
MAX_VEL = 8.
PID_KP = 5.0
PID_KI = 0.0
PID_KD = 0.05
MPPI_STEER_PROJ = .43
# 。45 for clockwise, .48 for counter
MPPI_STEER_SHIFT = 0.17
DELAY=0
waypoint_projection_shift_time = DT * .0
len_history = 8
LF = .16
LR = .15
L = LF+LR
RECOVOER_TIME = 10
RECOVER_COUNTDOWN = 20
WITHOUT_VICON = False

###################### Logging dir ##################
DATA_DIR = '/home/cmu/catkin_ws/offroad/data'
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
                 gps=None,
            )

teleop_state = dict( throttle = 0., steering = 0. )

dataset = []

print("init")

########## Action Filter ####################


# Filter requirements.
order = 4
cutoff = 3.  # desired cutoff frequency of the filter, Hz

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


if 'teleop' in controller_type:
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


def steering_to_ros_command(command: float):
    # check calibrate.ipynb
    #   , we found the maximum symmetric range to be [-0.75, 0.635]
    # a = 0.7225
    # b = -0.0375
    # return a * command + b
    return command


def joystick_controller():
        
    header_info = dict(goal_list=[],controller=controller_type, trajectory=trajectory_type,
                       tag='collect in the wild',)
    with open(os.path.join(DATA_DIR, logdir, f"header.json"), "w") as f:
        json.dump(header_info, f)
        
    delta_t = DT
    fs = 1. / DT
    t_ = 0.
    # action_list = [np.zeros(2)] * 10
    action = np.zeros(2)
    while True:
        global pub, car_state, prev_x, prev_y, velocity, time_complement, RECOVOER_TIME
        now = rospy.Time.now()
        cmd = Joy()
        cmd.header.stamp = now
        cmd.buttons = [0,0,0,0,0,0,1,0,0,0,0]
        # car_state['time'] = datetime.now()
        car_state['time'] =time.time()
        car_state['date'] = datetime.now()
        
        
        # print("before", car_state['vicon_loc'])
        st_control = time.time()
       
        if 'teleop' in controller_type:
            action_0 = teleop_state['throttle']
            action_1 = teleop_state['steering']
            # action_0 = np.random.normal(loc=teleop_state['throttle'], scale=0.05)
            # if teleop_state['steering'] not in [-1., 1.]:
            #     action_1 = np.random.normal(loc=teleop_state['steering'], scale=0.5)
            # else:
            #     action_1 = teleop_state['steering']
            # action_0 = np.random.uniform(0., 0.1)
            # action_1 = np.random.uniform(-1., 1.)
            action_0 = np.clip(action_0, -.5, .5)
            action_1 = np.clip(action_1, -1., 1.)
            # print(colored(f"teleop, {teleop_state}, {action_0}, {action_1}", "red"))
            action = np.array([action_0, action_1])
        
        elif 'random_walk' in controller_type:
            action_new = np.random.uniform(-1, 1, size=2) - 0.08
            action_new[0] = np.random.uniform(0.1, 0.5)
            action_new[1] = np.clip(action_new[1], -1., 1.)
            action = action * 0.4 + action_new * .6
            # action_list.append(action)
            # # Filter the data, and plot both the original and filtered signals.
            # action_smooth_0 = butter_lowpass_filter(np.array(action_list)[-4:, 0], cutoff, fs, order)
            # action_smooth_1 = butter_lowpass_filter(np.array(action_list)[-4:, 1], cutoff, fs, order)
            # action[0] = action_smooth_0[-1]
            # action[1] = action_smooth_1[-1]
            # action_list[-1] = action
        
        # action *= 0.

        car_state['throttle'] = action[0]
        car_state['steering'] = action[1]
        
        
        ################### Publish Command to ROS #############################
        ros_command_steering = steering_to_ros_command(car_state['steering'])
        # ros_command_steering = car_state['steering']
        # print("ROS Command", car_state['throttle'], ros_command_steering)
        cmd.axes = [0.,car_state['throttle'],ros_command_steering,0.,0.,0.]
        # cmd.axes = [0.,forward_speed,(steering/max_steering),steering/max_steering,0.,0.]
        # print("REAL COMMAND", cmd)
        pub.publish(cmd)
        ########################################################################
        
        dataset.append(car_state.copy()) 
            
        logging()
        time_complement = time.time() - st_control
        # print(time_complement)
        if DT > time_complement:
            time.sleep(DT - time_complement)
        else:
            ...
        
        t_ += time.time() - st_control
        
        
        if t_ > 10000 * DT:
            break
            
###############################################

############## Update GPS #####################

def update_gps():
    TCP_IP = ""
    TCP_PORT = 15213

    msg_decoder = FusionEngineDecoder(
                max_payload_len_bytes=4096, warn_on_unrecognized=False, return_bytes=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((TCP_IP, TCP_PORT))

    BUFFER_SIZE = 1024
    print("HERE")
    while True:
        global car_state
        data = s.recv(BUFFER_SIZE)
        if not data:
            break
        try:
            data = msg_decoder.on_data(data)[0][1]
            if type(data) is PoseMessage:
                global gps_loc, gps_vel
                print(f"loc: {data.position_std_enu_m}\nvel: {data.velocity_std_body_mps}")
                gps_loc = data.position_std_enu_m
                gps_vel = data.velocity_std_body_mps
                car_state['gps'] = dict(loc=gps_loc, vel=gps_vel)
        except:
            print(type(data))
            
############## ROS Callback functions ##########

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


################################################
        
if __name__ == '__main__':
    print("start")
    rospy.init_node('controller', anonymous=True, disable_signals=True)
    global pub
    pub = rospy.Publisher('/vesc/joy',Joy)
    
    gps_thread = threading.Thread(target=update_gps)
    gps_thread.start()

    joystick_thread = threading.Thread(target=joystick_controller)
    joystick_thread.start()
    
    if 'teleop' in controller_type:
        keyboard_thread = threading.Thread(target=keyboard_server, args=(teleop_state, ))
        keyboard_thread.start()
    
    
    ## Camera node
    # rospy.Subscriber('/front_camera/zed2/zed_node/left/image_rect_color', Image, left_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/right/image_rect_color', Image, right_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/velodyne_points', PointCloud2, lidar_callback, queue_size = 1)
    # rospy.Subscriber('/imu/all', Imu, imu_all_callback, queue_size = 1)

    rospy.spin()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        gps_thread.join()
        joystick_thread.join()
        if 'teleop' in controller_type:
            keyboard_thread.join()
