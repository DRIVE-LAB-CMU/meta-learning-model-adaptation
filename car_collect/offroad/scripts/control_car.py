import roslib
# roslib.load_manifest('racecar')
import sys
import rospy
import socket
import struct
import threading
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import geometry_msgs.msg
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import os 
from scipy.stats import norm
import time
from stable_baselines3 import PPO
# from termcolor import colored
from datetime import datetime
import random
import string

now = datetime.now()
res = ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
FILE_NAME = f"{now.strftime('%H:%M:%S')}-{res}"

MAX_STEP = 50
LOCO_MODE = 'vicon' # or 'slam'
# GOAL_POS = np.array([-0.0,-1.5])
VEL_ALPHA = 0.8

slam_pose_x = 0
slam_pose_y = 0
slam_pose_yaw = 0

prev_slam_pose_x = 0
prev_slam_pose_y = 0
prev_time = 0
velocity = 0

curvature_factor = 0.5
x_factor = 1
theta_factor = 0.1

start_time =0



# def plot_traj(x_coords, y_coords, theta_angles,file_name='test.png'):
#     import matplotlib.pyplot as plt
#     plt.clf()
#     # Plot the points
#     # plt.scatter(x_coords, y_coords, color='blue')
#
#     # Plot the trajectory (lines connecting points)
#     plt.plot(x_coords, y_coords, linestyle='-', color='grey')
#     plt.plot(x_coords[-1], y_coords[-1], marker='*')
#
#     # Add arrows to indicate direction
#     arrow_length = 0.2  # You can set this to be proportional to the distance between points if you like
#     for x, y, theta in zip(x_coords, y_coords, theta_angles):
#         dx = arrow_length * np.cos(theta)
#         dy = arrow_length * np.sin(theta)
#         plt.arrow(x, y, dx, dy, head_width=0.001, head_length=0.001, fc='red', ec='red')
#
#     plt.axis('equal')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Trajectory with Direction')
#     plt.savefig(file_name)

def pos_callback(data) :
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x, slam_pose_y = data.pose.position.x, data.pose.position.y
  _,_,slam_pose_yaw = convert_xyzw_to_rpy(data.pose.orientation.x,data.pose.orientation.y,data.pose.orientation.z,data.pose.orientation.w)

def pos_callback_amcl(data) :
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x, slam_pose_y = data.pose.pose.position.x, data.pose.pose.position.y
  _,_,slam_pose_yaw = convert_xyzw_to_rpy(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w)


def get_latest_pose() :
    global slam_pose_x, slam_pose_y, slam_pose_yaw
    server_ip = "0.0.0.0"
    server_port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((server_ip, server_port))

    print(f"UDP server listening on {server_ip}:{server_port}")

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
    while True :
        data, addr = server_socket.recvfrom(24)  # 3 doubles * 8 bytes each = 24 bytes
        unpacked_data = struct.unpack('ddd', data)
        # print(f"Received pose from {addr}: {unpacked_data}")
        slam_pose_x = unpacked_data[0]/1000.
        slam_pose_y = unpacked_data[1]/1000.
        slam_pose_yaw = unpacked_data[2]

        ## Coorect coords missmatch
        slam_pose_x = -slam_pose_x
        slam_pose_y = -slam_pose_y

        yaw1 = np.arctan2(-np.sin(slam_pose_yaw), -np.cos(slam_pose_yaw))
        # print("yaw",np.cos(slam_pose_yaw), np.sin(slam_pose_yaw))
        # print("yaw1", yaw1)
        slam_pose_yaw = yaw1


        ### Calibrate Coordinates


def convert_xyzw_to_rpy(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
    
        return roll_x, pitch_y, yaw_z # in radians



def get_bicycle_state():
    global prev_slam_pose_x, prev_slam_pose_y, prev_time
    global slam_pose_x, slam_pose_y, slam_pose_yaw
    global velocity

    # Current time
    current_time = time.time()
    
    if prev_time == 0:
        prev_time = current_time
        prev_slam_pose_x = slam_pose_x
        prev_slam_pose_y = slam_pose_y
        return None  # Skip the first iteration to initialize previous pose and time
    
    # Calculate time difference
    delta_t = current_time - prev_time
    
    # Calculate position differences
    delta_x = slam_pose_x - prev_slam_pose_x
    delta_y = slam_pose_y - prev_slam_pose_y

    # Check direction
    vec_pos = np.array([delta_x, delta_y])
    vec_car = np.array([np.cos(slam_pose_yaw), np.sin(slam_pose_yaw)])
    idc = np.dot(vec_pos, vec_car)

    if idc >= 0:
        idc = 1.0
    else:
        idc = -1.0

    # Calculate velocity
    velocity = velocity \
        + (idc * math.sqrt(delta_x**2 + delta_y**2) / delta_t - velocity) * VEL_ALPHA

    
    # Update previous pose and time
    prev_slam_pose_x = slam_pose_x
    prev_slam_pose_y = slam_pose_y
    prev_time = current_time
    
    return slam_pose_x, slam_pose_y, slam_pose_yaw, velocity

class controller:
    '''RL Controller'''
    max_acc = 0.3
    min_acc = -0.3
    # max_acc = 1.
    # min_acc = -1.
    max_steer = 1.
    min_steer = -1.
    max_x = 4.
    max_y = 4.
    max_vel = 8.

    def __init__(self):
        self.prevx = 0
        self.prevy = 0
        self.prev_cmd = 0
        self.prevyaw = 0
        self.prevt = 0
        self.traj = []
        self.frames = []
        self.nop = False

    def step(self,goal_x, goal_y):
        if self.nop:
            return
        global model, pub
        now = rospy.Time.now()
        #print("Current time of transform : ",float(str(now))/1e9)
        state = get_bicycle_state()
        if state is None:
            state = get_bicycle_state()

        # dist_goal=np.sqrt(state[0]**2 + (state[1]+2.)**2)

        # print("dist goal", dist_goal, state[:2])

        # if dist_goal < 0.5:
        #     self.nop = True  
        real_pos_x, real_pos_y, yaw, vel = state

        print("OBS LOC: ", real_pos_x, real_pos_y)

        if real_pos_x < -2. or real_pos_x > 2. or real_pos_y < -2.5 or real_pos_y > 2.5:
            print("[ERROR] Exceed region!!!!!")

            cmd = Joy()
            cmd.header.stamp = now
            cmd.buttons = [0,0,0,0,1,0,0,0,0,0,0]
        
            cmd.axes = [0.,.0,.0,0.,0.,0.]
        
            # cmd.axes = [0.,forward_speed,(steering/max_steering),steering/max_steering,0.,0.]
            print("REAL COMMAND", cmd)
            pub.publish(cmd)
            return

        obs = np.array([
            real_pos_x / self.max_x,
            real_pos_y / self.max_y,
            np.cos(yaw),
            np.sin(yaw),
            vel / self.max_vel,
            goal_x / self.max_x,
            goal_y / self.max_y,
            0.,
            0.
        ])

        # print(obs)

        (a_val, str_val), _ = model.predict(obs)


        print("original throttle", a_val, str_val)

        throttle = max(min(a_val, self.max_acc), self.min_acc)
        str_val = max(min(str_val, self.max_steer), self.min_steer)
        
        self.frames.append(np.concatenate((state,[goal_x, goal_y], [throttle,str_val, time.time()]),axis=0))
        print("throttle", throttle, str_val)
        #print(f"[INFO] Current Action: ({throttle}, {str_val})")
        cmd = Joy()
        cmd.header.stamp = now
        cmd.buttons = [0,0,0,0,0,0,1,0,0,0,0]
    
        cmd.axes = [0.,throttle,str_val,0.,0.,0.]
    
        # cmd.axes = [0.,forward_speed,(steering/max_steering),steering/max_steering,0.,0.]
        # print("REAL COMMAND", cmd)
        pub.publish(cmd)

    def export_traj(self):
        frames = np.array(self.frames)
        np.savetxt(f'car-data/{FILE_NAME}.txt',frames)
        # plot_traj(frames[:,0], frames[:,1], frames[:,2], 'traj.png')

    

def main(args):

    if LOCO_MODE == 'vicon':
        t1 = threading.Thread(target=get_latest_pose, args=())
        t1.start()

    rospy.init_node('controller', anonymous=True, disable_signals=True)
    global pub

    pub = rospy.Publisher('/vesc/joy',Joy)
    # tf_listener = tf.TransformListener()
    global slam_pose_x, slam_pose_y, slam_pose_yaw, model
    global prev_slam_pose_x, prev_slam_pose_y, prev_time
    slam_pose_x = 0
    slam_pose_y = 0
    slam_pose_yaw = 0
    model = PPO.load(f"./model.zip")
    print("[INFO] RL Loaded")

    if LOCO_MODE == 'slam':
        amcl_pose = rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,pos_callback_amcl,queue_size=1)

    con = controller()

    # goals = [(-0.,-1.5), (-1., 0), (-1, 1.5), 
    #           (0., -1.5,), (0., 0.), (0., 1.5),
    #           (1., -1.5), (1., 0.), (1., 1.5),]
    goals = []
    for i in range(20 * 60 * 20):
        if i % 2 == 0:
            goals.append([0., 1.])
        else:
            goals.append([0., -1.])

    for (goal_x, goal_y) in goals:
        print("\n\n\n RESAMPLE!!! \n\n\n")
        for step in range(MAX_STEP):
            st = time.time()
            con.step(goal_x, goal_y)
            time.sleep(0.05)
            if rospy.is_shutdown():
                raise Exception("[ERROR] ROSPY SHUTDOWN!")
    # while not rospy.is_shutdown():
    #     con.step()
    #     time.sleep(0.05)
    #     _step += 1
    #     if (_step > MAX_STEP):
    #         break
            if step % 3 == 0:
                con.export_traj()
            print("TIME: ", time.time() - st)
    print("DONE.")
    
    

if __name__ == '__main__':
    main(sys.argv)
