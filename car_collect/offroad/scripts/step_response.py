import rospy
import socket
from sensor_msgs.msg import NavSatFix, Joy
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import Odometry


from fusion_engine_client.parsers import FusionEngineDecoder
from fusion_engine_client.messages import PoseMessage
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
'''
def quaternion_to_euler(input):
    """ quat: [x,y,z,w] -> [roll, pitch, yaw]
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x-axis, pitch is rotation around y-axis
    and yaw is rotation around z-axis.
    """
    # Normalize the quaternion
    x, y, z, w = input
    magnitude = math.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x / magnitude, y / magnitude, z / magnitude, w / magnitude

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # Use 90 degrees if out of range
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
'''

###################### Logging dir ##################
DATA_DIR = '/home/cmu/catkin_ws/offroad/data'
timestr = time.strftime("%Y%m%d-%H%M%S")
logdir = 'data-' + timestr
os.mkdir(os.path.join(DATA_DIR, logdir))

#####################################################

######################### Build Connection ##########

# TCP To receive RTK info
# TCP_IP = ""
# TCP_PORT = 15213  


# msg_decoder = FusionEngineDecoder( # to decode RTK msg
#             max_payload_len_bytes=4096, warn_on_unrecognized=False, return_bytes=True)

# GPSConnect = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# GPSConnect.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# GPSConnect.connect((TCP_IP, TCP_PORT))


width = 336
height = 188
SAVE_FREQ = 10
BUFFER_SIZE = 8
num_log = 0
DT = 0.05

car_state = dict(
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
            )

dataset = []



####### Keyboard Receiver Callback ##########

def keyboard_server():
    TCP_IP = ""  # Replace with the IP address of your laptop
    TCP_PORT = 15214  # Replace with the desired port number
    # Used for laptop keyboard control
    KeyboardConnect = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    KeyboardConnect.bind((TCP_IP, TCP_PORT))
    # KeyboardConnect.listen(1)

    # print("Waiting for keyboard connection...")
    # conn, addr = KeyboardConnect.accept()
    # print("Keyboard connected!")

    while True:
        global car_state
        data, addr = KeyboardConnect.recvfrom(BUFFER_SIZE)
        if not data:
            break
        unpacked_data = struct.unpack('ff', data)
        car_state['throttle'] = unpacked_data[0]
        car_state['steering'] = unpacked_data[1]
        # print(f"data: {unpacked_data}")

####### Logging Fn ######################

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

#############################################

############ Controller ####################
def joystick_controller():
    while True:
        global pub, car_state
        now = rospy.Time.now()
        cmd = Joy()
        cmd.header.stamp = now
        cmd.buttons = [0,0,0,0,0,0,1,0,0,0,0]
        # car_state['time'] = datetime.now()
        car_state['time'] =time.time()
        car_state['date'] = datetime.now()
        cmd.axes = [0.,car_state['throttle'],car_state['steering'],0.,0.,0.]
        # cmd.axes = [0.,forward_speed,(steering/max_steering),steering/max_steering,0.,0.]
        # print("REAL COMMAND", cmd)
        pub.publish(cmd)
        # print('time publish', time.time())
        # print("pub", throttle, steering)
       
        dataset.append(car_state.copy()) 
            # if len(dataset) > 1:
                # print(dataset[-2]['time'], dataset[-1]['time'])
        logging()
        time.sleep(DT)
###############################################


################################################
        
if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True, disable_signals=True)
    global pub
    pub = rospy.Publisher('/vesc/joy',Joy)

    # vicon_thread = threading.Thread(target=update_vicon)
    # vicon_thread.start()

    keyboard_thread = threading.Thread(target=keyboard_server)
    keyboard_thread.start()

    joystick_thread = threading.Thread(target=joystick_controller)
    joystick_thread.start()

    # logging_thread = threading.Thread(target=logging)
    # logging_thread.start()

    ## Camera node
    # rospy.Subscriber('/front_camera/zed2/zed_node/left/image_rect_color', Image, left_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/right/image_rect_color', Image, right_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/velodyne_points', PointCloud2, lidar_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/imu/data', Imu, imu_callback, queue_size = 1)
    # rospy.Subscriber('/vesc/odom', Odometry, vesc_odom_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/odom', Odometry, zed2_odom_callback, queue_size = 1)
    

    rospy.spin()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        # vicon_thread.join()
        keyboard_thread.join()
        joystick_thread.join()
        # logging_thread.join()
