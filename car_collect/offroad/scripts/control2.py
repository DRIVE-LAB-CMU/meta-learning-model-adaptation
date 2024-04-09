import rospy
import socket
from sensor_msgs.msg import NavSatFix, Joy
from fusion_engine_client.parsers import FusionEngineDecoder
from fusion_engine_client.messages import PoseMessage
import threading
import time
import json
import os
import struct

DATA_DIR = '/home/cmu/catkin_ws/offroad/data'
timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = 'data-' + timestr + '.json'

TCP_IP = ""
TCP_PORT = 15213

msg_decoder = FusionEngineDecoder(
            max_payload_len_bytes=4096, warn_on_unrecognized=False, return_bytes=True)

GPSConnect = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
GPSConnect.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
GPSConnect.connect((TCP_IP, TCP_PORT))

KeyboardConnect = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)







BUFFER_SIZE = 8

gps_loc = ()
gps_vel = ()
throttle = 0.0
steering = 0.0
num_data = 0
dataset = dict(time=[], state=[], action=[])


def keyboard_server():
    TCP_IP = ""  # Replace with the IP address of your laptop
    TCP_PORT = 15214  # Replace with the desired port number

    KeyboardConnect.bind((TCP_IP, TCP_PORT))
    # KeyboardConnect.listen(1)

    # print("Waiting for keyboard connection...")
    # conn, addr = KeyboardConnect.accept()
    # print("Keyboard connected!")

    while True:
        global throttle, steering
        data, addr = KeyboardConnect.recvfrom(BUFFER_SIZE)
        if not data:
            break
        unpacked_data = struct.unpack('ff', data)
        throttle = unpacked_data[0]
        steering = unpacked_data[1]
        # print(f"data: {unpacked_data}")
        # logging()



def joystick_listener():
    while True:
        global pub, throttle, steering 
        now = rospy.Time.now()
        cmd = Joy()
        cmd.header.stamp = now
        cmd.buttons = [0,0,0,0,0,0,1,0,0,0,0]

        cmd.axes = [0.,throttle,steering,0.,0.,0.]
        # cmd.axes = [0.,forward_speed,(steering/max_steering),steering/max_steering,0.,0.]
        # print("REAL COMMAND", cmd)
        pub.publish(cmd)
        print("pub", throttle, steering)
        if rospy.is_shutdown():
                raise Exception("[ERROR] ROSPY SHUTDOWN!")
        time.sleep(0.01)


if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True, disable_signals=True)
    global pub
    pub = rospy.Publisher('/vesc/joy',Joy)

    keyboard_thread = threading.Thread(target=keyboard_server)
    keyboard_thread.start()

    joystick_thread = threading.Thread(target=joystick_listener)
    joystick_thread.start()

    # logging_thread = threading.Thread(target=logging)
    # logging_thread.start()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        keyboard_thread.join()
        joystick_thread.join()
        # logging_thread.join()
