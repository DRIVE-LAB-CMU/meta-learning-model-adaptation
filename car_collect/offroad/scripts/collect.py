import rospy
import socket
from sensor_msgs.msg import NavSatFix, Joy
from fusion_engine_client.parsers import FusionEngineDecoder
from fusion_engine_client.messages import PoseMessage
import threading
import time
import json
import os

DATA_DIR = '/home/cmu/catkin_ws/offroad/data'
timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = 'data-' + timestr + '.json'

TCP_IP = ""
TCP_PORT = 15213

msg_decoder = FusionEngineDecoder(
            max_payload_len_bytes=4096, warn_on_unrecognized=False, return_bytes=True)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.connect((TCP_IP, TCP_PORT))

BUFFER_SIZE = 1024

gps_loc = ()
gps_vel = ()

num_data = 0
dataset = dict(time=[], state=[], action=[])

def update_gps():
    while True:
        # print("receiving ...")
        data = s.recv(BUFFER_SIZE)
        # data = data.decode("ISO-8859-1")
        if not data:
            break
        try:
            data = msg_decoder.on_data(data)[0][1]
            if type(data) is PoseMessage:
                global gps_loc, gps_vel
                print(f"loc: {data.position_std_enu_m}\nvel: {data.velocity_std_body_mps}")
                gps_loc = data.position_std_enu_m
                gps_vel = data.velocity_std_body_mps

        except:
            print(type(data))

def joystick_callback(data):
    global gps_loc, gps_vel, num_data, dataset
    throttle = data.axes[1]
    steering = data.axes[2]
    toggle = data.buttons[6]
    print(f"throttle: {throttle}, steering: {steering}, toggle: {toggle}")
    print(f"loc: {gps_loc}, vel: {gps_vel}")
    dataset['state'].append(dict(loc=list(gps_loc), vel=list(gps_vel)))
    dataset['action'].append(dict(throttle=throttle, steering=steering, toggle=toggle))
    dataset['time'].append(time.time())
    num_data += 1
    if num_data % 100 == 0:
        with open(os.path.join(DATA_DIR, file_name), "w") as f:
            json.dump(dataset, f)


def joystick_listener():
    rospy.init_node("offroad", anonymous=True)
    rospy.Subscriber("/vesc/joy", Joy, joystick_callback)
    rospy.spin()

if __name__ == '__main__':
    gps_thread = threading.Thread(target=update_gps)
    gps_thread.start()
    joystick_listener()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        gps_thread.join()
