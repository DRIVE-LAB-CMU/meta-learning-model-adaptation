import rospy
import socket
from sensor_msgs.msg import NavSatFix, Joy, Image, CompressedImage
from fusion_engine_client.parsers import FusionEngineDecoder
from fusion_engine_client.messages import PoseMessage
import threading
import time
import json
import os
import struct
import numpy as np
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

DATA_DIR = '/home/cmu/catkin_ws/offroad/data'
timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = 'image-' + timestr + '.json'

image_dataset = dict(left=[], right=[], depth=[])

def left_rgb_callback(data):
    # image_cv = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    np_arr = np.frombuffer(data.data, dtype=np.uint8)
    # reshape(image_data.height, image_data.width, -1)
    # print(np_arr.shape)
    data = np_arr.tolist()
    print("left")
    global image_dataset
    image_dataset['left'].append(data)
    if len(image_dataset['left']) == 5:
        with open(os.path.join(DATA_DIR, file_name), "w") as f:
            json.dump(image_dataset, f)

def right_rgb_callback(data):
    image_cv = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    data = image_cv.asarray().tolist()

    print("right")
    global image_dataset
    image_dataset['right'].append(data)

    


def depth_callback(data):
    # data = bridge.imgmsg_to_cv2(data, "bgr8").tolist()
    header_size = data.header.__sizeof__()
    raw_data = data.data[header_size:]
    raw_header = data.data[:header_size]
    [compfmt, depthQuantA, depthQuantB] = struct.unpack(raw_header)
    print(compfmt, depthQuantA, depthQuantB)
    np_arr = np.fromstring(raw_data, np.uint8)
    print(np_arr.shape)
    data = np_arr.tolist()
    # print(type(data))
    # data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # print(data)

    print("depth")
    global image_dataset
    image_dataset['depth'].append(data)

    if len(image_dataset['depth']) % 10 == 0:
        with open(os.path.join(DATA_DIR, file_name), "w") as f:
            json.dump(image_dataset, f)

if __name__ == '__main__':
    rospy.init_node('ImageSaver', anonymous=True)
    rospy.Subscriber('/front_camera/zed2/zed_node/left/image_rect_color', Image, left_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/right/image_rect_color', Image, right_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/depth/depth_registered/compressedDepth', CompressedImage, depth_callback, queue_size = 1)

    rospy.spin()
