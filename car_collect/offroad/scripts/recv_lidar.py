import rospy
import socket
from sensor_msgs.msg import NavSatFix, Joy, Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pt2
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
import ros_numpy

bridge = CvBridge()

DATA_DIR = '/home/cmu/catkin_ws/offroad/data'
timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = 'lidar-' + timestr + '.json'

image_dataset = dict(lidar=[])

def lidar_callback(data):
    # data_arr = pt2.read_points(data, skip_nans=True)
    # import pdb; pdb.set_trace()

    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
    global image_dataset
    data = xyz_array.tolist()
    image_dataset['lidar'].append(data)
    print("lidar")
    if len(image_dataset['lidar']) == 5:
        print("save")
        with open(os.path.join(DATA_DIR, file_name), "w") as f:
            json.dump(image_dataset, f)


if __name__ == '__main__':
    rospy.init_node('ImageSaver', anonymous=True)
    rospy.Subscriber('/velodyne_points', PointCloud2, lidar_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/right/image_rect_color', Image, right_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/depth/depth_registered/compressedDepth', CompressedImage, depth_callback, queue_size = 1)

    rospy.spin()
