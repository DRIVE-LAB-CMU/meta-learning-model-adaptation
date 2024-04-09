import rospy
import socket
from sensor_msgs.msg import NavSatFix, Joy, Image, CompressedImage, Imu
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
from nav_msgs.msg import Odometry


bridge = CvBridge()

DATA_DIR = '/home/cmu/catkin_ws/offroad/data/'
timestr = time.strftime("%Y%m%d-%H%M%S")
file_name = 'imu-' + timestr + '.json'

imu_dataset = dict(imu=[], zed2_odom=[], vesc_odom=[])

def imu_callback(data):
    global imu_dataset
    orientation = np.array([data.orientation.x,
                            data.orientation.y,
                            data.orientation.z,
                            data.orientation.w]).tolist()

    # Extract the angular velocity
    angular_velocity = np.array([data.angular_velocity.x,
                                 data.angular_velocity.y,
                                 data.angular_velocity.z]).tolist()

    # Extract the linear acceleration
    linear_acceleration = np.array([data.linear_acceleration.x,
                                    data.linear_acceleration.y,
                                    data.linear_acceleration.z]).tolist()
    
    imu_dataset['imu'].append(dict(orientation=orientation,
                                   angular_vel=angular_velocity,
                                   linear_acc=linear_acceleration))
    
    if len(imu_dataset['zed2_odom']) == 20:
        # print("save")
        print("test", imu_dataset["zed2_odom"])
        with open(os.path.join(DATA_DIR, file_name), "w") as f:
            json.dump(imu_dataset, f)



def vesc_odom_callback(odometry_data):
    global imu_dataset

    position = np.array([odometry_data.pose.pose.position.x,
                         odometry_data.pose.pose.position.y,
                         odometry_data.pose.pose.position.z]).tolist()

    orientation = np.array([odometry_data.pose.pose.orientation.x,
                            odometry_data.pose.pose.orientation.y,
                            odometry_data.pose.pose.orientation.z,
                            odometry_data.pose.pose.orientation.w]).tolist()

    # Extract the twist data (linear and angular velocity)
    linear_velocity = np.array([odometry_data.twist.twist.linear.x,
                                odometry_data.twist.twist.linear.y,
                                odometry_data.twist.twist.linear.z]).tolist()

    angular_velocity = np.array([odometry_data.twist.twist.angular.x,
                                 odometry_data.twist.twist.angular.y,
                                 odometry_data.twist.twist.angular.z]).tolist()
    
    imu_dataset['vesc_odom'].append(dict(position=position,
                                         orientation=orientation,
                                         linear_vel=linear_velocity,
                                         angular_vel=angular_velocity)
                                    )
def zed2_odom_callback(odometry_data):
    global imu_dataset

    print("zed2")

    position = np.array([odometry_data.pose.pose.position.x,
                         odometry_data.pose.pose.position.y,
                         odometry_data.pose.pose.position.z]).tolist()

    orientation = np.array([odometry_data.pose.pose.orientation.x,
                            odometry_data.pose.pose.orientation.y,
                            odometry_data.pose.pose.orientation.z,
                            odometry_data.pose.pose.orientation.w]).tolist()

    # Extract the twist data (linear and angular velocity)
    linear_velocity = np.array([odometry_data.twist.twist.linear.x,
                                odometry_data.twist.twist.linear.y,
                                odometry_data.twist.twist.linear.z]).tolist()

    angular_velocity = np.array([odometry_data.twist.twist.angular.x,
                                 odometry_data.twist.twist.angular.y,
                                 odometry_data.twist.twist.angular.z]).tolist()
    
    imu_dataset['zed2_odom'].append(dict(position=position,
                                         orientation=orientation,
                                         linear_vel=linear_velocity,
                                         angular_vel=angular_velocity)
                                    )


if __name__ == '__main__':
    rospy.init_node('ImageSaver', anonymous=True)
    rospy.Subscriber('/front_camera/zed2/zed_node/imu/data', Imu, imu_callback, queue_size = 1)
    rospy.Subscriber('/vesc/odom', Odometry, vesc_odom_callback, queue_size = 1)
    rospy.Subscriber('/front_camera/zed2/zed_node/odom', Odometry, zed2_odom_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/right/image_rect_color', Image, right_rgb_callback, queue_size = 1)
    # rospy.Subscriber('/front_camera/zed2/zed_node/depth/depth_registered/compressedDepth', CompressedImage, depth_callback, queue_size = 1)

    rospy.spin()
