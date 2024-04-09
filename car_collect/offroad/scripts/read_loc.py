import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import imageio
from rich.progress import track
import matplotlib
import threading
from PIL import Image
from offroad.utils.gps import lla_to_utm, determine_utm_zone
file = "../data"
log_dir = "data-20231223-155625"
# imgs = []
num_log = 0

if not os.path.exists(os.path.join('tmp', log_dir)):
    os.makedirs(os.path.join('tmp', log_dir))

def save_img(path, image):
    image.save(path)
    print("saved", path)
    
    
lidar_data = []
loc_data = []

for log_num in track(range(0, 1000)): #33
    print(log_num)
    try:
        with open(os.path.join(file, log_dir, f"log{log_num}.pkl"), "rb") as f:
            data = pickle.load(f)
            
            for d_t in data:
                # import pdb; pdb.set_trace()
                # time = d_t['time']
                # import pdb; pdb.set_trace()
                print(d_t['time'])
                # print(np.array(d_t['gps_loc']))
                num_log += 1
    except:
        break
    
