import pickle
import os
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import imageio
from rich.progress import track
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading

import time
file = "../data"
log_dir = "data-20231219-115048"
lidar_data = []
for log_num in track(range(0, 1)):
    try:
        with open(os.path.join(file, log_dir, f"log{log_num}.pkl"), "rb") as f:
            data = pickle.load(f)
        for d_t in data:
            # import pdb; pdb.set_trace()
            # time = d_t['time']
            # import pdb; pdb.set_trace()
            # img = d_t["right_rgb"]
            lidar = np.array(d_t['lidar'])
            lidar_data.append(lidar)
            # imgs.append(img)
    except:
        print("error", log_num)

# Create a figure

# lidar_data = np.array(lidar_data)   


vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)

# # geometry is the point cloud used in your animaiton
geometry = o3d.geometry.PointCloud()
geometry.points = o3d.utility.Vector3dVector(lidar_data[0])
vis.add_geometry(geometry)

for i in track(range(len(lidar_data)):
    # now modify the points of your geometry
    # you can use whatever method suits you best, this is just an example
    geometry.points = o3d.utility.Vector3dVector(lidar_data[i])
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    image_filename = f"tmp/lidar_{i}.png"  # Filename for the image
    vis.capture_screen_image(image_filename)
    # time.sleep(0.001)

    
# time_step = 0
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(lidar_data[time_step, :, 0],  # X coordinates
#            lidar_data[time_step, :, 1],  # Y coordinates
#            lidar_data[time_step, :, 2],
#            s=0.01)  # Z coordinates
# ax.view_init(elev=73, azim=-45) 
# ax.set_xlim([-50, 50])
# ax.set_ylim([-50, 50])
# ax.grid(False)
# ax.set_axis_off()
# # ax.set_zlim([0, 10])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# def update(frame):
#     ax.clear()
#     ax.scatter(lidar_data[frame, :, 0], lidar_data[frame, :, 1], lidar_data[frame, :, 2], s=0.01)
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')

# ani = FuncAnimation(fig, update, frames=range(1), interval=100)
# plt.show()


# Add 3D axes to the figure
# for e in range(0, 360, 10):
#     for azim in range(0, 360, 10):
#         # time.sleep(1)
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(lidar[:, 0], lidar[:, 1], lidar[:, 2], s=0.01)
#         ax.view_init(elev=e, azim=azim) 
#         fig.savefig(f"tmp/{e}_{azim}.png")
# import pdb; pdb.set_trace()
# imageio.mimsave('output.gif', imgs, duration=len(imgs)*0.01)
# print("output done.")
# import pdb; pdb.set_trace()

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(lidar)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])