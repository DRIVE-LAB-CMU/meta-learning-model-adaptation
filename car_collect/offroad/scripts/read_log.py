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
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', 'data-20231219-122003', 'log dir.')

def main(argv):
    file = "../data"
    log_dir = FLAGS.logdir
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
                left_img = np.array(d_t["left_rgb"])   
                assert left_img.shape == (188, 336, 3)
                
                right_img = np.array(d_t["right_rgb"])   
                assert left_img.shape == (188, 336, 3)
                
                left_image = Image.fromarray(left_img.astype('uint8'))
                # image.save(f"tmp/img_left_{num_log}.png")
                left_thead = threading.Thread(target=save_img, args=(f"tmp/{log_dir}/img_left_{num_log}.png", left_image))
                
                right_image = Image.fromarray(right_img.astype('uint8'))
                right_thread = threading.Thread(target=save_img, args=(f"tmp/{log_dir}/img_right_{num_log}.png", right_image))
                
                left_thead.start()
                right_thread.start()
                left_thead.join()
                right_thread.join()
                
                lidar_data.append(np.array(d_t['lidar']))
                loc_data.append(np.array(d_t['gps_loc']))
                # print(np.array(d_t['gps_loc']))
                num_log += 1
        except:
            break
        

    # lidar_data = np.array(lidar_data)
    loc_data = np.array(loc_data)

    # Example usage
    gps_locations = loc_data[:, :2]

    # Taking the first location as the reference point
    reference_point_utm = lla_to_utm(gps_locations[0][0], gps_locations[0][1])
    utm_coords = []

    for lat, lon in gps_locations:
        if np.isnan(lon):
            continue
        # print(lon)
        x, y = lla_to_utm(lat, lon)
        utm_coords.append([x, y])

    # Subtracting the reference point to get relative coordinates
    utm_coords = np.array(utm_coords) - reference_point_utm

    # print(utm_coords)

    def plot_loc(path, loc):
        fig, ax = plt.subplots()
        ax.scatter(utm_coords[:, 0], utm_coords[:, 1], s=20, c='g')
        ax.scatter(loc[0], loc[1], s=20, c='r')
        ax.axis('off')
        # save figure to path
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        print("saved", path)
        
    for i, loc in enumerate(utm_coords):
        loc_thread = threading.Thread(target=plot_loc, args=(f"tmp/{log_dir}/loc_{i}.png", loc))
        loc_thread.start()
        loc_thread.join()
        

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    # # geometry is the point cloud used in your animaiton
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(lidar_data[0])
    vis.add_geometry(geometry)

    def trim_white_margin(image_path):
        image = Image.open(image_path)
        # Convert the image to a numpy array
        np_image = np.array(image)
        # Find all non-white pixels
        non_white_pixels = np.where(
            (np_image[:, :, 0] != 255) |
            (np_image[:, :, 1] != 255) |
            (np_image[:, :, 2] != 255)
        )
        # Get the bounding box of non-white pixels
        x_min, x_max = non_white_pixels[1].min(), non_white_pixels[1].max()
        y_min, y_max = non_white_pixels[0].min(), non_white_pixels[0].max()
        # Crop the image
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image.save(image_path)
        
    for i in track(range(len(lidar_data))):
        # now modify the points of your geometry
        # you can use whatever method suits you best, this is just an example
        geometry.points = o3d.utility.Vector3dVector(lidar_data[i])
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        image_filename = f"tmp/{log_dir}/lidar_{i}.png"  # Filename for the image
        vis.capture_screen_image(image_filename)
        trim_white_margin(image_filename)
        # time.sleep(0.001)
        
if __name__ == '__main__':
    app.run(main)