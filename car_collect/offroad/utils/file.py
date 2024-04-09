import os
import pickle
from rich.progress import track
from .gps import lla_to_utm, determine_utm_zone
from termcolor import colored
import numpy as np
from car_dynamics.analysis import quaternion_to_euler

def load_state(path, log_range: list, orientation_provider='imu', verbose=0):
    """ Given absolute path to the log directory, load the state (position, yaw, action) of the car.
        orientation_provider: [ imu or zed2_odom or vesc_odom ]
    """
    log_num = 0
    obs = []
    p = []
    p_vicon = []
    gps = []
    pdot = []
    yaw_imu = []
    yaw_vesc = []
    yaw_vesc_ang_vel = []
    yaw_zed2 = []
    yaw_zed2_ang_vel = []
    yaw_vicon = []
    actions = []
    mppi_actions = []
    mppi_sample_actions = []
    targets = []
    mppi_traj = []
    is_recover = []
    t = []
    date = []
    for log_num in track(range(log_range[0], log_range[1]), disable=True if verbose == 0 else False):
        with open(os.path.join(path, f"log{log_num}.pkl"), "rb") as f:
            data = pickle.load(f)
        log_num += 1
        for d_t in data:
            # print(d_t)
            t.append(d_t['time'])
            if 'gps' in d_t:
                gps.append(d_t['gps'])
            if 'date' in d_t:
                date.append(d_t['date'])
            # if 'gps_loc' in d_t:
            #     p.append(d_t['gps_loc'][:2])
            # if 'gps_vel' in d_t:
            #     pdot.append(d_t['gps_vel'][:2])
            if 'vicon_loc' in d_t and d_t['vicon_loc'] is not None:
                p_vicon.append(d_t['vicon_loc'][:2])
                yaw_vicon.append(d_t['vicon_loc'][2])
                
            if 'mppi_actions' in d_t:
                mppi_actions.append(d_t['mppi_actions'])
                
            if 'targets' in d_t:
                targets.append(d_t['targets'])
            
            if 'mppi_traj' in d_t:
                mppi_traj.append(d_t['mppi_traj'])
                
            if 'mppi_sample_actions' in d_t:
                mppi_sample_actions.append(d_t['mppi_sample_actions'])
            
            if 'recover' in d_t:
                is_recover.append(d_t['recover'])
                
            if 'imu' in d_t:  
                if d_t['imu'] is not None:
                    yaw_imu.append(quaternion_to_euler(d_t['imu']['orientation'])[-1])
                    yaw_zed2_ang_vel.append(d_t['imu']['angular_vel'][-1])
            if 'vesc_odom' in d_t:
                if d_t['vesc_odom'] is not None:      
                    yaw_vesc.append(quaternion_to_euler(d_t['vesc_odom']['orientation'])[-1])
                    yaw_vesc_ang_vel.append(d_t['vesc_odom']['angular_vel'][-1])
            if 'zed2_odom' in d_t:
                if d_t['zed2_odom'] is not None:
                    yaw_zed2.append(quaternion_to_euler(d_t['zed2_odom']['orientation'])[-1])
            actions.append([d_t['throttle'], d_t['steering']])
            obs.append(d_t['obs'])

    if verbose == 1:
        print(colored(f"[INFO] In total {log_range[1] - log_range[0]} logs", 'green'))
    
    gps_locations = np.array(p)
    if len(gps_locations) != 0:
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
        p = utm_coords
    p_vicon = np.array(p_vicon)
    
    # pdot = np.array(pdot)
    t = np.array(t)
    yaw_imu = np.array(yaw_imu)
    yaw_zed2 = np.array(yaw_zed2)
    yaw_zed2_ang_vel = np.array(yaw_zed2_ang_vel)
    yaw_vesc = np.array(yaw_vesc)
    yaw_vesc_ang_vel = np.array(yaw_vesc_ang_vel)
    yaw_vicon = np.array(yaw_vicon)
    actions = np.array(actions)
    obs = np.array(obs)
    mppi_actions = np.array(mppi_actions)
    # targets = np.array(targets)
    mppi_sample_actions = np.array(mppi_sample_actions)
    mppi_traj = np.array(mppi_traj)
    is_recover = np.array(is_recover)
    
    return t, \
            {'gps': gps, 'vicon': p_vicon, 'obs': obs, 'date': date}, \
            {'imu': yaw_imu, 'vesc': yaw_vesc, 'zed2': yaw_zed2, 'vicon': yaw_vicon, 'zed2_ang_vel': yaw_zed2_ang_vel, 'vesc_ang_vel': yaw_vesc_ang_vel,}, \
            actions, \
            {'mppi_actions': mppi_actions, 'targets': targets, 'mppi_traj': mppi_traj,
             'mppi_sample_actions': mppi_sample_actions,
             'is_recover': is_recover} #controller info
    