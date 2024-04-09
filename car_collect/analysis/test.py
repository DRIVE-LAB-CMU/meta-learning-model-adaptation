import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
from offroad.utils import load_state, lla_to_utm, lla_to_xy
import json
import ipywidgets as widgets
from car_dynamics.models_torch.dataset import clean_random_data
from car_dynamics.analysis import pos2vel_savgol, calc_delta, interpolate, rotate_point, calc_delta_v
from car_dynamics.modules import denormalize_angle_sequence
from ipywidgets import interact

df = pd.read_csv('/Users/wenlixiao/Dropbox/School/Graduate/LeCAR/safe-learning-control/playground/offroad/data/dataset/vicon-data-clean/vicon-circle-data-20240306-180059-1.csv')

xs = df['pos_x'].to_numpy()
dt = 0.05
ys = df['pos_y'].to_numpy()
thetas = df['yaw'].to_numpy()
dyaws = np.arctan2(ys[1:]-ys[:-1], xs[1:]-xs[:-1]) - thetas[:-1]
speeds = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)/dt
plt.plot(dyaws*(speeds>0.5))
plt.show()