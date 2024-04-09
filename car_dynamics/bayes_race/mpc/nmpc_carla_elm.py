#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA auto control.

"""
from __future__ import print_function

"""	Nonlinear MPC using true Dynamic bicycle model.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import matplotlib.pyplot as plt
try:
    sys.path.append('/home/dvij/bayesrace')
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Imports from Bayesrace ----------------------------------------------------
# ==============================================================================
import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# from carla import Vector3d
from bayes_race.params import ORCA, CarlaParams
from bayes_race.models import Dynamic, Kinematic6
from bayes_race.tracks import ETHZ, CarlaRace
from bayes_race.mpc.planner import ConstantSpeedCarla
from bayes_race.gp.utils import loadGPModel, loadGPModelVars, loadMLPModel, loadTorchModel, loadTorchModelEq, loadTorchModelImplicit
from bayes_race.mpc.gpmpc_carla import setupNLP
import torch
import random
import os
from bayes_race.utils import Projection, Spline, Spline2D
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# ==============================================================================
# -- Imports from carla --------------------------------------------------------
# ==============================================================================


import carla

import argparse
import logging
import math
from scipy.interpolate import interp1d
from carla_utils import *

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
TRACK_CONS = False
ERROR_CORR = True
SAVE_VIDEO = False
ITERS_EACH_STEP = 50
LOAD_MODEL = True
ACT_FN = 'relu'
UPDATE_PLOTS = True
mu_init = 1.
#####################################################################
# default settings

v_factor = 1.52
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([0, 0])/43
COST_P = np.diag([1, 1])/15
COST_R = np.diag([.05, 1])
MANUAL_CONTROL = False
RUN_NO = 0
BUFFER_LEN = 2000
RUN_FOLDER = 'RUN_ONLINE_' + str(RUN_NO) + '_' + str(BUFFER_LEN) + '/'
SIM_TIME = 175

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''

torch.manual_seed(3)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)

alpha_f_distribution_y = np.zeros(2000)
alpha_f_distribution_x = np.arange(-1.,1.,2./2000)

alpha_r_distribution_y = np.zeros(2000)
alpha_r_distribution_x = np.arange(-1.,1.,2./2000)

class DynamicModel(torch.nn.Module):
	def __init__(self, model, deltat = 0.01):
		"""
		In the constructor we instantiate four parameters and assign them as
		member parameters.
		"""
		super().__init__()
		if ACT_FN == 'relu' :
			self.act = torch.nn.ReLU()
		elif ACT_FN =='tanh' :
			self.act = torch.nn.Tanh()
		elif ACT_FN =='lrelu' :
			self.act = torch.nn.LeakyReLU()
		elif ACT_FN =='sigmoid' :
			self.act = torch.nn.Sigmoid()
		
		self.Rx = torch.nn.Sequential(torch.nn.Linear(1,1).to(torch.float64))
		# self.Rx[0].weight.data.fill_(0.)
		# self.Rx[0].bias.data = torch.tensor([0.]).to(torch.float64)
		
		self.Ry = torch.nn.Sequential(torch.nn.Linear(1,12).to(torch.float64), \
					self.act, \
					torch.nn.Linear(12,1).to(torch.float64))
		self.Ry[0].weight.data.fill_(1.)
		self.Ry[0].bias.data = -torch.arange(0.,0.4,(.4)/12.).to(torch.float64)
		self.Ry[2].weight.data.fill_(0.)
		self.Ry[2].bias.data = torch.zeros(1).to(torch.float64)
		
		self.Fy = torch.nn.Sequential(torch.nn.Linear(1,12).to(torch.float64), \
					self.act, \
					torch.nn.Linear(12,1).to(torch.float64))
		
		self.Fy[0].weight.data.fill_(1.)
		self.Fy[0].bias.data = -torch.arange(0.,0.4,(.4)/12.).to(torch.float64)
		self.Fy[2].weight.data.fill_(0.)
		self.Fy[2].bias.data = torch.zeros(1).to(torch.float64)
		
		self.deltat = deltat
		self.model = model

	def forward(self, x, debug=False):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		# print(x.shape)
		# out = X
		deltatheta = x[:,1]
		roll = x[:,6]
		pitch = x[:,7]
		theta = x[:,2]
		pwm = x[:,0]
		out = torch.zeros_like(x[:,3:6])
		# print(out)
		for i in range(2) :
			vx = (x[:,3] + out[:,0]).unsqueeze(1)
			vy = x[:,4] + out[:,1]
			w = x[:,5] + out[:,2]
			alpha_f = (theta - torch.atan2(w*self.model.lf+vy,vx[:,0])).unsqueeze(1)
			alpha_r = torch.atan2(w*self.model.lr-vy,vx[:,0]).unsqueeze(1)
			if debug :
				for alpha in alpha_f[:,0] :
					alpha_f_distribution_y[int((alpha+1.)*1000)] += 1
				for alpha in alpha_r[:,0] :
					alpha_r_distribution_y[int((alpha+1.)*1000)] += 1
			# print(torch.max(alpha_r))
			Ffy = self.Fy(alpha_f)[:,0]*(alpha_f[:,0]>0.) - self.Fy(-alpha_f)[:,0]*(alpha_f[:,0]<=0.)
			Fry = self.Ry(alpha_r)[:,0]*(alpha_r[:,0]>0.) - self.Ry(-alpha_r)[:,0]*(alpha_r[:,0]<=0.)
			Frx = self.Rx(vx**2)[:,0]/self.model.mass + (pwm>0)*self.model.Cm1*pwm/self.model.mass \
				+ (pwm<=0)*self.model.Cm2*pwm/self.model.mass
			
			vx_dot = (Frx-Ffy*torch.sin(theta)+vy*w-9.8*torch.sin(pitch))
			vy_dot = (Fry+Ffy*torch.cos(theta)-vx[:,0]*w)
			w_dot = self.model.mass*(Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/self.model.Iz
			out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat
		out2 = (out)
		return out2

#####################################################################
# load vehicle parameters

params = CarlaParams(control='pwm')
model = Dynamic(**params)
model_kin = Kinematic6(**params)

#####################################################################
# Define change of friction params over time 

fric_factor = 1.

def update_fric_factor(t,t_start=44.,half_time=100.) :
    global fric_factor
    if t > t_start :
        fric_factor = 1. - .32*(t-t_start)/100

def change_params(t,player,t_start=2.,half_time=100.) :
    global fric_factor
    if t > t_start and t<t_start+0.1:
        fric_factor = 1. - .5*(t-t_start)/100
        physics_control = player.get_physics_control()
        wheels = physics_control.wheels
        wheels[0].tire_friction = 2.75*fric_factor
        wheels[0].lat_stiff_value = 40.*fric_factor
        wheels[1].tire_friction = 2.75*fric_factor
        wheels[1].lat_stiff_value = 40.*fric_factor
        wheels[2].tire_friction = 2.75*fric_factor
        wheels[2].lat_stiff_value = 40.*fric_factor
        wheels[3].tire_friction = 2.75*fric_factor
        wheels[3].lat_stiff_value = 40.*fric_factor
        physics_control.wheels = wheels
        
        print("Modified")
        return physics_control
    else :
        return None

def wear_tear(alpha_f,alpha_r,delta,yaw,params,Iz=345201.0) :
    Ff = (fric_factor-1)*params['Df']*math.sin(params['Cf']*math.atan(params['Bf']*alpha_f))
    Fr = (fric_factor-1)*params['Dr']*math.sin(params['Cr']*math.atan(params['Br']*alpha_r))
    force_y_ = math.cos(delta)*(fric_factor-1)*params['Df']*math.sin(params['Cf']*math.atan(params['Bf']*alpha_f)) \
        + (fric_factor-1)*params['Dr']*math.sin(params['Cr']*math.atan(params['Br']*alpha_r))
    force_x_ = -math.sin(delta)*(fric_factor-1)*params['Df']*math.sin(params['Cf']*math.atan(params['Bf']*alpha_f))
    torque_z = (Iz/params['mass'])*(Ff*params["lf"]*math.cos(delta)-Fr*params['lr'])*180./math.pi
    force_x = force_x_*math.cos(yaw) - force_y_*math.sin(yaw)
    force_y = force_x_*math.sin(yaw) + force_y_*math.cos(yaw)
    return carla.Vector3D(force_x,force_y,0.), carla.Vector3D(0.,0.,torque_z)

#####################################################################
# load track

TRACK_NAME = 'Carla'
track = CarlaRace(reference='optimal')
a = 300
b = 331
c = 382
factor = 0.83
for i in range(len(track.v_raceline)) :
    # print(i,track.x_raceline[i],track.v_raceline[i])
    if i>=a and i<=b :
        track.v_raceline[i] *= 1 + (factor-1)*(i-a)/(b-a)
    if i>=b and i<=c :
        track.v_raceline[i] *= factor + (1-factor)*(i-b)/(c-b)
track.spline_v = Spline(track.spline.s, track.v_raceline)
for i in range(len(track.v_raceline)) :
    print(i,track.x_raceline[i],track.v_raceline[i])

# exit(0)

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# load mlp models

MODEL_PATH = '../gp/orca/semi_mlp-v1.pickle'
model_ = DynamicModel(model)
if LOAD_MODEL :
	model_.load_state_dict(torch.load(MODEL_PATH))

print(model_.Rx[0].weight)
print(model_.Rx[0].bias)
model_Rx = loadTorchModelImplicit('Rx',model_.Rx)
model_Ry = loadTorchModelImplicit('Ry',model_.Ry)
model_Fy = loadTorchModelImplicit('Fy',model_.Fy)

models = {
	'Rx' : model_Rx,
	'Ry' : model_Ry,
	'Fy' : model_Fy,
	'act_fn' : ACT_FN
}
x_train = np.zeros((BUFFER_LEN,2+3+1))

optimizer = torch.optim.SGD(model_.parameters(), lr=8.,momentum=0.9)
loss_fn = torch.nn.MSELoss()

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, models, track, GP_EPS_LEN=BUFFER_LEN, 
	track_cons=TRACK_CONS, error_correction=ERROR_CORR)
#####################################################################
# define load_data (Need atleast 6 elements as it maintains a moving average of 6)

def load_data(data_dyn, VARIDX):
	y_all = (data_dyn['states'][:6,6:]-data_dyn['states'][:6,:-6])/6. #- data_kin['states'][:6,1:N_SAMPLES+1]
	# print(y_all)
	x = np.concatenate([
		data_dyn['inputs'][:,3:-2].T,
		data_dyn['inputs'][1,3:-2].reshape(1,-1).T,
		data_dyn['states'][3:6,3:-3].T,
		data_dyn['roll_pitch'][:,3:-3].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	return x, y

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states+1, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
inputs_kin = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

data_dyn = {}
data_kin = {}
data_dyn['time'] = time
data_kin['time'] = time

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))
states[:n_states,:] = np.expand_dims(x_init,axis=1)
_,projidx = track.project_fast(x_init[0],x_init[1],track.raceline)
states_kin = np.zeros([7,n_steps+1])
states_kin[:,0] = states[:,0]

# dynamic plot
H = .08
W = .04
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])


# dynamic plot
fig = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=0.8)
LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(xyproj[0], xyproj[1], 'g', marker='o', alpha=0.5, markersize=5, label="current position")
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5, label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-g', marker='o', markersize=1, lw=0.5, label="prediction")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

fig_speeds = plt.figure()
plt.grid(True)
ax2 = plt.gca()
Lnv_ref, = ax2.plot(0, 0, label='v (ref)')
Lnv, = ax2.plot(0, 0, label='v')
plt.xlim([0, SIM_TIME])
plt.ylim([0, 40])
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.legend()
plt.ion()
plt.show()

mus_fig = plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnDf_pred, = ax2.plot(0, 0, label='mu predicted(f)')
LnDr_pred, = ax2.plot(0, 0, label='mu predicted(r)')
LnFf_pred, = ax2.plot(0, 0, label='mu (GT)')
plt.xlim([0, SIM_TIME])
plt.ylim([0, params['Df']*1.5/(params['mass']*9.8)])
plt.xlabel('time [s]')
plt.ylabel('lateral force [N]')
plt.legend()
plt.ion()

plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnFry_pred, = ax2.plot(0, 0, label='Fry pred')
# LnFry_gt, = ax2.plot(0, 0, label='Fry gt')
LnFfy_pred, = ax2.plot(0, 0, label='Ffy pred')
# LnFfy_gt, = ax2.plot(0, 0, label='Ffy gt')
plt.xlim([-.3, .3])
plt.ylim([-12., 12.])
plt.xlabel('alpha [rad]')
plt.ylabel('force [N]')
plt.legend()
plt.ion()

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def dist(a,b) :
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def find_point_at_dist(pt,traj,dist,n_divs=100) :
    dists = ((traj[:,0]-pt.x)**2+(traj[:,1]-pt.y)**2)
    i = np.argmin(dists)
    N = len(dists)
    alphas = np.expand_dims(np.arange(0.,1.,1./n_divs),1)
    prev_i = (i-1)%N
    next_i = (i+1)%N
    traj_left = traj[prev_i:prev_i+1,:]*(1-alphas) + alphas*traj[i:i+1,:]
    traj_right = traj[i:i+1,:]*(1-alphas) + alphas*traj[next_i:next_i+1,:]
    traj_extra = np.concatenate((traj_left,traj_right),0)
    dists = ((traj_extra[:,0]-pt.x)**2+(traj_extra[:,1]-pt.y)**2)
    j = np.argmin(dists)
    if j < n_divs :
        factor = j/n_divs + dist
        pt_before = traj[(i-1+int(factor))%N,:]
        pt_after = traj[(i+int(factor))%N,:]
        factor -= int(factor)
        return pt_after*factor + pt_before*(1-factor)
    else :
        j -= n_divs
        factor = j/n_divs + dist
        pt_before = traj[(i+int(factor))%N,:]
        pt_after = traj[(i+1+int(factor))%N,:]
        factor -= int(factor)
        return pt_after*factor + pt_before*(1-factor)

def game_loop(args):
    global projidx, states, inputs, UPDATE_PLOTS
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    if not os.path.exists(RUN_FOLDER):
        os.makedirs(RUN_FOLDER)
    if not os.path.exists(RUN_FOLDER+'Video/'):
        os.makedirs(RUN_FOLDER+'Video/')
    # main simulation loop
    ref_speeds = []
    Drs = []
    Dfs = []
    Drs_pred = []
    Dfs_pred = []
    Df_init = model.Df
    Dr_init = model.Dr

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        world = client.load_world('Town07_Opt')
        if True : # Unload specific layers if required 
            # world.unload_map_layer(carla.MapLayer.Buildings)
            # world.unload_map_layer(carla.MapLayer.Decals)
            # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
            # world.unload_map_layer(carla.MapLayer.Particles)
            # world.unload_map_layer(carla.MapLayer.Props)
            # world.unload_map_layer(carla.MapLayer.Foliage)
            # world.unload_map_layer(carla.MapLayer.All)
            world.unload_map_layer(carla.MapLayer.StreetLights)
            world.unload_map_layer(carla.MapLayer.Walls)
        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.02
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        
        controller = KeyboardControl(world, args.autopilot)
        
        # traj_to_follow = np.loadtxt('carla_center.csv',delimiter=',')[:,:2]
        # track = CarlaRace(reference='optimal')
        # print(track.mus)
        x_raceline, y_raceline = track.x_raceline, track.y_raceline
        v_raceline = track.v_raceline
        traj_to_follow = np.array([x_raceline,y_raceline]).T
        # print("Saved")
        # exit(0)
        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        _control = carla.VehicleControl()
        integral = 0.
        run_data = []
        itr = 0
        dt = 0.02
        v_refs = []
        rolls = []
        pitches = []
        Ffs = []
        use_kinematic = False
        mu_pred = 1.
        while True:
            itr += 1
            Ffs.append(fric_factor)
            if itr == 1 :
                print(world.player.get_angular_velocity())
                sim_world.tick()
                # exit(0)
            update_fric_factor(itr*Ts)
            if args.sync:
                sim_world.tick()
                clock.tick_busy_loop(60)
                
                # physics_params = change_params(itr*Ts,world.player)
                # if physics_params is not None :   
                #     print(world.player.get_angular_velocity())
                #     print(world.player.get_velocity())
                #     temp_vel = world.player.get_velocity() 
                #     temp_ang_vel = world.player.get_angular_velocity() 
                #     world.player.apply_physics_control(physics_params)
                #     _control.steer = 0.
                #     _control.throttle = 0.
                #     _control.brake = 0.
                #     _control.manual_gear_shift = True
                #     _control.gear = 4
                #     world.player.apply_control(_control)
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     sim_world.tick()
                #     print(world.player.get_angular_velocity())
                #     print(world.player.get_velocity())
                #     world.player.add_angular_impulse(17260050.*temp_ang_vel)
                #     world.player.add_impulse(carla.Vector3D(temp_vel.x*params['mass'],temp_vel.y*params['mass'],temp_vel.z*params['mass']))
                #     sim_world.tick()
                #     # sim_world.tick()
                #     # sim_world.tick()
                #     # sim_world.tick()
                #     # sim_world.tick()
                #     # sim_world.tick()
                #     print(world.player.get_angular_velocity())
                #     print(world.player.get_velocity())
                #     clock.tick_busy_loop(60)
                #     exit(0)
                # physics_control = world.player.get_physics_control()
                # print(physics_control)

            if itr > n_steps-horizon :
                fig.savefig('path_carla_with.png')
                fig_speeds.savefig('speeds_carla_with.png')
                mus_fig.savefig('mus_carla.png')
                break
            
            location = world.player.get_location()
            velocity = world.player.get_velocity()
            vx = velocity.x
            vy = velocity.y
            acc = world.player.get_acceleration()
            w = world.player.get_angular_velocity().z*math.pi/180.
            yaw = world.player.get_transform().rotation.yaw*math.pi/180.
            roll = world.player.get_transform().rotation.roll
            pitch = world.player.get_transform().rotation.pitch
            rolls.append(roll*math.pi/180.)
            pitches.append(pitch*math.pi/180.)
            states[0,itr] = location.x
            states[1,itr] = location.y
            states[2,itr] = yaw
            states[3,itr] = vx*math.cos(yaw) + vy*math.sin(yaw)
            states[4,itr] = -vx*math.sin(yaw) + vy*math.cos(yaw)
            states[5,itr] = w
            states[6,itr] = inputs[1,itr-1]
            
            # print(states[:,itr+1])
            # states[6,itr+1] = (inputs[1,itr]-states[6,itr])/dt
            
            # load new experience into data_dyn and data_kin
            if itr > 300 : 
                start = tm.time()	
                min_ind = max(itr-BUFFER_LEN-1,200)
                data_dyn['states'] = states[:,min_ind:min(itr-1,BUFFER_LEN+1+min_ind)]
                data_dyn['inputs'] = inputs[:,min_ind:min(itr-2,BUFFER_LEN+min_ind)]
                data_dyn['roll_pitch'] = np.array([rolls[min_ind:min(itr-1,BUFFER_LEN+1+min_ind)],pitches[min_ind:min(itr-1,BUFFER_LEN+1+min_ind)]])
                end = tm.time()
                print("GP init time : ", end-start)
                
                y_trains = []
                for VARIDX in [3,4,5] :
                    x_train, y_train = load_data(data_dyn,VARIDX)
                    y_trains.append(torch.tensor(y_train))
                y_train = torch.cat(y_trains,axis=1)
                x_train = torch.tensor(x_train)
            # Fine-tune the model
            if itr > 2320 :
                start = tm.time()	
                for param in model_.Fy[0].parameters():
                    param.requires_grad = False
                for param in model_.Ry[0].parameters():
                    param.requires_grad = False
                for i in range(ITERS_EACH_STEP) :
                    # Zero your gradients for every batch!
                    optimizer.zero_grad()
                    outputs = model_(x_train[10:])
                    loss = loss_fn(outputs, y_train[10:])
                    loss.backward()
                    # Adjust learning weights
                    optimizer.step()
                end = tm.time()	
                print("Iter " + str(itr) + " loss : ", loss.item(), "time : ", end-start)

            uprev = inputs[:,itr-1]
            x0 = states[:,itr]
            x0[3] = max(5,x0[3])
            # planner based on BayesOpt
            # print("State : ", x0)
            xref, projidx, v_ref = ConstantSpeedCarla(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=mu_pred)
            # projidx_inner, x_inner, theta_inner, curv_inner = GetCBFSateInner(x0=x0[:3], track=track, projidx=projidx_inner)
            # projidx_outer, x_outer, theta_outer, curv_outer = GetCBFSateOuter(x0=x0[:3], track=track, projidx=projidx_outer)
            # print(xref)
            v_refs.append(v_ref)
            if projidx > track.raceline.shape[1]-5 :
                projidx = 0
            start = tm.time()
            umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev, use_kinematic=use_kinematic,models=model_)
            end = tm.time()
            # print(umpc)
            inputs[:,itr] = np.array([umpc[0,0], states[n_states,itr] + Ts*umpc[1,0]])
            print("iter: {}, cost: {:.5f}, time: {:.2f}".format(itr, fval, end-start))

            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            beta = math.atan2(velocity.y,velocity.x)
            dists = ((traj_to_follow[:,0]-location.x)**2+(traj_to_follow[:,1]-location.y)**2)
            i = np.argmin(dists)
            curr_point = np.array([location.x,location.y])
            dist = max(3,6*speed/20)
            target_point = find_point_at_dist(location,traj_to_follow,dist)
            nearest_point = find_point_at_dist(location,traj_to_follow,0)
            # hstates[:,0] = x0
            hstates2[:,0] = x0[:n_states]
            # print(umpc[1,:])
            # print(xmpc[3:6,:])

            for idh in range(horizon):
                # x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
                # hstates[:,idh+1] = x_next[:,-1]
                hstates2[:,idh+1] = xmpc[:n_states,idh+1]
            vec_near = nearest_point-curr_point
            print("Lateral error : ", math.sqrt(vec_near[0]**2+vec_near[1]**2))
            alpha_r = torch.tensor(np.arange(-.3,.3,0.001)).unsqueeze(1)
            Fry_pred = model_.Ry(alpha_r)[:,0].detach().numpy()
            alpha_r1 = torch.tensor(np.arange(-.22,.22,0.001)).unsqueeze(1)
            Fry_pred1 = model_.Ry(alpha_r1)[:,0].detach().numpy()
            if itr < 2000 :
                Drs_pred.append(mu_init)
            else :
                Drs_pred.append(np.max(Fry_pred1)/9.8)
            # Fry_true = params['Dr']*torch.sin(params['Cr']*torch.atan(params['Br']*alpha_r))
            alpha_f = torch.tensor(np.arange(-.3,.3,0.001)).unsqueeze(1)
            Ffy_pred = model_.Fy(alpha_f)[:,0].detach().numpy()
            alpha_f1 = torch.tensor(np.arange(-.22,.22,0.001)).unsqueeze(1)
            Ffy_pred1 = model_.Fy(alpha_f1)[:,0].detach().numpy()
            # Ffy_true = params['Df']*torch.sin(params['Cf']*torch.atan(params['Bf']*alpha_f))
            if itr < 2000 :
                Dfs_pred.append(mu_init)
            else :          
                Dfs_pred.append(np.max(Ffy_pred1)/9.8)
            mu_pred = min(1.,(np.max(Ffy_pred1)/9.8 + np.max(Fry_pred1)/9.8)/2.)
            LnDf_pred.set_xdata(time[:itr])
            LnDf_pred.set_ydata(Dfs_pred[:itr])
            LnFfy_pred.set_xdata(alpha_f)
            LnFfy_pred.set_ydata(Ffy_pred)
            
            LnFf_pred.set_xdata(time[:itr])
            LnFf_pred.set_ydata(Ffs[:itr])
            LnDr_pred.set_xdata(time[:itr])
            LnDr_pred.set_ydata(Drs_pred[:itr])
            LnFry_pred.set_xdata(alpha_r)
            LnFry_pred.set_ydata(Fry_pred)
            if UPDATE_PLOTS :
                # update plot
                LnS.set_xdata(states[0,:itr+1])
                LnS.set_ydata(states[1,:itr+1])

                LnR.set_xdata(xref[0,1:])
                LnR.set_ydata(xref[1,1:])

                LnP.set_xdata(states[0,itr])
                LnP.set_ydata(states[1,itr])
                
                LnH2.set_xdata(hstates2[0])
                LnH2.set_ydata(hstates2[1])
                
                # print(np.max(Ffy_pred)/9.8)
                # LnDf.set_xdata(time[:itr+1])
                # LnDf.set_ydata(Dfs[:itr+1])
                
                
                
                # print(np.max(Fry_pred)/9.8)
                # LnDr.set_xdata(time[:itr+1])
                # LnDr.set_ydata(Drs[:itr+1])
                
                
                
                Lnv_ref.set_xdata(time[:itr])
                Lnv_ref.set_ydata(v_refs)
                
                Lnv.set_xdata(time[:itr+1])
                Lnv.set_ydata(states[3,:itr+1])
                # LnFrx.set_xdata(time[:itr+1])
                # LnFrx.set_ydata(Frx[:itr+1])

                # LnFry.set_xdata(time[:itr+1])
                # LnFry.set_ydata(Fry[:itr+1])
                plt.pause(Ts/1000)
            
            # vec = target_point-nearest_point
            # x_ = vec[0]*math.cos(yaw) + vec[1]*math.sin(yaw)
            # y_ = vec[1]*math.cos(yaw) - vec[0]*math.sin(yaw)
            # print(x_,y_)
            if controller.toggle_update_plots(client, world, clock, args.sync) :
                UPDATE_PLOTS = not UPDATE_PLOTS

            if MANUAL_CONTROL :
                if controller.parse_events(client, world, clock, args.sync):
                    return
            else :
                steering = inputs[1,itr]
                v_perp = -vx*math.sin(yaw) + vy*math.cos(yaw)
                v = vx*math.cos(yaw) + vy*math.sin(yaw)
                alpha_f = steering - math.atan2(w*params['lf']+v_perp,v)
                alpha_r = math.atan2(w*params['lr']-v_perp,v)
                extra_force,extra_torque = wear_tear(alpha_f,alpha_r,steering,yaw,params)
                b = 2.
                steering /= (1-b*abs(steering)/(2*3.4))
                steering *= 180./(3.14*70.)
                # print("Steering : ", steering, inputs[1,itr])
                _control.steer = min(1.,max(-1.,steering))
                _control.throttle = max(0.,min(1.,inputs[0,itr]))
                _control.brake = -max(-1.,min(0.,inputs[0,itr]))
                _control.manual_gear_shift = True
                _control.gear = 4
                world.player.add_force(extra_force)
                world.player.add_torque(extra_torque)
                world.player.apply_control(_control)
                print("Control applied")
            
            # time : 0, pos x : 1, pos y : 2, pos z : 3, roll : 4, 
            # pitch : 5, yaw : 6, velx : 7, vely : 8, w : 9, 
            # ax : 10, ay : 11, throttle : 12, steering : 13, brake : 14
            
            if itr*dt > 5.4 :
                # print("Steering : ", _control.steer, steering, inputs[1,itr])
                run_data.append([itr*dt, location.x,location.y,location.z,roll,pitch,yaw,velocity.x,velocity.y,w,acc.x,acc.y,_control.throttle,_control.steer,_control.brake])
            if itr%1000 == 0 :
                np.savetxt('run'+str(RUN_NO)+'_data.csv',np.array(run_data),delimiter=',')
                fig.savefig('path_carla_with.png')
                fig_speeds.savefig('speeds_carla_with.png')
		        # fig_mus.savefig(RUN_FOLDER+'Video_mus/frame'+str(idt)+'.png', dpi=200)

            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            if itr%4==0 :
                mus_fig.savefig('Video_mus/frame'+str(itr//4)+'.png')
                pygame.image.save(display, "_out_without/frame"+str(itr//4)+'.png')
        

        np.savetxt('run'+str(RUN_NO)+'_data.csv',np.array(run_data),delimiter=',')
        plt.show()
    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        plt.show()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x420',
        help='window resolution (default: 800x420)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.ford.mustang")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
