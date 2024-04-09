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

from bayes_race.params import ORCA, CarlaParams
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZ, CarlaRace
from bayes_race.mpc.planner import ConstantSpeed
from bayes_race.mpc.nmpc import setupNLP

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
if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''

#####################################################################
# load vehicle parameters

params = CarlaParams(control='pwm')
model = Dynamic(**params)

#####################################################################
# load track

TRACK_NAME = 'Carla'
track = CarlaRace(reference='optimal')
SIM_TIME = 85

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,:] = np.expand_dims(x_init,axis=1)
_,projidx = track.project_fast(x_init[0],x_init[1],track.raceline)
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

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

plt.figure()
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
    global projidx, states, inputs
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

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
        track = CarlaRace(reference='optimal')
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
        while True:
            itr += 1
            if itr > n_steps-horizon :
                break
            
            if args.sync:
                sim_world.tick()
                clock.tick_busy_loop(60)
                
            
            location = world.player.get_location()
            velocity = world.player.get_velocity()
            vx = velocity.x
            vy = velocity.y
            acc = world.player.get_acceleration()
            w = world.player.get_angular_velocity().z*math.pi/180.
            yaw = world.player.get_transform().rotation.yaw*math.pi/180.
            roll = world.player.get_transform().rotation.roll
            pitch = world.player.get_transform().rotation.pitch
            states[0,itr] = location.x
            states[1,itr] = location.y
            states[2,itr] = yaw
            states[3,itr] = vx*math.cos(yaw) + vy*math.sin(yaw)
            states[4,itr] = -vx*math.sin(yaw) + vy*math.cos(yaw)
            states[5,itr] = w
            # print(states[:,itr+1])
            # states[6,itr+1] = (inputs[1,itr]-states[6,itr])/dt
            
            uprev = inputs[:,itr-1]
            x0 = states[:,itr]
            x0[3] = max(5,x0[3])
            # planner based on BayesOpt
            # print("State : ", x0)
            xref, projidx, v_ref = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)
            v_refs.append(v_ref)
            if projidx > track.raceline.shape[1]-5 :
                projidx = 0
            start = tm.time()
            umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev, pitch=pitch*(3.14/180.), roll=roll*(3.14/180.))
            end = tm.time()
            inputs[:,itr] = umpc[:,0]
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
            hstates2[:,0] = x0
            # print(umpc[1,:])
            # print(xmpc[3:6,:])

            for idh in range(horizon):
                # x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
                # hstates[:,idh+1] = x_next[:,-1]
                hstates2[:,idh+1] = xmpc[:,idh+1]
            vec = target_point-curr_point
            vec_near = nearest_point-curr_point
            if velocity.x==0 :
                x_ = vec[0]*math.cos(yaw) + vec[1]*math.sin(yaw)
                y_ = vec[1]*math.cos(yaw) - vec[0]*math.sin(yaw)
            else :
                x_ = vec[0]*math.cos(yaw+0.0*(beta-yaw)) + vec[1]*math.sin(yaw+0.0*(beta-yaw))
                y_ = vec[1]*math.cos(yaw+0.0*(beta-yaw)) - vec[0]*math.sin(yaw+0.0*(beta-yaw))
            # print(x_,y_,v_raceline[i],speed)
            print("Lateral error : ", itr, math.sqrt(vec_near[0]**2+vec_near[1]**2))
            # print("Pitch, roll : ", pitch, roll)
            
            # update plot
            LnS.set_xdata(states[0,:itr+1])
            LnS.set_ydata(states[1,:itr+1])

            LnR.set_xdata(xref[0,1:])
            LnR.set_ydata(xref[1,1:])

            LnP.set_xdata(states[0,itr])
            LnP.set_ydata(states[1,itr])
            
            LnH2.set_xdata(hstates2[0])
            LnH2.set_ydata(hstates2[1])
            
            plt.pause(Ts/1000)
            
            Lnv_ref.set_xdata(time[:itr+1])
            Lnv_ref.set_ydata(v_refs)
            
            Lnv.set_xdata(time[:itr+1])
            Lnv.set_ydata(states[3,:itr+1])
            # LnFrx.set_xdata(time[:idt+1])
            # LnFrx.set_ydata(Frx[:idt+1])

            # LnFry.set_xdata(time[:idt+1])
            # LnFry.set_ydata(Fry[:idt+1])
            # vec = target_point-nearest_point
            # x_ = vec[0]*math.cos(yaw) + vec[1]*math.sin(yaw)
            # y_ = vec[1]*math.cos(yaw) - vec[0]*math.sin(yaw)
            # print(x_,y_)
            if MANUAL_CONTROL :
                if controller.parse_events(client, world, clock, args.sync):
                    return
            else :
                steering = inputs[1,itr]
                b = 2.
                steering /= (1-b*abs(steering)/(2*3.4))
                steering *= 180./(3.14*70.)
                print("Steering : ", steering, inputs[1,itr])
                _control.steer = min(1.,max(-1.,steering))
                _control.throttle = max(0.,min(1.,inputs[0,itr]))
                _control.brake = -max(-1.,min(0.,inputs[0,itr]))
                _control.manual_gear_shift = True
                _control.gear = 4
                world.player.apply_control(_control)
            
            # time : 0, pos x : 1, pos y : 2, pos z : 3, roll : 4, 
            # pitch : 5, yaw : 6, velx : 7, vely : 8, w : 9, 
            # ax : 10, ay : 11, throttle : 12, steering : 13, brake : 14
            
            if itr*dt > 5.4 :
                print("Steering : ", _control.steer, steering, inputs[1,itr])
                run_data.append([itr*dt, location.x,location.y,location.z,roll,pitch,yaw,velocity.x,velocity.y,w,acc.x,acc.y,_control.throttle,_control.steer,_control.brake])
            if itr%100 == 0 :
                np.savetxt('run'+str(RUN_NO)+'_data.csv',np.array(run_data),delimiter=',')
             
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

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
        default='1280x720',
        help='window resolution (default: 1280x720)')
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
