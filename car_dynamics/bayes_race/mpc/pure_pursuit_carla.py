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

from bayes_race.params import ORCA, F110, CarlaParams
from bayes_race.tracks import ETHZ, ETHZMobil, UCB, CarlaRace
print("Imported bayesrace files")

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from scipy.interpolate import interp1d
from carla_utils import *

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


MANUAL_CONTROL=False
L = 2.64
Kp = 0.2
Ki = 0.000
RUN_NO = 4
# TARGET_SPEED = 22
v_factor = 1.22
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
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        world = client.load_world('Town07_Opt')
        # world.unload_map_layer(carla.MapLayer.Buildings)
        # world.unload_map_layer(carla.MapLayer.Decals)
        # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        # world.unload_map_layer(carla.MapLayer.Particles)
        # world.unload_map_layer(carla.MapLayer.Props)
        world.unload_map_layer(carla.MapLayer.StreetLights)
        world.unload_map_layer(carla.MapLayer.Walls)
        # world.unload_map_layer(carla.MapLayer.Foliage)
        # world.unload_map_layer(carla.MapLayer.All)
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
        while True:
            itr += 1
            if args.sync:
                sim_world.tick()
                clock.tick_busy_loop(60)
                
            
            location = world.player.get_location()
            velocity = world.player.get_velocity()
            acc = world.player.get_acceleration()
            w = world.player.get_angular_velocity().z*3.14/180.
            yaw = world.player.get_transform().rotation.yaw*math.pi/180.
            roll = world.player.get_transform().rotation.roll
            pitch = world.player.get_transform().rotation.pitch
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            beta = math.atan2(velocity.y,velocity.x)
            dists = ((traj_to_follow[:,0]-location.x)**2+(traj_to_follow[:,1]-location.y)**2)
            i = np.argmin(dists)
            curr_point = np.array([location.x,location.y])
            dist = max(3,6*speed/20)
            target_point = find_point_at_dist(location,traj_to_follow,dist)
            nearest_point = find_point_at_dist(location,traj_to_follow,0)
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
            print("Pitch, roll : ", pitch, roll)
            
            # vec = target_point-nearest_point
            # x_ = vec[0]*math.cos(yaw) + vec[1]*math.sin(yaw)
            # y_ = vec[1]*math.cos(yaw) - vec[0]*math.sin(yaw)
            # print(x_,y_)
            if MANUAL_CONTROL :
                if controller.parse_events(client, world, clock, args.sync):
                    return
            else :
                steering = 2*L*y_/(x_**2+y_**2)
                _control.steer = min(1.,max(-1.,steering))
                integral+=(v_raceline[i]-speed)*Ki
                _control.throttle = max(0.,min(1.,Kp*(v_raceline[(i+10)%len(v_raceline)]*v_factor-speed)+integral))
                _control.brake = -max(-1.,min(0.,Kp*(v_raceline[(i+10)%len(v_raceline)]*v_factor-speed)+integral))
                _control.manual_gear_shift = True
                _control.gear = 4
                # if itr*dt < 10. :
                #     _control.throttle = 0.
                #     _control.steer = 0.
                # else :
                #     _control.throttle = 0.
                #     _control.steer = 0.
                world.player.apply_control(_control)
            
            # time : 0, pos x : 1, pos y : 2, pos z : 3, roll : 4, 
            # pitch : 5, yaw : 6, velx : 7, vely : 8, w : 9, 
            # ax : 10, ay : 11, throttle : 12, steering : 13, brake : 14
            
            if itr*dt > 5.4 :
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
