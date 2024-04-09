import numpy as np
import matplotlib.pyplot as plt
import random

saved_raceline = np.loadtxt('/home/dvij/bayesrace/bayes_race/global_racetrajectory_optimization/outputs/traj_race_cl.csv',delimiter=';')
# print(saved_raceline.shape)
x = saved_raceline[:,1]
y = saved_raceline[:,2]
v = saved_raceline[:,5]
time = saved_raceline[:,0]
np.savez('/home/dvij/bayesrace/bayes_race/tracks/src/carla_raceline.npz',x=x,y=y,speed=v,time=time,inputs=time)
# print(saved_raceline['y_ei'].shape)