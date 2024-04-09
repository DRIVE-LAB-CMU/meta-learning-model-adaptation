from copy import deepcopy
from typing import Union
import gym
from gym import spaces
import numpy as np
from car_dynamics.models_jax import DynamicBicycleModel, CarState, CarAction
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use('Agg')



class OffroadCar(gym.Env):
    '''We assume the only effect of box/chair on the car is changing the Kp,Kd,max_vel,max_steer,
        This env is used for dynamics learning, no hazard
        
        - Apply Dynamic Bicycle Model    
    '''

    DEFAULT = {
        'max_step': 100,
    }


    def __init__(self, config: dict, dynamics: DynamicBicycleModel):
        super(OffroadCar, self).__init__()
        
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)
            
        self.sim = dynamics


        # Action space: move in x and y direction [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1., -1.]), 
                                       high=np.array([1., 1.]), dtype=np.float32)
        
        self.observation_space = spaces.Box(
                ## x, y, yaw_x, yaw_y, vel, Fx, Fy
                low=np.array([-np.inf] * 6), 
                high=np.array([np.inf] * 6), 
                dtype=np.float32,
        )

        self._step = None

        self.reset()


    def obs_state(self):
        return np.array([self.state.x, self.state.y, self.state.psi, self.state.vx, self.state.vy, self.state.omega])
    

    def reset(self):
        
        self.state = CarState(
            x=0.,
            y=-1.4,
            psi=0.,
            vx=0.,
            vy=0.,
            omega=.0,
        )
     

        self._step = 0
        self.sim.reset()

        return self.obs_state()

    def reward(self,):
        return .0
    
    def step(self, action_):

        # normalize action to [-1, 1.]
        self._step += 1
        action = action_ + .0
        action[0] = max(min(action[0], 1.), 0.)
        action[1] = max(min(action[1], 1.), -1.)

        action = CarAction(
            target_vel = action[0],
            target_steer = action[1],
        )
        self.state = self.sim.step_gym(self.state, action)

        reward = self.reward()

        if self._step >= self.max_step:
            done = True
        else:
            done = False

        return self.obs_state(), reward, done, {}
    
    def render(self, mode='human'):
        ...


