import imageio
import numpy as np
import matplotlib.pyplot as plt
from car_dynamics.envs import make_env

env = make_env('car-base-single')

done = False
frames = []

obs_list = []
while not done:
    action = env.action_space.sample()
    action = np.array([1., .5])
    obs, reward, done, info = env.step(action)
    # frames.append(env.render(mode='rgb_array'))
    obs_list.append(obs)
    
obs_list = np.array(obs_list)
plt.plot(obs_list[:, 0], obs_list[:, 1])
plt.savefig('tmp/test_single_car_env.png')
    

# imageio.mimsave('tmp/test_single_car_env.gif', frames, fps=30)