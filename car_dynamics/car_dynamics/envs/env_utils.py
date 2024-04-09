

def make_env(env_name):
    '''config env
    '''
    if env_name == 'car-base':
        from car_dynamics.envs import Navigation2
        env = Navigation2({})
        return env
    elif env_name == 'car-base-single':
        from car_dynamics.envs import Navigation2
        env = Navigation2({
            'max_step': 10000,
            'env_Kp': [7., 7.],
            'env_Kd': [.02, .02],
            'env_max_vel': [5.,5.],
            'env_max_steer': [.34,.34],
            'init_yaw': 0.,
            # 'init_yaw': 3.14,
            'init_pos': [0.8, -0.4],
            'dt': 0.05,
            'proj_steer': .48,
            'shift_steer': -0.08,
        })
        return env
    else:
        raise ValueError(f'Unknown env {env_name}')