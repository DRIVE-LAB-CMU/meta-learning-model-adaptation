import jax
import jax.numpy as jnp
from car_dynamics.models_jax.utils import normalize_angle_tensor, fold_angle_tensor
import numpy as np
import torch

#----------------- reward functions -----------------#

def reward_track_fn(goal_list: jnp.ndarray, defaul_speed: float):
    def reward(state, action, discount):
        """
            - state: contains current state of the car
        """
        # import pdb; pdb.set_trace()
        num_rollouts = action.shape[0]
        horizon = action.shape[1]
        reward_rollout = jnp.zeros((num_rollouts))
        reward_activate = jnp.ones((num_rollouts))
        
        # import pdb; pdb.set_trace()
        for h in range(horizon):
            
            state_step = state[h+1]
            action_step = action[:, h]
            # import pdb; pdb.set_trace()
            dist = jnp.linalg.norm(state_step[:, :2] - goal_list[h+1, :2], axis=1)
            # vel_direction = state[h][:,:2] - state[h-1][:,:2]
            # pos_direction = - state[h][:,:2] + goal_list[h, :2] 
            # dot_product = (vel_direction * pos_direction).sum(dim=1)
            # cos_angle = dot_product / (jnp.norm(pos_direction, dim=1) * jnp.norm(vel_direction, dim=1) + 1e-7)
            vel_diff = jnp.linalg.norm(state_step[:, 3:4] - defaul_speed, axis=1)
            reward = -dist/10. - 0.0 * vel_diff - 0.0 * jnp.linalg.norm(action_step[:, 1:2], axis=1)
            # reward = - 0.4 * dist - 0.0 * jnp.norm(action_step, dim=1) - 0.0 * vel_diff - 0.1 * jnp.log(1 + dist)
            # reward = - 0.4 * dist
            reward_rollout += reward *(discount ** h) * reward_activate
        return reward_rollout
    return reward

#----------------- rollout functions -----------------#

def rollout_fn_select(model_struct, model, dt, L, LR):
    
    
    @jax.jit
    def rollout_fn_dbm(obs_history, state, action, params, debug=False):
        # import pdb; pdb.set_trace()
        assert state.shape[1] == 6
        assert action.shape[1] == 2
        # need state = [x, y, psi, vx, vy, omega]
        # omega = obs_history[:, -1, 2] - obs_history[:, -2, 2]
        # omega = jnp.atan2(jnp.sin(omega), jnp.cos(omega)) / dt
        # v_vec = obs_history[:, -1, :2] - obs_history[:, -2, :2]
        # vx = v_vec[:, 0] * jnp.cos(state[:, 2]) + v_vec[:, 1] * jnp.sin(state[:, 2])
        # vy = v_vec[:, 1] * jnp.cos(state[:, 2]) - v_vec[:, 0] * jnp.sin(state[:, 2])
        # vx = state[:, 3] * jnp.cos(state[:, 2])
        # vy = state[:, 3] * jnp.sin(state[:, 2])
        next_state = model.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5], action[:, 0], action[:, 1])
        # next state is [x, y, psi, vx, vy, omega]
        next_state = jnp.stack(next_state, axis=1)
        # next_state = next_state[:, :4]
        return next_state, {}

    @jax.jit
    def forward(params, x):
        # print(params)
        (w1, b1), (w2, b2) = params
        hidden = jnp.dot(x, w1) + b1
        hidden = jax.nn.relu(hidden)
        output = jnp.dot(hidden, w2) + b2
        return output

    @jax.jit
    def rollout_fn_nn(obs_history, state, action, params, debug=False):
        assert state.shape[1] == 6
        assert action.shape[1] == 2
        # print("Here?")
        # print(obs_history.shape)
        # _X = obs_history[1:, :, 3:].reshape(obs_history.shape[1], -1)
        X = jnp.concatenate((state[:, 3:], action), axis=1)
        # X = jnp.concatenate((_X, X_), axis=1)
        next_state = jnp.array(state)
        # print("What? ",params)
        # print(X.shape)
        gradX = forward(params, jnp.array(X))

        next_state = next_state.at[:, 3].add(gradX[:, 0] * dt)
        next_state = next_state.at[:, 4].add(gradX[:, 1] * dt)
        next_state = next_state.at[:, 5].add(gradX[:, 2] * dt)
        next_state = next_state.at[:, 2].add(state[:, 5] * dt)

        next_state = next_state.at[:, 0].add(state[:, 3] * jnp.cos(state[:, 2]) * dt - state[:, 4] * jnp.sin(state[:, 2]) * dt)
        next_state = next_state.at[:, 1].add(state[:, 3] * jnp.sin(state[:, 2]) * dt + state[:, 4] * jnp.cos(state[:, 2]) * dt)
        return next_state, {}

    if model_struct == 'dbm':
        return rollout_fn_dbm
    elif model_struct == 'nn':
        return rollout_fn_nn
    else:
        raise Exception(f"model_struct {model_struct} not supported!")