from dataclasses import dataclass
import jax
import jax.numpy as jnp
import time
from termcolor import colored
from .base import BaseController
from functools import partial



@dataclass
class MPPIParams:
    sigma: float
    gamma_mean: float
    gamma_sigma: float
    discount: float
    sample_sigma: float
    lam: float
    n_rollouts: int
    H: int
    a_min: jnp.ndarray
    a_max: jnp.ndarray
    a_mag: jnp.ndarray
    a_shift: jnp.ndarray
    delay: int
    len_history: int
    debug: bool
    fix_history: bool
    num_obs: int
    num_actions: int
    smooth_alpha: float = 0.8
    
class MPPIController(BaseController):
    def __init__(self,
                params: MPPIParams, rollout_fn: callable, rollout_start_fn: callable, key, nn_model, rollout_fn_nn=None):
        """ MPPI implemented in Jax """
        self.params = params
        self.nn_model = nn_model
        if rollout_fn_nn is None:
            self.rollout_fn_nn = rollout_fn
            # print("Yess!!WTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
        else :
            self.rollout_fn_nn = rollout_fn_nn
        self.rollout_fn = rollout_fn
        self.rollout_start_fn = rollout_start_fn
        self.key, _ = jax.random.split(key)
        self._init_buffers()
        
    def _init_buffers(self, ):
        self.a_mean = jnp.zeros((self.params.H, self.params.num_actions))
        sigmas = jnp.array([self.params.sigma] * 2)
        a_cov_per_step = jnp.diag(sigmas ** 2)
        self.a_cov = jnp.tile(a_cov_per_step[None, :, :], (self.params.H, 1, 1))
        self.a_mean_init = self.a_mean[-1:]
        self.a_cov_init = self.a_cov[-1:]
        
        self.prev_a = [jnp.zeros((self.params.num_actions))] * self.params.delay
        self.step_count = 0
        
        if self.params.len_history > 0:
            self.state_hist = jnp.zeros((self.params.len_history, self.params.num_obs + self.params.num_actions))
            
        self.action_init_buf = jnp.zeros((self.params.n_rollouts, self.params.H + self.params.delay, self.params.num_actions))
        self.state_init_buf = jnp.ones((self.params.num_obs,))
        self.x_all = []
        self.y_all = []
        
    @partial(jax.jit, static_argnums=(0,))
    def _running_average(self, carry, x):
        prev_x = carry
        new_x = x * self.params.smooth_alpha + prev_x * (1 - self.params.smooth_alpha)
        return new_x, new_x
    
    @partial(jax.jit, static_argnums=(0,))
    def _sample_actions(self, ):
        
        key, self.key = jax.random.split(self.key)
        a_sampled = jax.random.multivariate_normal(key, self.a_mean*0, self.a_cov, (self.params.n_rollouts,self.params.H))
        # a_sampled = jax.random.uniform(key, (self.params.n_rollouts,self.params.H, 2), jnp.float32, -1., 1.)
        
        # smooth action
        a_sampled_swap = jnp.swapaxes(a_sampled, 0, 1)
        initial_action = a_sampled_swap[0]
        actions_to_update = a_sampled_swap[1:]
        _, a_sampled_swap = jax.lax.scan(self._running_average, initial_action, actions_to_update)
        a_sampled_swap = jnp.concatenate([initial_action[None], a_sampled_swap], axis=0)
        a_sampled = jnp.swapaxes(a_sampled_swap, 0, 1)
        
        # import pdb; pdb.set_trace()
        for d in range(len(self.params.a_min)):
            a_sampled = a_sampled.at[:, :, d].set(jnp.clip(a_sampled[:, :, d], self.params.a_min[d], self.params.a_max[d]) * self.params.a_mag[d] + self.params.a_shift[d])
        
        # a_sampled = a_sampled.at[:, :, 1].set(-1.)
        # a_sampled = a_sampled.at[:, :, 0].set(1.)
            
        return a_sampled
    
    @partial(jax.jit, static_argnums=(0,))
    def _rollout_jit(self, carry, action):
        state, obs_history, params = carry
        obs_history = obs_history.at[:-1].set(obs_history[1:])    
        obs_history = obs_history.at[-1, :self.params.n_rollouts, :self.params.num_obs].set(state[:, :self.params.num_obs])
        obs_history = obs_history.at[-1, :self.params.n_rollouts, -self.params.num_actions:].set(action)
        state, debug_info = self.rollout_fn(obs_history, state, action, params, self.params.debug)
        return (state, obs_history, params), state
    
    @partial(jax.jit, static_argnums=(0,))
    def _rollout_jit_nn(self, carry, action):
        state, obs_history, params = carry
        obs_history = obs_history.at[:-1].set(obs_history[1:])    
        obs_history = obs_history.at[-1, :self.params.n_rollouts, :self.params.num_obs].set(state[:, :self.params.num_obs])
        obs_history = obs_history.at[-1, :self.params.n_rollouts, -self.params.num_actions:].set(action)
        print(params)
        # print("Passing to nn")
        state, debug_info = self.rollout_fn_nn(obs_history, state, action, params, self.params.debug)
        return (state, obs_history, params), state
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_rollout(self, state_init, actions, fix_history=False, use_nn=False, model_params=None):
        
        # st = time.time()
        # print(model_params)
        n_rollouts = actions.shape[0]
        state = jnp.tile(jnp.expand_dims(state_init, 0), (n_rollouts, 1))
        # import pdb; pdb.set_trace()
        state_list = [state]
        # import pdb; pdb.set_trace()
        obs_history = jnp.tile(jnp.expand_dims(self.state_hist, 0), (n_rollouts, 1, 1))
        self.rollout_start_fn()
        
        # print("rollout start", time.time() - st)
        
        # st = time.time()
        obs_history = jnp.swapaxes(obs_history, 0, 1)
        actions = jnp.swapaxes(actions, 0, 1)
        
        # print("rollout_fn_start", time.time() - st)
        # import pdb; pdb.set_trace()
        # use_nn = False
        # print("Passing to rollout_jit", model_params)
        (_, state_list2) = jax.lax.cond(use_nn,
             lambda _: jax.lax.scan(self._rollout_jit_nn, (state, obs_history, model_params), actions),
             lambda _: jax.lax.scan(self._rollout_jit, (state, obs_history, model_params), actions),
             None)
        # (_, state_list2) = jnp.where(jnp.array([use_nn==True]), jnp.array([jax.lax.scan(self._rollout_jit, (state, obs_history), actions)]), jnp.array([jax.lax.scan(self._rollout_jit_nn, (state, obs_history), actions)]))
        # if use_nn==True:
        #     _, state_list2 = jax.lax.scan(self._rollout_jit_nn, (state, obs_history), actions)
        # else :
        #     _, state_list2 = jax.lax.scan(self._rollout_jit, (state, obs_history), actions)
        state_list = jnp.array(state_list + list(state_list2))
        # print("rollout_fn_end", time.time() - st)
        # import pdb; pdb.set_trace()
        # st = time.time()
        # for step in range(actions.shape[1]):
            
        #     # st = time.time()
        #     a_rollout = actions[:, step]
        #     if (not fix_history) or (step == 0):
        #         obs_history.at[:, -1, :self.params.num_obs].set(state[:, :self.params.num_obs])
        #         obs_history.at[:, -1, -self.params.num_actions:].set(a_rollout)
        #     # print(f"action shape, {a_rollout.shape}")
            
        #     # print("part 1", time.time() - st)
        #     # st = time.time()
        #     state, debug_info = self.rollout_fn(obs_history, state, a_rollout, self.params.debug)
        #     # print("part 2", time.time() - st)
        #     # st = time.time()
        #     if self.params.debug:
        #         self.x_all += debug_info['x'].tolist()
        #         self.y_all += debug_info['y'].tolist()
        #     if not fix_history:
        #         obs_history.at[:, :-1].set(obs_history[:, 1:])
        #     state_list.append(state)
        #     # print("part 3", time.time() - st)
        #     # print("mppi inner loop", time.time() - st)
        #     # import pdb; pdb.set_trace()
        #     # reward_rollout += reward_fn(state, a_rollout) * (self.discount ** step)
        # # print("rollout_fn_end", time.time() - st)
        return state_list
    
    @partial(jax.jit, static_argnums=(0,))
    def single_step_reward(self, carry, pair):
        step, prev_action = carry
        state_step, action_step, goal = pair
        dist_pos = jnp.linalg.norm(state_step[:, :2] - goal[:2], axis=1)
        diff_psi = state_step[:, 2] - goal[2]
        diff_psi = jnp.abs(jnp.arctan2(jnp.sin(diff_psi), jnp.cos(diff_psi)))
        r_slip = jnp.abs(state_step[:, 3]) / (jnp.abs(state_step[:, 4]) + 1e-5)
        # u_square = jnp.sum(action_step ** 2, axis=1)
        # du = jnp.sum((action_step - prev_action) ** 2, axis=1)
        diff_vel = (state_step[:, 3] - 2.) ** 2.
        diff_vx = (state_step[:, 3] - 1.) ** 2.
        diff_vy = (state_step[:, 4] - 1.) ** 2.
        diff_vyaw = (state_step[:, 5] - .5) ** 2.
        
        reward_pos_err = -dist_pos ** 2
        reward_slip_angle = -r_slip ** 2
        reward_psi = -diff_psi ** 2
        reward_vel = -diff_vel ** 2
        reward_vy = -(state_step[:, 4] - (-.5)) ** 2
        reward = reward_pos_err * 5. + reward_slip_angle * .00 + reward_psi * 0. + reward_vel * 0.0 + reward_vy * 0.
        # reward -= diff_vy + diff_vyaw + diff_vx
        reward *= (self.params.discount ** step)
        return (step + 1, action_step), reward
    
    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, state, action, goal_list):
        actions = jnp.swapaxes(action, 0, 1)
        # import pdb; pdb.set_trace()
        # goal_list = jnp.tile(jnp.expand_dims(goal_list, 0), (actions.shape[0], 1, 1)
        _, reward_list = jax.lax.scan(self.single_step_reward, (0, actions[0]), (state[1:], actions, goal_list[1:]))
        # import pdb; pdb.set_trace()
        return jnp.sum(reward_list, axis=0)
        # for h in range(horizon):
            
        #     state_step = state[h+1]
        #     action_step = action[:, h]
        #     # import pdb; pdb.set_trace()
        #     dist = jnp.linalg.norm(state_step[:, :2] - goal_list[h+1, :2], axis=1)
        #     reward = -dist
        #     # reward = - 0.4 * dist
        #     reward_rollout += reward *(discount ** h) * reward_activate
        # return reward_rollout
        
        reward = reward_fn(state, action, self.params.discount) # Tensor
    
    # @partial(jax.jit, static_argnums=(0,))
    def feed_hist(self, obs, action):
        state = jnp.array(obs[:self.params.num_obs])
        action_tensor = jnp.array(action[:self.params.num_actions])
        self.state_hist = self.state_hist.at[:-1].set(self.state_hist[1:])
        self.state_hist = self.state_hist.at[-1, :self.params.num_obs].set(state)
        self.state_hist = self.state_hist.at[-1, self.params.num_obs:self.params.num_obs + self.params.num_actions].set(action_tensor)
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        obs,
        goal_list, 
        model_params,
        vis_optim_traj=False,
        use_nn = False,
        vis_all_traj = False,
    ):
        
        st_mppi = time.time()
        # print("MPPI START", st_mppi)
        st = time.time()

        a_sampled_raw = self._sample_actions() # Tensor

        
        # import pdb; pdb.set_trace()
        a_sampled = self.action_init_buf.copy()
        for i in range(self.params.delay):
            a_sampled = a_sampled.at[:, i, :].set(self.prev_a[i - self.params.delay])
        a_sampled = a_sampled.at[:, self.params.delay:, :].set(a_sampled_raw)
        ########
        
        # import pdb; pdb.set_trace()
        # print("delay time", time.time() - st)
        t_sample = time.time() - st

        st = time.time()

        # st_ = time.time()
        ## After: 0.00005 s
        state_init = self.state_init_buf.copy()
        for i_ in range(self.params.num_obs):
            state_init = state_init.at[i_].set(state_init[i_] * obs[i_])
        
        # print("rollout init", time.time() - st_)
        # print(obs)
        
        
        # print("rollout_start", time./time() - st)
        # import pdb; pdb.set_trace()
        
        # st = time.time()
        # import pdb; pdb.set_trace()
        state_list = self._get_rollout(state_init, a_sampled, self.params.fix_history, use_nn=use_nn, model_params=model_params) # List
        
        # print("rollout_end", time.time() - st)
        
        t_rollout = time.time() - st
        
        st = time.time()
        
        # calculate reward
        reward_rollout = self.get_reward(state_list, a_sampled, goal_list)
        cost_rollout = -reward_rollout

        # print("rollout time", time.time() - st)

        cost_exp = jnp.exp(-(cost_rollout - jnp.min(cost_rollout)) / self.params.lam)
        weight = cost_exp / cost_exp.sum()
        # import pdb; pdb.set_trace()
        
        t_calc_reward = time.time() - st
        
        st = time.time()
        a_sampled = a_sampled[:, self.params.delay:, :]
        
        
        best_k_idx = jnp.argsort(reward_rollout, descending=True)[:1000]
        a_sampled = a_sampled[best_k_idx]
        weight = weight[best_k_idx]
        self.a_mean = jnp.sum(weight[:, None, None] * a_sampled, axis=0) * self.params.gamma_mean + self.a_mean * (1 - self.params.gamma_mean)
        # import pdb; pdb.set_trace()
        
        self.a_cov = jnp.sum(
                        weight[:, None, None, None] * ((a_sampled - self.a_mean)[..., None] * (a_sampled - self.a_mean)[:, :, None, :]),
                        axis=0,
                    ) * self.params.gamma_sigma + self.a_cov * (1 - self.params.gamma_sigma)
        
        u = self.a_mean[0]

        t_misc = time.time() - st
                
        optim_traj = None
        action_expand = jnp.tile(jnp.expand_dims(self.a_mean, 0), (self.params.n_rollouts, 1, 1))
        optim_traj = jnp.stack(self._get_rollout(state_init, action_expand, self.params.fix_history, model_params=model_params))[:, 0]
        # if vis_optim_traj:
        #     if use_nn:
        #         raise NotImplementedError
        #         # print(colored(f"state init: {state_init}", "green"))
        #         # optim_traj = jnp.vstack(self._get_rollout(state_init, self.a_mean.expand_dims(0), self.params.fix_history)).detach().cpu().numpy()
                
        #         # # import pdb; pdb.set_trace()
        #         # # print(colored(f"optimal tra (-1): {optim_traj[-1, :2]}" , "red"))
        #         # if jnp.abs(optim_traj[-1, 0]) > 10:
        #         #     import pdb; pdb.set_trace()
                    
        #     else:
        #         # import pdb; pdb.set_trace()
        #         action_expand = jnp.tile(jnp.expand_dims(self.a_mean, 0), (self.params.n_rollouts, 1, 1))
        #         optim_traj = jnp.stack(self._get_rollout(state_init, action_expand, self.params.fix_history))[:, 0]
                
        #         # import pdb; pdb.set_trace()
        #         # print(colored(f"optimal tra (-1): {optim_traj[-1, :2]}" , "red"))
                
        #         # if jnp.abs(optim_traj[-1, 0]) > 10:
        #         #     import pdb; pdb.set_trace()
                    
                
        # import pdb; pdb.set_trace()
        
        self.a_mean = jnp.concatenate([self.a_mean[1:], self.a_mean[-1:]], axis=0)
        # self.a_mean = jnp.tile(self.a_mean[:1], (self.params.H, 1))
        self.a_cov = jnp.concatenate([self.a_cov[1:], self.a_cov[-1:]], axis=0)

        self.prev_a.append(u)
        # print("mppi time", time.time() - st)
        self.step_count += 1
        
        best_100_idx = jnp.argsort(reward_rollout, descending=True)[:1000]
        
        mppi_time = time.time() - st_mppi
        print("MPPI END", time.time())
        info_dict = {
            'trajectory': optim_traj, 
            'action': self.a_mean, 
            'all_action': a_sampled_raw,
             'all_traj': state_list,#[:, best_100_idx],
                'action_candidate': None, 'x_all': None, 'y_all': None,
                't_debug': {'sample': t_sample, 'rollout': t_rollout, 'calc_reward': t_calc_reward, 'misc': t_misc, 'total': mppi_time, 'MPPI_START': st_mppi, "MPPI_END": time.time()},} 
        

        # print("MPPI END", time.time())
        return u, info_dict
