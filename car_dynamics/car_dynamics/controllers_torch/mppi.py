import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
import time
from termcolor import colored
from car_dynamics.controllers_torch import BaseController
import numpy as np

class MPPIController(BaseController):
    def __init__(self,
                gamma_mean,
                gamma_sigma,
                discount,
                sample_sigma,
                lam,
                a_mean,
                a_cov,
                n_rollouts,
                H,
                device,
                rollout_fn,
                a_min,
                a_max,
                a_mag, # magnitude
                a_shift, # shift,
                delay, # delay
                len_history, 
                rollout_start_fn,
                debug,
                fix_history,
                num_obs,
                num_actions,
                alpha,
    ):
        
        self.gamma_mean: float = gamma_mean
        self.gamma_sigma: float = gamma_sigma
        self.discount: float = discount
        self.sample_sigma: float = sample_sigma
        self.lam: float = lam
        self.alpha: float = alpha
        self.a_mean: torch.Tensor = a_mean # a should be normalized to [-1, 1] in dynamics
        self.a_cov: torch.Tensor = a_cov
        self.a_mean_init = a_mean[-1:]
        self.a_cov_init = a_cov[-1:]
        # self.a_init: torch.Tensor = ...
        
        self.n_rollouts: int = n_rollouts
        self.H: int = H # horizon
        self.device = device
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        
        # self.reward_fn = reward_fn
        self.rollout_fn = rollout_fn
        
        self.a_min = a_min
        self.a_max = a_max
        self.a_mag = a_mag
        self.a_shift = a_shift
        self.delay = delay
        # assert self.delay > 0
        self.prev_a = [torch.zeros(self.num_actions).to(device)] * delay
        self.len_history = len_history
        self.step_count = 0
        if self.len_history > 0:
            self.state_hist = torch.zeros((self.len_history, self.num_obs + self.num_actions)).to(device)
            
        self.rollout_start_fn = rollout_start_fn
        self.debug = debug
        self.fix_history = fix_history
        self.action_init_buf = torch.zeros((self.n_rollouts, self.H + self.delay, self.num_actions), device=self.device)
        self.state_init_buf = torch.ones((self.num_obs,), device=self.device)
        self.x_all = []
        self.y_all = []
        
    def _sample_actions(self, ):
        
        # st = time.time()
        action_dist = MultivariateNormal(loc=self.a_mean, covariance_matrix=self.a_cov)
        # action_dist = MultivariateNormal(loc=self.a_mean*0, covariance_matrix=self.a_cov)
        # action_dist = Uniform(self.a_mean*0-1., self.a_mean*0+1.)
        # print("torch sample time", time.time() - st)
        
        # st = time.time()
        # import pdb; pdb.set_trace()
        # print(self.a_mean.device, self.a_cov.device)
        a_sampled = action_dist.sample((self.n_rollouts,))
        # import pdb; pdb.set_trace()
        for d in range(len(self.a_min)):
            a_sampled[:, :, d] = torch.clip(a_sampled[:, :, d], self.a_min[d], self.a_max[d]) * self.a_mag[d] + self.a_shift[d]
        # import pdb; pdb.set_trace()
        # print("clip time", time.time() - st)
        return a_sampled.to(self.device)
        
    def _get_rollout(self, state_init, actions, fix_history=False, one_hot_delay=None, debug=False):
        
        # st = time.time()
        n_rollouts = actions.shape[0]
        state = state_init.unsqueeze(0).repeat(n_rollouts, 1)
        state_list = [state]
        obs_history = self.state_hist.unsqueeze(0).repeat(n_rollouts, 1, 1)
        # reward_rollout = torch.zeros((self.n_rollouts), device=self.device)
        self.rollout_start_fn()
        
        
        # print("rollout_fn_start", time.time() - st)
        
        # st = time.time()
        total_var = 0.
        for step in range(actions.shape[1]):
            
            # st = time.time()
            a_rollout = actions[:, step]
            if (not fix_history) or (step == 0):
                obs_history[:, :-1] = obs_history[:, 1:].clone()
                obs_history[:, -1, :self.num_obs] = state[:, :self.num_obs].clone()
                obs_history[:, -1, -self.num_actions:] = a_rollout.clone()
            # print(f"action shape, {a_rollout.shape}")
            # print("part 1", time.time() - st)
            # st = time.time()
            if (step==4) and debug:
                state, debug_info = self.rollout_fn(obs_history, state, a_rollout, True, one_hot_delay)
            else :
                state, debug_info = self.rollout_fn(obs_history, state, a_rollout, self.debug, one_hot_delay)
            # print("part 2", time.time() - st)
            # st = time.time()
            if self.debug:
                self.x_all += debug_info['x'].tolist()
                self.y_all += debug_info['y'].tolist()
            total_var += debug_info['var']
            state_list.append(state)
            # print("part 3", time.time() - st)
            # print("mppi inner loop", time.time() - st)
            # import pdb; pdb.set_trace()
            # reward_rollout += reward_fn(state, a_rollout) * (self.discount ** step)
        # print("rollout_fn_end", time.time() - st)
        
        return state_list, total_var
    
    def feed_hist(self, obs, action):
        state = torch.tensor(obs[:self.num_obs], device=self.device)
        action_tensor = torch.tensor(action[:self.num_actions], device=self.device)
        self.state_hist[:-1] = self.state_hist[1:].clone()
        self.state_hist[-1, :self.num_obs] = state.clone()
        self.state_hist[-1, self.num_obs:self.num_obs + self.num_actions] = action_tensor.clone()
        
    def __call__(
        self,
        obs,
        reward_fn,
        vis_optim_traj=False,
        use_nn = False,
        vis_all_traj = False,
        one_hot_delay = None,
    ):
        
        st_mppi = time.time()
        # print("MPPI START", st_mppi)
        st = time.time()

        a_sampled_raw = self._sample_actions() # Tensor

        
        # print("sample time", time.time() - st)
        ## Delay
        # st = time.time()
        # a_sampled = torch.zeros((self.n_rollouts, self.H + self.delay, a_sampled_raw.shape[2])).to(self.device)
        a_sampled = self.action_init_buf.clone()
        for i in range(self.delay):
            a_sampled[:, i, :] = self.prev_a[i - self.delay]
        a_sampled[:, self.delay:, :] = a_sampled_raw
        ########
        # print("delay time", time.time() - st)
        t_sample = time.time() - st

        st = time.time()
        ## rollout 
        
        ## Before: 0.005 s
        # state_init = torch.Tensor(obs).to(self.device)            
        
        ## After: 0.00005 s
        state_init = self.state_init_buf.clone()
        for i_ in range(self.num_obs):
            state_init[i_] *= obs[i_]
        
        # print(obs)
        
        
        # print("rollout_start", time./time() - st)
        # import pdb; pdb.set_trace()
        
        # st = time.time()
        state_list, total_var = self._get_rollout(state_init, a_sampled, self.fix_history, one_hot_delay) # List
        
        
        # print("rollout_end", time.time() - st)
        
        t_rollout = time.time() - st
        
        st = time.time()
        # print(torch.mean(total_var),torch.max(total_var),torch.min(total_var))
        # calculate reward
        reward_rollout = reward_fn(state_list, a_sampled, self.discount) # Tensor
        cost_rollout = -reward_rollout + self.alpha * total_var/30.

        # print("rollout time", time.time() - st)

        cost_exp = torch.exp(-(cost_rollout - torch.min(cost_rollout)) / self.lam)
        weight = cost_exp / cost_exp.sum()
        # import pdb; pdb.set_trace()
        
        t_calc_reward = time.time() - st
        
        st = time.time()
        a_sampled = a_sampled[:, self.delay:, :]
        self.a_mean = torch.sum(weight[:, None, None] * a_sampled, dim=0) * self.gamma_mean + self.a_mean * (1 - self.gamma_mean)
        # import pdb; pdb.set_trace()
        
        self.a_cov = torch.sum(
                        weight[:, None, None, None] * ((a_sampled - self.a_mean)[..., None] * (a_sampled - self.a_mean)[:, :, None, :]),
                        dim=0,
                    ) * self.gamma_sigma + self.a_cov * (1 - self.gamma_sigma)
        
        u = self.a_mean[0]

        t_misc = time.time() - st
                
        optim_traj = None
        if vis_optim_traj:
            if use_nn:
                
                print(colored(f"state init: {state_init}", "green"))
                optim_traj = torch.vstack(self._get_rollout(state_init, self.a_mean.unsqueeze(0), self.fix_history, one_hot_delay)[0]).detach().cpu().numpy()
                
                # import pdb; pdb.set_trace()
                # print(colored(f"optimal tra (-1): {optim_traj[-1, :2]}" , "red"))
                if np.abs(optim_traj[-1, 0]) > 10:
                    import pdb; pdb.set_trace()
                    
            else:
                optim_traj = torch.stack(self._get_rollout(state_init, self.a_mean.unsqueeze(0).repeat(self.n_rollouts, 1, 1), self.fix_history, one_hot_delay,debug=True)[0])[:, 0, :].detach().cpu().numpy()
                # import pdb; pdb.set_trace()
                print(colored(f"optimal tra (-1): {optim_traj[-1, :2]}" , "red"))
                
                if np.abs(optim_traj[-1, 0]) < -10000.:
                    import pdb; pdb.set_trace()
                    
                
        # import pdb; pdb.set_trace()
        actions = self.a_mean.detach().cpu().numpy()
        self.a_mean = torch.cat([self.a_mean[1:], self.a_mean[-1:]], dim=0)
        self.a_cov = torch.cat([self.a_cov[1:], self.a_cov[-1:]], dim=0)
        # self.a_mean = torch.cat([self.a_mean[1:], self.a_mean_init], dim=0)
        # self.a_cov = torch.cat([self.a_cov[1:], self.a_cov_init], dim=0)
        # print(self.a_mean_init)
        self.prev_a.append(u)
        # print("mppi time", time.time() - st)
        self.step_count += 1
        
        
        mppi_time = time.time() - st_mppi
        print("MPPI END", time.time())
        info_dict = {'trajectory': optim_traj, 'action': actions, 
                'action_candidate': None, 'x_all': None, 'y_all': None,
                't_debug': {'sample': t_sample, 'rollout': t_rollout, 'calc_reward': t_calc_reward, 'misc': t_misc, 'total': mppi_time},} 
        
        # info_dict = {'trajectory': optim_traj, 'action': self.a_mean.detach().cpu().numpy(), 
        #              'action_candidate': a_sampled.detach().cpu().numpy(), 'x_all': self.x_all, 'y_all': self.y_all,
        #              't_debug': {'sample': t_sample, 'rollout': t_rollout, 'calc_reward': t_calc_reward, 'misc': t_misc, 'total': mppi_time},} 
        # if vis_all_traj:
        #     raise NotImplementedError
        #     info_dict['all_trajectory'] = torch.stack(state_list).detach().cpu().numpy()
            # info_dict['all_trajectory'] = state_list
        # print("MPPI REAL END", time.time())
        return u.cpu(), info_dict
