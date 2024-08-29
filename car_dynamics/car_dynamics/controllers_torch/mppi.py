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
                lstm=False
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
        self.lstm = lstm
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
    
    def pure_pursuit(self, state, target_pos, L=2.,steer_factor=.34):
        # print(target_pos, state)
        dx_ = target_pos[0] - state[0]
        dy_ = target_pos[1] - state[1]
        theta = state[2]
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)
        steer = 2. * L * dy / (dx**2 + dy**2)
        return float(steer/steer_factor)
        
    def _sample_actions(self, curr_state, target_pos):
        
        # st = time.time()
        act_mean = self.a_mean
        action_dist = MultivariateNormal(loc=act_mean, covariance_matrix=self.a_cov)
        
        a_sampled = action_dist.sample((self.n_rollouts,))
        # import pdb; pdb.set_trace()
        for d in range(len(self.a_min)):
            a_sampled[:, :, d] = torch.clip(a_sampled[:, :, d], self.a_min[d], self.a_max[d]) * self.a_mag[d] + self.a_shift[d]
        # import pdb; pdb.set_trace()
        # print("clip time", time.time() - st)
        return a_sampled.to(self.device)
        
    def _get_rollout(self, state_init, actions, fix_history=False, one_hot_delay=None, debug=False, h_0s=None, c_0s=None):
        
        # st = time.time()
        n_rollouts = actions.shape[0]
        # h_0s = h_0s.repeat(n_rollouts,1,1)
        if self.lstm :
            for i in range(len(h_0s)):
                h_0s[i] = h_0s[i].repeat(1, n_rollouts, 1)
                c_0s[i] = c_0s[i].repeat(1, n_rollouts, 1)
        # for c_0 in c_0s:
        #     print("a", c_0.shape)
        #     c_0 = c_0.repeat(n_rollouts, 1, 1)
        #     print("b", c_0.shape)
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
            # print(one_hot_delay)
            if self.lstm :
                # print(h_0s[0])
                state, debug_info = self.rollout_fn(obs_history, state, a_rollout, h_0s=h_0s, c_0s=c_0s)
            else :
                state, debug_info = self.rollout_fn(obs_history, state, a_rollout, self.debug, one_hot_delay)
            # print("part 2", time.time() - st)
            # st = time.time()
            if self.debug:
                self.x_all += debug_info['x'].tolist()
                self.y_all += debug_info['y'].tolist()
            total_var += debug_info['var']
            if self.lstm :
                h_0s = debug_info['h_ns']
                c_0s = debug_info['c_ns']
            state_list.append(state)
            # print("part 3", time.time() - st)
            # print("mppi inner loop", time.time() - st)
            # import pdb; pdb.set_trace()
            # reward_rollout += reward_fn(state, a_rollout) * (self.discount ** step)
        
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
        target_pos=None,
        h_0s=None,
        c_0s=None
    ):
        
        st_mppi = time.time()
        # print("MPPI START", st_mppi)
        st = time.time()

        a_sampled_raw = self._sample_actions(obs,target_pos) # Tensor

        
        # print("sample time", time.time() - st)
        ## Delay
        st = time.time()
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
        
        
        # print("rollout_start", time.time() - st)
        # import pdb; pdb.set_trace()
        
        st = time.time()
        # print(h_0s.shape,c_0s.shape)
        # a_sampled[-1,:,1] = 1.
        state_list, total_var = self._get_rollout(state_init, a_sampled, self.fix_history, one_hot_delay, h_0s=h_0s, c_0s=c_0s) # List
        
        
        # print("rollout_end", time.time() - st)
        
        t_rollout = time.time() - st
        
        st = time.time()
        # print(torch.mean(total_var),torch.max(total_var),torch.min(total_var))
        # calculate reward
        if self.delay > 0:
            reward_rollout = reward_fn(state_list, a_sampled[:,:-self.delay,:], self.discount) # Tensor
        else :
            reward_rollout = reward_fn(state_list, a_sampled, self.discount) # Tensor
        cost_rollout = -reward_rollout + self.alpha * total_var/30.

        # ind = torch.argmin(cost_rollout)
        # print(f"min cost: {cost_rollout[ind]}")
        # print("actions: ", a_sampled[ind, :, :])
        # print("actions2: ", a_sampled[-1, :, :])
        # print("states: ", torch.stack(state_list,dim=1)[ind,:,:])
        # print("states2: ", torch.stack(state_list,dim=1)[-1,:,:])
        # print(f"cost2: {cost_rollout[-1]}")
        cost_exp = torch.exp(-(cost_rollout - torch.min(cost_rollout)) / self.lam)
        weight = cost_exp / cost_exp.sum()
        # import pdb; pdb.set_trace()
        # print("reward calc + rollout time", time.time() - st)
        
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
        # print("update time", time.time() - st)
        t_misc = time.time() - st
        st = time.time()
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
                # optim_traj = torch.stack(self._get_rollout(state_init, self.a_mean.unsqueeze(0), self.fix_history, one_hot_delay)[0])[:, 0, :].detach().cpu().numpy()
                # import pdb; pdb.set_trace()
                # optim_traj 
                optim_traj = np.zeros((self.H + 1, 3))
                # print(colored(f"optimal tra (-1): {optim_traj[-1, :2]}" , "red"))
                
                if np.abs(optim_traj[-1, 0]) < -10000.:
                    import pdb; pdb.set_trace()
                    
        # print("vis time", time.time() - st)
        # import pdb; pdb.set_trace()
        # print(self.a_mean.shape)
        st = time.time()
        actions = self.a_mean.detach()
        # print("SAMPLE END", time.time()-st)
        actions = actions.cpu()
        actions = actions.numpy()
        # print("SAMPLE END", time.time()-st)
        # print("SAMPLE END", time.time()-st)
        self.a_mean = torch.cat([self.a_mean[1:], self.a_mean[-1:]], dim=0)
        self.a_cov = torch.cat([self.a_cov[1:], self.a_cov[-1:]], dim=0)
        # self.a_mean = torch.cat([self.a_mean[1:], self.a_mean_init], dim=0)
        # self.a_cov = torch.cat([self.a_cov[1:], self.a_cov_init], dim=0)
        # print(self.a_mean_init)
        self.prev_a.append(u)
        # print("mppi time", time.time() - st)
        self.step_count += 1
        
        
        mppi_time = time.time() - st_mppi
        # print("MPPI END", time.time()-st)
        info_dict = {'trajectory': optim_traj, 'action': actions, 
                'action_candidate': None, 'x_all': None, 'y_all': None,
                't_debug': {'sample': t_sample, 'rollout': t_rollout, 'calc_reward': t_calc_reward, 'misc': t_misc, 'total': mppi_time},} 
        
        # print("MPPI REAL END", time.time() - st)
        t1 = time.time()
        u = u.cpu()
        t2 = time.time()
        # print("u to cpu", t2 - t1)
        # print(optim_traj)
        return u, info_dict
