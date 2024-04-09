import torch
from car_dynamics.models_torch.utils import normalize_angle_tensor, fold_angle_tensor


#----------------- reward functions -----------------#

def reward_track_fn(goal_list: torch.Tensor, defaul_speed: float):
    def reward(state, action, discount):
        """
            - state: contains current state of the car
        """
        # import pdb; pdb.set_trace()
        num_rollouts = action.shape[0]
        horizon = action.shape[1]
        reward_rollout = torch.zeros((num_rollouts), device=action.device)
        reward_activate = torch.ones((num_rollouts), device=action.device)
        
        # import pdb; pdb.set_trace()
        for h in range(horizon):
            
            state_step = state[h+1]
            action_step = action[:, h]
            # import pdb; pdb.set_trace()
            dist = torch.norm(state_step[:, :2] - goal_list[h+1, :2], dim=1)
            # vel_direction = state[h][:,:2] - state[h-1][:,:2]
            # pos_direction = - state[h][:,:2] + goal_list[h, :2] 
            # dot_product = (vel_direction * pos_direction).sum(dim=1)
            # cos_angle = dot_product / (torch.norm(pos_direction, dim=1) * torch.norm(vel_direction, dim=1) + 1e-7)
            vel_diff = torch.norm(state_step[:, 3:4] - defaul_speed, dim=1)
            reward = -dist - 0.0 * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1)
            # reward = - 0.4 * dist - 0.0 * torch.norm(action_step, dim=1) - 0.0 * vel_diff - 0.1 * torch.log(1 + dist)
            # reward = - 0.4 * dist
            reward_rollout += reward *(discount ** h) * reward_activate
        return reward_rollout
    return reward

#----------------- rollout functions -----------------#

def rollout_fn_select(model_struct, model, dt, L, LR):
    
    def rollout_fn_nn_heading(obs_history, last_state, action):
        dt_nn = dt
        print(dt_nn)
        n_rollouts = action.shape[0]
        # import pdb; pdb.set_trace()
        # print("obs_hist", obs_history.shape)
        obs_hist = obs_history.clone()
        obs_hist[:, :, :2] -= obs_hist[:, 0, :2].unsqueeze(1)
        output_dot = model.predict(obs_hist.view(n_rollouts, -1))
        output_dot = fold_angle_tensor(output_dot, idx=0)
        next_state = last_state.clone()
        next_state[:, 0] += torch.cos(last_state[:, 2] + output_dot[:, 0]) * last_state[:, 3] * dt_nn 
        next_state[:, 1] += torch.sin(last_state[:, 2] + output_dot[:, 0]) * last_state[:, 3] * dt_nn 
        new_delta = torch.atan(torch.tan(output_dot[:, 0])*L/LR)  
        next_state[:, 2] += last_state[:, 3]*torch.cos(output_dot[:, 0])/L*torch.tan(new_delta) * dt_nn
        next_state[:, 3] += output_dot[:, 1]
        return next_state
    
    def rollout_fn_nn_heading_psi(obs_history, last_state, action):
        dt_nn = dt
        n_rollouts = action.shape[0]
        
        output_dot = model.predict(obs_history.view(n_rollouts, -1))
        next_state = last_state + 0.
        new_beta = torch.atan2(output_dot[:, 1], output_dot[:, 0])
        psi_dot = torch.atan2(output_dot[:, 3], output_dot[:, 2])
        next_state[:, 0] += torch.cos(last_state[:, 2] + new_beta) * last_state[:, 3] * dt_nn 
        next_state[:, 1] += torch.sin(last_state[:, 2] + new_beta) * last_state[:, 3] * dt_nn 
        next_state[:, 2] += psi_dot
        next_state[:, 3] += output_dot[:, 4]
        return next_state
    
    def rollout_fn_nn_end2end(obs_history, last_state, action):
        n_rollouts = action.shape[0]
        n_traj = obs_history.shape[1]
        obs_hist = obs_history.clone()
        obs_hist[:, :, :2] -= obs_hist[:, 0, :2].unsqueeze(1)
        # old_obs = normalize_angle_tensor(obs_hist, idx=2)
        old_obs = obs_hist + 0.
        # output_dot = fold_angle_tensor(model.predict(old_obs.view(n_rollouts, -1)), idx=2) + last_state
        output_dot = model.predict(old_obs.view(n_rollouts, -1)) + last_state
        output_dot = normalize_angle_tensor(output_dot, idx=2)
        next_state = fold_angle_tensor(output_dot, idx=2)
        return next_state
    
    def rollout_fn_kbm(obs_history, state, action, debug=False):
        next_state = model.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], 
                                    action[:, 0], action[:, 1])
        return torch.stack(next_state, dim=1), {}

    def rollout_fn_dbm(obs_history, state, action, debug=False):
        assert state.shape[1] == 6
        assert action.shape[1] == 2
        # need state = [x, y, psi, vx, vy, omega]
        # omega = obs_history[:, -1, 2] - obs_history[:, -2, 2]
        # omega = torch.atan2(torch.sin(omega), torch.cos(omega)) / dt
        # v_vec = obs_history[:, -1, :2] - obs_history[:, -2, :2]
        # vx = v_vec[:, 0] * torch.cos(state[:, 2]) + v_vec[:, 1] * torch.sin(state[:, 2])
        # vy = v_vec[:, 1] * torch.cos(state[:, 2]) - v_vec[:, 0] * torch.sin(state[:, 2])
        # vx = state[:, 3] * torch.cos(state[:, 2])
        # vy = state[:, 3] * torch.sin(state[:, 2])
        next_state = model.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5], action[:, 0], action[:, 1])
        # next state is [x, y, psi, vx, vy, omega]
        next_state = torch.stack(next_state, dim=1)
        # next_state = next_state[:, :4]
        return next_state, {}
    
    def rollout_fn_nn_phyx_kbm(obs_history, last_state, action, debug=False):
        n_rollouts = action.shape[0]
        # import pdb; pdb.set_trace()
        # print("obs_hist", obs_history.shape)
        obs_hist = obs_history.clone()
        obs_hist[:, :, :2] -= obs_hist[:, 0, :2].unsqueeze(1)
        old_obs = normalize_angle_tensor(obs_hist, idx=2)
        # old_obs = obs_hist + 0.
        debug_info = {}
        output = model.predict(old_obs.view(n_rollouts, -1))
        
        if debug:
            debug_info['x'] = old_obs.view(n_rollouts, -1).detach().cpu().numpy()
            debug_info['y'] = output.detach().cpu().numpy()
            
        x = last_state[:, 0]
        y = last_state[:, 1]
        psi = last_state[:, 2]
        v = last_state[:, 3]
        throttle = action[:, 0]
        steer = action[:, 1]
        oLF = output[:, 0]
        oLR = output[:, 1]
        odelta_proj = output[:, 2]
        odelta_shift = output[:, 3]
        ovel_proj = output[:, 4]
        ovel_shift = output[:, 5]
        beta = torch.atan(torch.tan(odelta_proj * steer + odelta_shift) * oLF / (oLF + oLR))
        pred_next = torch.zeros_like(last_state).to(last_state.device)
        # outputs [LF, LR, delta_proj, delta_shift, vel_shift, vel_proj]
        # tragets [dx, dy, cos(dpsi), sin(dpsi), dv]
        pred_next[:, 0] = x + v * torch.cos(psi + beta) * dt
        pred_next[:, 1] = y + v * torch.sin(psi + beta) * dt
        pred_next[:, 2] = psi + v / LR * torch.sin(beta) * dt
        pred_next[:, 2] = torch.atan2(torch.sin(pred_next[:, 2]), torch.cos(pred_next[:, 2]))
        pred_next[:, 3] = v + (ovel_proj * throttle + ovel_shift) * dt
        
        return pred_next, debug_info
    
    if model_struct == 'nn-heading':
        return rollout_fn_nn_heading
    elif model_struct == 'nn-heading-psi':
        return rollout_fn_nn_heading_psi
    elif model_struct == 'nn-end2end':
        return rollout_fn_nn_end2end
    elif model_struct == 'nn-end2end-trunk':
        return rollout_fn_nn_end2end_trunk
    elif model_struct == 'kbm':
        return rollout_fn_kbm
    elif model_struct == 'dbm':
        return rollout_fn_dbm
    elif model_struct == 'nn-phyx-kbm':
        return rollout_fn_nn_phyx_kbm
    else:
        raise Exception(f"model_struct {model_struct} not supported!")