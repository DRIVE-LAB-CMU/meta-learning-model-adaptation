import torch
from car_dynamics.models_torch.utils import normalize_angle_tensor, fold_angle_tensor


#----------------- reward functions -----------------#

def reward_track_fn(goal_list: torch.Tensor, defaul_speed: float, sim):
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
        for h in range(0,horizon):
            
            state_step = state[h+1]
            action_step = action[:, h]
            if h>0:
                prev_action_step = action[:, h-1]
            # 
            # import pdb; pdb.set_trace()
            dist = torch.norm(state_step[:, :2] - goal_list[h+1, :2], dim=1)**2
            theta = state_step[:,2]
            theta_diff = torch.atan2(torch.sin(theta - goal_list[h+1, 2]), torch.cos(theta - goal_list[h+1, 2]))
            # vel_direction = state[h][:,:2] - state[h-1][:,:2]
            # pos_direction = - state[h][:,:2] + goal_list[h, :2] 
            # dot_product = (vel_direction * pos_direction).sum(dim=1)
            # cos_angle = dot_product / (torch.norm(pos_direction, dim=1) * torch.norm(vel_direction, dim=1) + 1e-7)
            # vel_diff = torch.norm(state_step[:, 3:4] - defaul_speed, dim=1)
            vel_diff = torch.abs(state_step[:, 3] - goal_list[h+1, 3]) * (state_step[:, 3] > goal_list[h+1, 3])
            vel_diff += torch.abs(state_step[:, 3] - 0.5) * (state_step[:, 3] < 0.5)
            if h < horizon - 1:
                if sim == 'unity' :
                    reward = -dist - 3. * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1) -theta_diff**2 #- 10.*torch.norm(action_step-prev_action_step,dim=1)
                elif sim == 'vicon' :
                    reward = -30.*dist - 3. * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1) -theta_diff**2 #- 10.*torch.norm(action_step-prev_action_step,dim=1)
                else :
                    reward = -10.*dist - 3. * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1) -theta_diff**2 #- 10.*torch.norm(action_step-prev_action_step,dim=1)
            else :
                if sim == 'unity' :
                    reward = -dist - 3. * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1) -theta_diff**2
                elif sim == 'vicon' :
                    reward = -30.*dist - 3. * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1) -theta_diff**2
                else :
                    reward = -10.*dist - 3. * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1) -theta_diff**2
            # reward = - 0.4 * dist - 0.0 * torch.norm(action_step, dim=1) - 0.0 * vel_diff - 0.1 * torch.log(1 + dist)
            # reward = - 0.4 * dist
            reward_rollout += reward *(discount ** h) * reward_activate
        return reward_rollout
    return reward

#----------------- rollout functions -----------------#

def rollout_fn_select(model_struct, models, dt, L, LR, n_ensembles=3):
    
    def rollout_fn_nn_heading(obs_history, last_state, action, one_hot_delay=None):
        model = models[0]
        dt_nn = dt
        # print(dt_nn)
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
    
    def rollout_fn_nn_heading_psi(obs_history, last_state, action, one_hot_delay=None):
        model = models[0]
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
    
    def rollout_fn_nn_end2end(obs_history, last_state, action, one_hot_delay=None):
        model = models[0]
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
    
    def rollout_fn_kbm(obs_history, state, action, debug=False, one_hot_delay=None):
        model = models[0]
        next_state = model.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], 
                                    action[:, 0], action[:, 1])
        return torch.stack(next_state, dim=1), {}

    def rollout_fn_dbm(obs_history, state, action, debug=False, one_hot_delay=None):
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
        model = models[0]
        next_state = model.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5], action[:, 0], action[:, 1])
        # next state is [x, y, psi, vx, vy, omega]
        next_state = torch.stack(next_state, dim=1)
        # next_state = next_state[:, :4]
        return next_state, {}
    
    def rollout_fn_nn(obs_history, state, action, debug=False, one_hot_delay=None):
        # import pdb; pdb.set_trace()
        # print(state.shape)
        assert state.shape[1] == 6
        assert action.shape[1] == 2
        _X = obs_history[:,:-1,3:].reshape(obs_history.shape[0], -1)
        X_ = torch.concat((state[:,3:], action),dim=1).double()
        X = torch.cat((_X, X_), dim=1)
        # print(one_hot_delay)
        X = torch.cat((X, one_hot_delay.repeat((state.shape[0],1))), dim=1)
        # if debug:
        #     print(X[0])
        # print(X.shape)
        next_state = torch.tensor(state)
        # gradX = 0.
        outs = []
        for i in range(len(models)):
            model = models[i]        
            # gradX += model(X).detach()
            outs.append(model(X).detach())
        outs = torch.stack(outs,dim=0)
        # print(outs.shape)
        gradX = torch.mean(outs,dim=0)
        if outs.shape[0] > 1:
            varX = torch.sum(torch.var(outs,dim=0),dim=1)
        else :
            varX = torch.zeros(state.shape[0]).to(state.device)
        # print(torch.min(gradX[:,2]), torch.max(gradX[:,2]), torch.mean(gradX[:,2]))
        next_state[:,3] += gradX[:,0] * dt
        next_state[:,4] += gradX[:,1] * dt
        next_state[:,5] += gradX[:,2] * dt
        next_state[:,2] += state[:,5] * dt
        # next_state = normalize_angle_tensor(next_state,idx=2)
        next_state[:,0] += state[:,3] * torch.cos(state[:,2]) * dt - state[:,4] * torch.sin(state[:,2]) * dt
        next_state[:,1] += state[:,3] * torch.sin(state[:,2]) * dt + state[:,4] * torch.cos(state[:,2]) * dt
        
        # next_state = next_state[:, :4]
        # print(next_state.shape)
        return next_state, {'var': varX}
    
    def rollout_fn_nn_lstm(obs_history, state, action, h_0s=None, c_0s=None):
        assert state.shape[1] == 6
        assert action.shape[1] == 2
        X = torch.concat((state[:,3:], action),dim=1).double().unsqueeze(1)
        next_state = torch.tensor(state)
        # gradX = 0.
        outs = []
        h_ns = []
        c_ns = []
        for i in range(len(models)):
            model = models[i]        
            # gradX += model(X).detach()
            # print(X.shape,h_0s[i].shape,c_0s[i].shape)
            out, (h_n,c_n) = model(X,h_0=h_0s[i],c_0=c_0s[i])
            h_ns.append(h_n)
            c_ns.append(c_n)
            outs.append(out.detach()[:,-1])
        outs = torch.stack(outs,dim=0)
        # print(outs.shape)
        gradX = torch.mean(outs,dim=0)
        if outs.shape[0] > 1:
            varX = torch.sum(torch.var(outs,dim=0),dim=1)
        else :
            varX = torch.zeros(state.shape[0]).to(state.device)
        # print(torch.min(gradX[:,2]), torch.max(gradX[:,2]), torch.mean(gradX[:,2]))
        next_state[:,3] += gradX[:,0] * dt
        next_state[:,4] += gradX[:,1] * dt
        next_state[:,5] += gradX[:,2] * dt
        next_state[:,2] += state[:,5] * dt
        # next_state = normalize_angle_tensor(next_state,idx=2)
        next_state[:,0] += state[:,3] * torch.cos(state[:,2]) * dt - state[:,4] * torch.sin(state[:,2]) * dt
        next_state[:,1] += state[:,3] * torch.sin(state[:,2]) * dt + state[:,4] * torch.cos(state[:,2]) * dt
        
        # next_state = next_state[:, :4]
        # print(next_state.shape)
        return next_state, {'var': varX, 'h_ns': h_ns, 'c_ns': c_ns}
    
    def rollout_fn_nn_phyx_kbm(obs_history, last_state, action, debug=False):
        model = models[0]
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
    elif model_struct == 'nn':
        return rollout_fn_nn
    elif model_struct == 'nn-lstm':
        return rollout_fn_nn_lstm
    elif model_struct == 'kbm':
        return rollout_fn_kbm
    elif model_struct == 'dbm':
        return rollout_fn_dbm
    elif model_struct == 'nn-phyx-kbm':
        return rollout_fn_nn_phyx_kbm
    else:
        raise Exception(f"model_struct {model_struct} not supported!")