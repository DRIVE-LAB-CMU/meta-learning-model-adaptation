import torch
import numpy as np
from car_dynamics.models_torch.utils import normalize_angle_tensor, fold_angle_tensor

def sig(x):
    return 1/(1 + np.exp(-x))

def k_step_prediction_error(model, obs_list, action_list, history_length, predict_horizon, device, model_struct, dt, L, LR,normalize_pos=True):
    
    ## slice batch data
    batch_slice_obs = []
    batch_slice_action = []
    for i in range(history_length+predict_horizon-1, len(obs_list)):
        batch_slice_obs.append(obs_list[i-history_length-predict_horizon+1:i+1, :4])
        batch_slice_action.append(action_list[i-history_length-predict_horizon+1:i+1, :2])
    batch_slice_obs = np.array(batch_slice_obs)
    batch_slice_action = np.array(batch_slice_action)
    batch_size = batch_slice_obs.shape[0]
    ## stackup history
    # print(history_length, predict_horizon, batch_size)
    pred_k_trajectory = np.zeros((batch_size, predict_horizon+history_length, 4))
    
    # print(batch_slice_obs.shape)
    pred_k_trajectory[:, :history_length] = batch_slice_obs[:, :history_length]
    
    if model_struct == 'kbm':
        model.reset()
    
    obs_debug = []
    # # NN Predict k steps
    for step in range(predict_horizon):
        state_hist = np.array(pred_k_trajectory[:, step:step+history_length])
        if normalize_pos:
            state_hist[:, :, :2] -= state_hist[:, 0, :2][:, None, :]
        action_hist = np.array(batch_slice_action[:, step:step+history_length])
        old_obs_np = np.concatenate((state_hist,action_hist),axis=2)
        
        if model_struct == 'nn-heading':
            old_obs = torch.tensor(old_obs_np, device=device, dtype=torch.float32)
            old_obs = old_obs.view(batch_size, -1)
            # print("old obs shape", old_obs.shape)
            nn_output = model.predict(old_obs)
            nn_output = fold_angle_tensor(nn_output, idx=0).detach().cpu().numpy()
            assert nn_output.shape[1] == 2
            curr_x = pred_k_trajectory[:, history_length+step-1, 0]
            curr_y = pred_k_trajectory[:, history_length+step-1, 1]
            curr_psi = pred_k_trajectory[:, history_length+step-1, 2]
            curr_vel = pred_k_trajectory[:, history_length+step-1, 3]
            
            next_x =curr_x + np.cos(curr_psi + nn_output[:, 0]) * dt * curr_vel
            next_y = curr_y + np.sin(curr_psi + nn_output[:, 0]) * dt * curr_vel
            next_delta = np.arctan2(L*np.tan(nn_output[:, 0]), LR)
            next_psi = curr_psi + curr_vel*np.cos(nn_output[:, 0])/L*np.tan(next_delta) * dt
            next_psi = np.arctan2(np.sin(next_psi), np.cos(next_psi))
            next_vel = curr_vel + nn_output[:, 1]
            next_pred_state = np.column_stack((next_x, next_y, next_psi, next_vel))
            
        elif model_struct == 'nn-heading-psi':
            old_obs = torch.tensor(old_obs_np, device=device, dtype=torch.float32)
            old_obs = old_obs.view(batch_size, -1)
            # print("old obs shape", old_obs.shape)
            nn_output = model.predict(old_obs)
            nn_output = fold_angle_tensor(nn_output, idx=0)
            nn_output = fold_angle_tensor(nn_output, idx=1).detach().cpu().numpy()
            assert nn_output.shape[1] == 3
            curr_x = pred_k_trajectory[:, history_length+step-1, 0]
            curr_y = pred_k_trajectory[:, history_length+step-1, 1]
            curr_psi = pred_k_trajectory[:, history_length+step-1, 2]
            curr_vel = pred_k_trajectory[:, history_length+step-1, 3]
            
            next_x =curr_x + np.cos(curr_psi + nn_output[:, 0]) * dt * curr_vel
            next_y = curr_y + np.sin(curr_psi + nn_output[:, 0]) * dt * curr_vel
            # next_delta = np.arctan2(L*np.tan(nn_output[:, 0]), LR)
            # next_psi = curr_psi + curr_vel*np.cos(nn_output[:, 0])/L*np.tan(next_delta) * dt
            next_psi = curr_psi + nn_output[:, 1]
            next_psi = np.arctan2(np.sin(next_psi), np.cos(next_psi))
            next_vel = curr_vel + nn_output[:, 2]
            next_pred_state = np.column_stack((next_x, next_y, next_psi, next_vel))
            
        elif model_struct == 'nn-end2end':
            # old_obs = normalize_angle_tensor(torch.tensor(old_obs_np, device=device), idx=2)
            old_obs = torch.tensor(old_obs_np, device=device,dtype=torch.float32)
            old_obs = old_obs.view(batch_size, -1)
            # print("in", old_obs.shape)
            obs_debug.append(old_obs)
            nn_output = model.predict(old_obs) + torch.tensor(pred_k_trajectory[:, history_length+step-1], device=device)
            # nn_output = fold_angle_tensor(nn_output, idx=2).detach().cpu().numpy()
            next_pred_state = nn_output.detach().cpu().numpy()
        elif model_struct == 'kbm':
            old_state = torch.tensor(pred_k_trajectory[:, step+history_length-1], device=device)
            old_action = torch.tensor(batch_slice_action[:, step+history_length-1], device=device)
            next_state = torch.stack(model.step(old_state[:, 0], old_state[:, 1], old_state[:, 2], old_state[:, 3], 
                                old_action[:, 0], old_action[:, 1]), dim=1)
            # print(type(next_state), len(next_state))
            next_pred_state = next_state.detach().cpu().numpy()
        elif model_struct == 'nn-phyx-kbm':
            curr_x = pred_k_trajectory[:, history_length+step-1, 0]
            curr_y = pred_k_trajectory[:, history_length+step-1, 1]
            curr_psi = pred_k_trajectory[:, history_length+step-1, 2]
            curr_vel = pred_k_trajectory[:, history_length+step-1, 3]
            old_state = pred_k_trajectory[:, history_length+step-1]
            curr_throttle = batch_slice_action[:, history_length+step-1, 0]
            curr_steer = batch_slice_action[:, history_length+step-1, 1]
            old_obs = normalize_angle_tensor(torch.tensor(old_obs_np, device=device), idx=2)
            old_obs = old_obs.view(batch_size, -1)
            nn_output = model.predict(old_obs).detach().cpu().numpy()
            oLF = nn_output[:, 0]
            oLR = nn_output[:, 1]
            odelta_proj = nn_output[:, 2]
            odelta_shift = nn_output[:, 3]
            ovel_shift = nn_output[:, 4]
            ovel_proj = nn_output[:, 5]
            beta = np.arctan(np.tan(odelta_proj * curr_steer + odelta_shift) * oLF / (oLF + oLR))
            next_pred_state = np.zeros_like(old_state)
            next_pred_state[:, 0] = curr_x + curr_vel * np.cos(curr_psi + beta) * dt
            next_pred_state[:, 1] = curr_y + curr_vel * np.sin(curr_psi + beta) * dt
            next_pred_state[:, 2] = curr_psi + curr_vel / oLR * np.sin(beta) * dt
            next_pred_state[:, 2] = np.arctan2(np.sin(next_pred_state[:, 2]), np.cos(next_pred_state[:, 2]))
            next_pred_state[:, 3] = curr_vel + (ovel_shift * curr_throttle + ovel_proj) * dt
            
        else:
            raise Exception(f"model_struct {model_struct} not supported!")
        # print("out", nn_output.shape)
        
        pred_k_trajectory[:, history_length+step] = next_pred_state

    # print(batch_slice_obs.shape, pred_k_trajectory.shape)
    mse_predict = ((pred_k_trajectory[:, history_length:] - batch_slice_obs[:, history_length:])**2).mean(axis=1).mean(axis=0)
    mae_predict = np.abs(pred_k_trajectory[:, history_length:] - batch_slice_obs[:, history_length:]).mean(axis=1).mean(axis=0)
    # print(np.abs(pred_k_trajectory[:, history_length:] - batch_slice_obs[:, history_length:]).shape)
    ae_predict = np.abs(pred_k_trajectory[:, history_length:] - batch_slice_obs[:, history_length:]).mean(axis=1)
    return {'mse': mse_predict,'mae': mae_predict, 'ae': ae_predict}, {'pred_k_trajectory': pred_k_trajectory, 'batch_slice_obs': batch_slice_obs, 'batch_slice_action': batch_slice_action, 'obs_debug': obs_debug}

        