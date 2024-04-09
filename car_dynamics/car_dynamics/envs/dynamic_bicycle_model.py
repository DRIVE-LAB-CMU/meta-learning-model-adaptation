import torch
import numpy as np
from dataclasses import dataclass
import time 

@dataclass
class DynamicParams:
    num_envs: int
    LF: float = .11
    LR: float = .23
    MASS: float = 1.
    DT: float = .05
    K_RFY: float = 20.
    K_FFY: float = 20.
    Iz: float = .95*1.*0.34**2
    Ta: float = 5.5
    Tb: float = -1.
    Sa: float = 0.36
    Sb: float = 0.03
    
    def to_dict(self):
        return {
            'num_envs': self.num_envs,
            'LF': self.LF,
            'LR': self.LR,
            'MASS': self.MASS,
            'DT': self.DT,
            'K_RFY': self.K_RFY,
            'K_FFY': self.K_FFY,
            'Iz': self.Iz,
            'Ta': self.Ta,
            'Tb': self.Tb,
            'Sa': self.Sa,
            'Sb': self.Sb,
        }
    
class DynamicBicycleModel:
    def __init__(self, params: DynamicParams, device) -> None:
        self.params = params
        self.device = device
        self.batch_Ta = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.Ta
        self.batch_Tb = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.Tb
        self.batch_Sa = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.Sa
        self.batch_Sb = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.Sb
        self.batch_LF = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.LF
        self.batch_LR = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.LR
        self.batch_MASS = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.MASS
        self.batch_DT = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.DT
        self.batch_K_RFY = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.K_RFY
        self.batch_K_FFY = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.K_FFY
        self.batch_Iz = torch.ones(params.num_envs, device=device, dtype=torch.float32) * params.Iz 

    def reset(self, ):
        ...
               
    def step(self,
             batch_x,
             batch_y,
             batch_psi,
             batch_vx,
             batch_vy,
             batch_omega,
             batch_target_vel,
             batch_target_steer,
        ):
        return dbm(batch_x, 
                   batch_y,
                   batch_psi,
                   batch_vx,
                   batch_vy,
                   batch_omega,
                   batch_target_vel,
                   batch_target_steer,
                   self.batch_Ta,
                   self.batch_Tb,
                   self.batch_Sa,
                   self.batch_Sb,
                   self.batch_LF,
                   self.batch_LR,
                   self.batch_MASS,
                   self.batch_DT,
                   self.batch_K_RFY,
                   self.batch_K_FFY,
                   self.batch_Iz,
            )
    
    def step_torch(self, 
            batch_x, 
            batch_y,
            batch_psi,
            batch_vx, 
            batch_vy,
            batch_omega,
            batch_target_vel,
            batch_target_steer,
        ):
        """ In batch
        """
        #TODO: check with @dvij
        # print(self.batch_Sa.shape, batch_target_steer.shape, self.batch_Sb.shape)
        
        
        # st = time.time()
        steer = batch_target_steer * self.batch_Sa + self.batch_Sb 
        
        
        prev_vel = torch.sqrt(batch_vx**2 + batch_vy**2)
        # print("vel omega", prev_vel, batch_omega)
        # print(batch_target_vel.shape, self.batch_Ta.shape, self.batch_Tb.shape, prev_vel.shape)
        throttle = batch_target_vel * self.batch_Ta + self.batch_Tb * prev_vel
        # print(throttle)
        # print(batch_psi)
        next_x = batch_x + (batch_vx * torch.cos(batch_psi) - batch_vy * torch.sin(batch_psi)) * self.batch_DT
        # print(batch_x.shape, self.batch_DT.shape)
        next_y = batch_y + (batch_vx * torch.sin(batch_psi) + batch_vy * torch.cos(batch_psi)) * self.batch_DT
        next_psi = batch_psi + batch_omega * self.batch_DT
        
        # print("dbm part 1", time.time() - st)
        # st = time.time()
        # next_psi = torch.atan2(torch.sin(next_psi), torch.cos(next_psi))
        alpha_f = steer - torch.atan2(self.batch_LF * batch_omega + batch_vy, torch.maximum(batch_vx,batch_vx*.0+.5))
        alpha_r = torch.atan2(self.batch_LR * batch_omega - batch_vy, torch.maximum(batch_vx,batch_vx*.0+.5))
        
        # print("dbm part 2", time.time() - st)
        # st = time.time()
        # print(alpha_f, alpha_r)
        # print(batch_vx, batch_vy)
        F_rx = throttle
        # print("Frx", F_rx, "alpha f", alpha_f, "alpha r", alpha_r, 'vx', batch_vx, 'vy', batch_vy, 'omega', batch_omega, 'steer', steer)
        F_fy = self.batch_K_FFY * alpha_f
        F_ry = self.batch_K_RFY * alpha_r
        
        next_vx = batch_vx + (F_rx - F_fy * torch.sin(steer) + batch_vy * batch_omega * self.batch_MASS) * self.batch_DT / self.batch_MASS
        next_vy = batch_vy + (F_ry + F_fy * torch.cos(steer) - batch_vx * batch_omega * self.batch_MASS) * self.batch_DT / self.batch_MASS
        next_omega = batch_omega + (F_fy*self.batch_LF*torch.cos(steer) - F_ry*self.batch_LR) * self.batch_DT / self.batch_Iz
        # print("dbm part 3", time.time() - st)
        # print("dbm step time", time.time() - st)
        # print(next_x.shape, next_y.shape, next_psi.shape, next_vx.shape, next_vy.shape, next_omega.shape)
        return next_x, next_y, next_psi, next_vx, next_vy, next_omega
    
    def step_(self, 
            batch_x, 
            batch_y,
            batch_psi,
            batch_vx, 
            batch_vy,
            batch_omega,
            batch_target_vel,
            batch_target_steer,
        ):
        """ In batch
        """
        raise DeprecationWarning
        #TODO: check with @dvij
        steer = batch_target_steer * self.params.Sa + self.params.Sb 
        
        
        prev_vel = torch.sqrt(batch_vx**2 + batch_vy**2)
        # print("vel omega", prev_vel, batch_omega)
        
        throttle = batch_target_vel * self.params.Ta + self.params.Tb * prev_vel
        # print(throttle)
        # print(batch_psi)
        next_x = batch_x + (batch_vx * torch.cos(batch_psi) - batch_vy * torch.sin(batch_psi)) * self.params.DT
        next_y = batch_y + (batch_vx * torch.sin(batch_psi) + batch_vy * torch.cos(batch_psi)) * self.params.DT
        next_psi = batch_psi + batch_omega * self.params.DT
        # next_psi = torch.atan2(torch.sin(next_psi), torch.cos(next_psi))
        alpha_f = steer - torch.atan2(self.params.LF * batch_omega + batch_vy, torch.maximum(batch_vx,batch_vx*.0+.5))
        alpha_r = torch.atan2(self.params.LR * batch_omega - batch_vy, torch.maximum(batch_vx,batch_vx*.0+.5))
        # print(alpha_f, alpha_r)
        # print(batch_vx, batch_vy)
        F_rx = throttle
        # print("Frx", F_rx, "alpha f", alpha_f, "alpha r", alpha_r, 'vx', batch_vx, 'vy', batch_vy, 'omega', batch_omega, 'steer', steer)
        F_fy = self.params.K_FFY * alpha_f
        F_ry = self.params.K_RFY * alpha_r
        
        next_vx = batch_vx + (F_rx - F_fy * torch.sin(steer) + batch_vy * batch_omega * self.params.MASS) * self.params.DT / self.params.MASS
        next_vy = batch_vy + (F_ry + F_fy * torch.cos(steer) - batch_vx * batch_omega * self.params.MASS) * self.params.DT / self.params.MASS
        next_omega = batch_omega + (F_fy*self.params.LF*torch.cos(steer) - F_ry*self.params.LR) * self.params.DT / self.params.Iz
        return next_x, next_y, next_psi, next_vx, next_vy, next_omega
        
    def single_step_numpy(self, obs, action):
        assert obs.shape[0] == 6
        assert action.shape[0] == 2
        batch_x = torch.tensor(obs[0:1], device=self.device).unsqueeze(0)
        batch_y = torch.tensor(obs[1:2], device=self.device).unsqueeze(0)
        batch_psi = torch.tensor(obs[2:3], device=self.device).unsqueeze(0)
        batch_vx = torch.tensor(obs[3:4], device=self.device).unsqueeze(0)
        batch_vy = torch.tensor(obs[4:5], device=self.device).unsqueeze(0)
        batch_omega = torch.tensor(obs[5:6], device=self.device).unsqueeze(0)
        
        batch_target_vel = torch.tensor(action[0:1], device=self.device).unsqueeze(0)
        batch_target_steer = torch.tensor(action[1:2], device=self.device).unsqueeze(0)
        next_x, next_y, next_psi, next_vx, next_vy, next_omega = self.step(
            batch_x,
            batch_y,
            batch_psi,
            batch_vx,
            batch_vy,
            batch_omega,
            batch_target_vel,
            batch_target_steer,
        )
        return np.array([
            next_x.cpu().numpy().squeeze(),
            next_y.cpu().numpy().squeeze(),
            next_psi.cpu().numpy().squeeze(),
            next_vx.cpu().numpy().squeeze(),
            next_vy.cpu().numpy().squeeze(),
            next_omega.cpu().numpy().squeeze(),
        ])



@torch.jit.script
def dbm(
    batch_x, 
    batch_y,
    batch_psi,
    batch_vx, 
    batch_vy,
    batch_omega,
    batch_target_vel,
    batch_target_steer,
    batch_Ta,
    batch_Tb,
    batch_Sa,
    batch_Sb,
    batch_LF,
    batch_LR,
    batch_MASS,
    batch_DT,
    batch_K_RFY,
    batch_K_FFY,
    batch_Iz,
):
    steer = batch_target_steer * batch_Sa + batch_Sb 
        
    prev_vel = torch.sqrt(batch_vx**2 + batch_vy**2)
    # print("vel omega", prev_vel, batch_omega)
    # print(batch_target_vel.shape, batch_Ta.shape, batch_Tb.shape, prev_vel.shape)
    throttle = batch_target_vel * batch_Ta + batch_Tb * prev_vel
    # print(throttle)
    # print(batch_psi)
    next_x = batch_x + (batch_vx * torch.cos(batch_psi) - batch_vy * torch.sin(batch_psi)) * batch_DT
    # print(batch_x.shape, batch_DT.shape)
    next_y = batch_y + (batch_vx * torch.sin(batch_psi) + batch_vy * torch.cos(batch_psi)) * batch_DT
    next_psi = batch_psi + batch_omega * batch_DT
    
    # print("dbm part 1", time.time() - st)
    # st = time.time()
    # next_psi = torch.atan2(torch.sin(next_psi), torch.cos(next_psi))
    alpha_f = steer - torch.atan2(batch_LF * batch_omega + batch_vy, torch.maximum(batch_vx,batch_vx*.0+.5))
    alpha_r = torch.atan2(batch_LR * batch_omega - batch_vy, torch.maximum(batch_vx,batch_vx*.0+.5))
    
    # print("dbm part 2", time.time() - st)
    # st = time.time()
    # print(alpha_f, alpha_r)
    # print(batch_vx, batch_vy)
    F_rx = throttle
    # print("Frx", F_rx, "alpha f", alpha_f, "alpha r", alpha_r, 'vx', batch_vx, 'vy', batch_vy, 'omega', batch_omega, 'steer', steer)
    F_fy = batch_K_FFY * alpha_f
    F_ry = batch_K_RFY * alpha_r
    
    next_vx = batch_vx + (F_rx - F_fy * torch.sin(steer) + batch_vy * batch_omega * batch_MASS) * batch_DT / batch_MASS
    next_vy = batch_vy + (F_ry + F_fy * torch.cos(steer) - batch_vx * batch_omega * batch_MASS) * batch_DT / batch_MASS
    next_omega = batch_omega + (F_fy*batch_LF*torch.cos(steer) - F_ry*batch_LR) * batch_DT / batch_Iz
    # print("dbm part 3", time.time() - st)
    # print("dbm step time", time.time() - st)
    # print(next_x.shape, next_y.shape, next_psi.shape, next_vx.shape, next_vy.shape, next_omega.shape)
    return next_x, next_y, next_psi, next_vx, next_vy, next_omega