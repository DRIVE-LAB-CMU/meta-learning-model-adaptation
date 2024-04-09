import torch
import numpy as np
from dataclasses import dataclass


LF = .16
LR = .15

@dataclass
class KinematicParams:
    num_envs: int
    last_diff_vel: torch.Tensor
    KP_VEL: float
    KD_VEL: float
    MAX_VEL: float
    LF: float = .16
    LR: float = .15
    PROJ_STEER: float = 1.
    SHIFT_STEER: float = .0
    DT: float = .05
    
class KinematicBicycleModel:
    def __init__(self, params: KinematicParams, device) -> None:
        self.params = params
        self.device = device

    def reset(self, ):
        self.params.last_diff_vel = torch.zeros(self.params.num_envs, device=self.device)
        # self.params = self.params._replace(last_diff_vel=torch.zeros(self.params.num_envs, 1, device=self.device))
        # assert self.params.last_diff_vel.device == self.device, f"device: {self.params.last_diff_vel.device}, {self.device}"
        
    
    def calc_diff_vel(self, target_vel: torch.Tensor, vel: torch.Tensor):
        """ In batch
        """
        assert target_vel.shape[0] == self.params.num_envs
        batched_projected_vel = self.params.MAX_VEL * target_vel
        batched_diff_vel = batched_projected_vel - vel
        delta_vel = batched_diff_vel - self.params.last_diff_vel
        # print("pred", f"diff vel: {batched_diff_vel}, maxv: {self.params.MAX_VEL}, vel: {vel}")
        acceleration = self.params.KP_VEL * batched_diff_vel + self.params.KD_VEL * delta_vel
        
        self.params.last_diff_vel = delta_vel
        # self.params = self.params._replace(last_diff_vel=self.params.last_diff_vel)
        # import pdb; pdb.set_trace()
        
        return acceleration
    
    def step(self, 
            batch_pos_x, 
            batch_pos_y,
            batch_psi,
            batch_real_vel,
            batch_target_vel,
            batch_target_steer,
        ):
        """ In batch
        """
        #import pdb; pdb.set_trace()
        real_steer = batch_target_steer * self.params.PROJ_STEER + self.params.SHIFT_STEER
        beta = torch.atan(self.params.LR / (self.params.LR + self.params.LF) * torch.tan(real_steer))
        # print("pred", f"steer: {real_steer}, beta: {beta}")
        next_pos_x = batch_pos_x + batch_real_vel * torch.cos(batch_psi + beta) * self.params.DT
        next_pos_y = batch_pos_y + batch_real_vel * torch.sin(batch_psi + beta) * self.params.DT
        
        diff_yaw = batch_real_vel * torch.sin(beta) / self.params.LR
        # print("pred", f"real vel: {batch_real_vel}, target vel: {batch_target_vel}")
        
        acc = self.calc_diff_vel(batch_target_vel, batch_real_vel)        

        next_yaw = batch_psi + diff_yaw * self.params.DT
        next_yaw = torch.atan2(torch.sin(next_yaw), torch.cos(next_yaw))

        next_vel = batch_real_vel + acc * self.params.DT
        # import pdb; pdb.set_trace()
        # print("pred", f"diff vel {acc}, dyaw: {diff_yaw}")
        return next_pos_x, next_pos_y, next_yaw, next_vel
    
    def single_step_numpy_gym(self, obs, action):
        """
            For compatible with env.Navigation2
        """
        batch_pos_x = torch.tensor(obs[0:1], device=self.device).unsqueeze(0)
        batch_pos_y = torch.tensor(obs[1:2], device=self.device).unsqueeze(0)
        batch_psi = torch.tensor([np.arctan2(obs[3],obs[2])], device=self.device).unsqueeze(0)
        # batch_psi = torch.Tensor(obs[2:3], device=self.device).unsqueeze(0)
        batch_real_vel = torch.tensor(obs[4:5], device=self.device).unsqueeze(0)
        batch_target_vel = torch.tensor(action[0:1], device=self.device).unsqueeze(0)
        batch_target_steer = torch.tensor(action[1:2], device=self.device).unsqueeze(0)
        next_pos_x, next_pos_y, next_yaw, next_vel = self.step(
            batch_pos_x,
            batch_pos_y,
            batch_psi,
            batch_real_vel,
            batch_target_vel,
            batch_target_steer,
        )
        return np.array([
            next_pos_x[0].item(),
            next_pos_y[0].item(),
            next_yaw[0].item(),
            next_vel[0].item(),
        ])
        
    def single_step_numpy(self, obs, action):
        assert obs.shape[0] == 4
        batch_pos_x = torch.tensor(obs[0:1], device=self.device).unsqueeze(0)
        batch_pos_y = torch.tensor(obs[1:2], device=self.device).unsqueeze(0)
        batch_psi = torch.tensor(obs[2:3], device=self.device).unsqueeze(0)
        batch_real_vel = torch.tensor(obs[3:4], device=self.device).unsqueeze(0)
        batch_target_vel = torch.tensor(action[0:1], device=self.device).unsqueeze(0)
        batch_target_steer = torch.tensor(action[1:2], device=self.device).unsqueeze(0)
        next_pos_x, next_pos_y, next_yaw, next_vel = self.step(
            batch_pos_x,
            batch_pos_y,
            batch_psi,
            batch_real_vel,
            batch_target_vel,
            batch_target_steer,
        )
        return np.array([
            next_pos_x[0].item(),
            next_pos_y[0].item(),
            next_yaw[0].item(),
            next_vel[0].item(),
        ])
        
class BikePID:
    lf = LF
    lr = LR
    dr = lr / (lr + lf)
    # Integrate = 0.
    last_vel = 0.
    # Kp = 2.
    # Ki = .00
    # Kd = .05

    def __init__(self):
        self.reset()

    def reset(self):
        # self.Integrate = 0.
        self.last_vel = 0.

    def calc_dvel(self, MaxVel, Kp, Kd, target_vel, vel):
        diff_v = MaxVel * target_vel - vel
        # self.Integrate += diff_v
        # print(f"diff vel: {diff_v}, maxv: {MaxVel}, vel: {vel}")
        acc = Kp * diff_v + Kd * (diff_v - self.last_vel)
                        # + self.Ki * self.Integrate \
        self.last_v = diff_v
        return acc

    def step(self, 
             pos_x: float,
             pos_y: float,
             psi: float,
             vel: float,
             target_vel: float,
             steer: float,
             dt: float,
             max_steer = .34,
             MaxVel = 5.5,
             Kp = 4.,
             Kd = .05,
             # Fx = 0,
             # Fy = 0,
             proj_steer = .34,
             shift_steer = .0,
             ):

        steer = proj_steer * steer + shift_steer
        beta = np.arctan(self.dr*np.tan(steer))
        
        dposx = vel * np.cos(psi+beta)
        dposy = vel * np.sin(psi+beta)
        
        # dyaw = vel * np.cos(beta) / (self.lr+self.lf)*np.tan(steer)
        dyaw = vel * np.sin(beta)/self.lr
        # print(f"real vel: {vel}, target vel: {target_vel}")
        dvel = self.calc_dvel(MaxVel, Kp, Kd, target_vel, vel)

        next_yaw = psi + dyaw * dt
        next_yaw = np.arctan2(
                            np.sin(next_yaw),
                            np.cos(next_yaw))

        # print(f"diff vel: {dvel}, dyaw: {dyaw}")
        
        
        return np.array([
            pos_x + dposx * dt,
            pos_y + dposy * dt,
            next_yaw,
            vel + dvel * dt,
        ])
