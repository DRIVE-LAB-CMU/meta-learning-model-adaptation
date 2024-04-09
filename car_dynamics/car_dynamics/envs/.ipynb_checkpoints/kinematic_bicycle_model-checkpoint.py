import numpy as np

LF = .16
LR = .15


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
             ):

        steer *= max_steer
        
        beta = np.arctan(self.dr*np.tan(steer))
        dposx = vel * np.cos(psi+beta)
        dposy = vel * np.sin(psi+beta)
        # dyaw = vel * np.cos(beta) / (self.lr+self.lf)*np.tan(steer)
        dyaw = vel * np.sin(beta)/self.lr
        dvel = self.calc_dvel(MaxVel, Kp, Kd, target_vel, vel)

        next_yaw = psi + dyaw * dt
        next_yaw = np.arctan2(
                            np.sin(next_yaw),
                            np.cos(next_yaw))

        return np.array([
            pos_x + dposx * dt,
            pos_y + dposy * dt,
            next_yaw,
            vel + dvel * dt,
        ])