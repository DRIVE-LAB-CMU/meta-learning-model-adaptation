import jax
import jax.numpy as jnp
from dataclasses import dataclass
import time 
from functools import partial
import numpy as np

GRAVITY = 9.81
@dataclass
class CarState:
    x: float
    y: float
    psi: float
    vx: float
    vy: float
    omega: float
    
@dataclass
class CarAction:
    target_vel: float
    target_steer: float
    
@dataclass
class DynamicParams:
    num_envs: int
    LF: float = .11
    LR: float = .23
    MASS: float = 4.65
    DT: float = .05
    K_RFY: float = 20.
    K_FFY: float = 20.
    Iz: float = 0.1
    Ta: float = 5.5
    Tb: float = -1.
    Sa: float = 0.36
    Sb: float = 0.03
    mu: float = 3.0
    Cf: float = 1.0
    Cr: float = 1.0
    Bf: float = 60.0
    Br: float = 60.0
    hcom: float = 0.0
    fr: float = 0.0
    delay: int = 4
    
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
            'mu': self.mu,
            'Cf': self.Cf,
            'Cr': self.Cr,
            'Bf': self.Bf,
            'Br': self.Br,
            'hcom': self.hcom,
            'fr': self.fr,
            'delay': self.delay,
        }
    
    

class DynamicBicycleModel:
    def __init__(self, params: DynamicParams,) -> None:
        self.params = params
        self.batch_Ta = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Ta
        self.batch_Tb = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Tb
        self.batch_Sa = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Sa
        self.batch_Sb = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Sb
        self.batch_LF = jnp.ones(params.num_envs, dtype=jnp.float32) * params.LF
        self.batch_LR = jnp.ones(params.num_envs, dtype=jnp.float32) * params.LR
        self.batch_MASS = jnp.ones(params.num_envs, dtype=jnp.float32) * params.MASS
        self.batch_DT = jnp.ones(params.num_envs, dtype=jnp.float32) * params.DT
        self.batch_K_RFY = jnp.ones(params.num_envs, dtype=jnp.float32) * params.K_RFY
        self.batch_K_FFY = jnp.ones(params.num_envs, dtype=jnp.float32) * params.K_FFY
        self.batch_Iz = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Iz 
        self.batch_mu = jnp.ones(params.num_envs, dtype=jnp.float32) * params.mu
        self.batch_Cf = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Cf
        self.batch_Cr = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Cr
        self.batch_Bf = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Bf
        self.batch_Br = jnp.ones(params.num_envs, dtype=jnp.float32) * params.Br
        self.batch_hcom = jnp.ones(params.num_envs, dtype=jnp.float32) * params.hcom
        self.batch_fr = jnp.ones(params.num_envs, dtype=jnp.float32) * params.fr
        self.batch_buffer = np.zeros((params.num_envs, params.delay, 2))
        
    def reset(self, ):
        ...

    
    # @partial(jax.jit, static_argnums=(0,))      
    def step(self,
             batch_x,
             batch_y,
             batch_psi,
             batch_vx,
             batch_vy,
             batch_omega,
             batch_target_vel,
             batch_target_steer,
            #  debug=False
        ):
        
        """ Implement RK 4, with dynamics forward fn: dbm_dxdt """
        # k1
        # self.batch_buffer.at[:,1:,:].set(self.batch_buffer[:,:-1,:])
        # self.batch_buffer.at[:,0,0].set(batch_target_vel)
        # self.batch_buffer.at[:,0,1].set(batch_target_steer)
        # print("Haha?")
            # print(self.batch_buffer)
        k1 = dbm_dxdt(batch_x,
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
                        self.batch_K_RFY,
                        self.batch_K_FFY,
                        self.batch_Iz,
                        self.batch_mu,
                        self.batch_Cf,
                        self.batch_Cr,
                        self.batch_Bf,
                        self.batch_Br,
                        self.batch_hcom,
                        self.batch_fr,
                    )
        # k2
        k2 = dbm_dxdt(batch_x + 0.5 * self.params.DT * k1[0],
                        batch_y + 0.5 * self.params.DT * k1[1],
                        batch_psi + 0.5 * self.params.DT * k1[2],
                        batch_vx + 0.5 * self.params.DT * k1[3],
                        batch_vy + 0.5 * self.params.DT * k1[4],
                        batch_omega + 0.5 * self.params.DT * k1[5],
                        batch_target_vel,
                        batch_target_steer,
                        self.batch_Ta,
                        self.batch_Tb,
                        self.batch_Sa,
                        self.batch_Sb,
                        self.batch_LF,
                        self.batch_LR,
                        self.batch_MASS,
                        self.batch_K_RFY,
                        self.batch_K_FFY,
                        self.batch_Iz,
                        self.batch_mu,
                        self.batch_Cf,
                        self.batch_Cr,
                        self.batch_Bf,
                        self.batch_Br,
                        self.batch_hcom,
                        self.batch_fr,
                    )
        
        # k3
        k3 = dbm_dxdt(batch_x + 0.5 * self.params.DT * k2[0],
                        batch_y + 0.5 * self.params.DT * k2[1],
                        batch_psi + 0.5 * self.params.DT * k2[2],
                        batch_vx + 0.5 * self.params.DT * k2[3],
                        batch_vy + 0.5 * self.params.DT * k2[4],
                        batch_omega + 0.5 * self.params.DT * k2[5],
                        batch_target_vel,
                        batch_target_steer,
                        self.batch_Ta,
                        self.batch_Tb,
                        self.batch_Sa,
                        self.batch_Sb,
                        self.batch_LF,
                        self.batch_LR,
                        self.batch_MASS,
                        self.batch_K_RFY,
                        self.batch_K_FFY,
                        self.batch_Iz,
                        self.batch_mu,
                        self.batch_Cf,
                        self.batch_Cr,
                        self.batch_Bf,
                        self.batch_Br,
                        self.batch_hcom,
                        self.batch_fr,
                    )
        
        # k4
        k4 = dbm_dxdt(batch_x + self.params.DT * k3[0],
                        batch_y + self.params.DT * k3[1],
                        batch_psi + self.params.DT * k3[2],
                        batch_vx + self.params.DT * k3[3],
                        batch_vy + self.params.DT * k3[4],
                        batch_omega + self.params.DT * k3[5],
                        batch_target_vel,
                        batch_target_steer,
                        self.batch_Ta,
                        self.batch_Tb,
                        self.batch_Sa,
                        self.batch_Sb,
                        self.batch_LF,
                        self.batch_LR,
                        self.batch_MASS,
                        self.batch_K_RFY,
                        self.batch_K_FFY,
                        self.batch_Iz,
                        self.batch_mu,
                        self.batch_Cf,
                        self.batch_Cr,
                        self.batch_Bf,
                        self.batch_Br,
                        self.batch_hcom,
                        self.batch_fr,
                    )
        
        next_x = batch_x + self.params.DT / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        next_y = batch_y + self.params.DT / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        next_psi = batch_psi + self.params.DT / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        next_vx = batch_vx + self.params.DT / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        next_vy = batch_vy + self.params.DT / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        next_omega = batch_omega + self.params.DT / 6.0 * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])
        
        print("ha",next_vx)
        return next_x, next_y, next_psi, next_vx, next_vy, next_omega
    
    def step_euler(self,
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
    
    def step_jnp(self, 
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
        
        
        prev_vel = jnp.sqrt(batch_vx**2 + batch_vy**2)
        # print("vel omega", prev_vel, batch_omega)
        # print(batch_target_vel.shape, self.batch_Ta.shape, self.batch_Tb.shape, prev_vel.shape)
        throttle = batch_target_vel * self.batch_Ta + self.batch_Tb * prev_vel
        # print(throttle)
        # print(batch_psi)
        next_x = batch_x + (batch_vx * jnp.cos(batch_psi) - batch_vy * jnp.sin(batch_psi)) * self.batch_DT
        # print(batch_x.shape, self.batch_DT.shape)
        next_y = batch_y + (batch_vx * jnp.sin(batch_psi) + batch_vy * jnp.cos(batch_psi)) * self.batch_DT
        next_psi = batch_psi + batch_omega * self.batch_DT
        
        # print("dbm part 1", time.time() - st)
        # st = time.time()
        # next_psi = jnp.atan2(jnp.sin(next_psi), jnp.cos(next_psi))
        alpha_f = steer - jnp.atan2(self.batch_LF * batch_omega + batch_vy, jnp.maximum(batch_vx,batch_vx*.0+.5))
        alpha_r = jnp.atan2(self.batch_LR * batch_omega - batch_vy, jnp.maximum(batch_vx,batch_vx*.0+.5))
        
        # print("dbm part 2", time.time() - st)
        # st = time.time()
        # print(alpha_f, alpha_r)
        # print(batch_vx, batch_vy)
        F_rx = throttle
        # print("Frx", F_rx, "alpha f", alpha_f, "alpha r", alpha_r, 'vx', batch_vx, 'vy', batch_vy, 'omega', batch_omega, 'steer', steer)
        F_fy = self.batch_K_FFY * alpha_f
        F_ry = self.batch_K_RFY * alpha_r
        
        next_vx = batch_vx + (F_rx - F_fy * jnp.sin(steer) + 0.* batch_vy * batch_omega * self.batch_MASS) * self.batch_DT / self.batch_MASS
        next_vy = batch_vy + (F_ry + F_fy * jnp.cos(steer) - batch_vx * batch_omega * self.batch_MASS) * self.batch_DT / self.batch_MASS
        next_omega = batch_omega + (F_fy*self.batch_LF*jnp.cos(steer) - F_ry*self.batch_LR) * self.batch_DT / self.batch_Iz
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
        
        
        prev_vel = jnp.sqrt(batch_vx**2 + batch_vy**2)
        # print("vel omega", prev_vel, batch_omega)
        
        throttle = batch_target_vel * self.params.Ta + self.params.Tb * prev_vel
        # print(throttle)
        # print(batch_psi)
        next_x = batch_x + (batch_vx * jnp.cos(batch_psi) - batch_vy * jnp.sin(batch_psi)) * self.params.DT
        next_y = batch_y + (batch_vx * jnp.sin(batch_psi) + batch_vy * jnp.cos(batch_psi)) * self.params.DT
        next_psi = batch_psi + batch_omega * self.params.DT
        # next_psi = jnp.atan2(jnp.sin(next_psi), jnp.cos(next_psi))
        alpha_f = steer - jnp.atan2(self.params.LF * batch_omega + batch_vy, jnp.maximum(batch_vx,batch_vx*.0+.5))
        alpha_r = jnp.atan2(self.params.LR * batch_omega - batch_vy, jnp.maximum(batch_vx,batch_vx*.0+.5))
        # print(alpha_f, alpha_r)
        # print(batch_vx, batch_vy)
        F_rx = throttle
        # print("Frx", F_rx, "alpha f", alpha_f, "alpha r", alpha_r, 'vx', batch_vx, 'vy', batch_vy, 'omega', batch_omega, 'steer', steer)
        F_fy = self.params.K_FFY * alpha_f
        F_ry = self.params.K_RFY * alpha_r
        
        next_vx = batch_vx + (F_rx - F_fy * jnp.sin(steer) + batch_vy * batch_omega * self.params.MASS) * self.params.DT / self.params.MASS
        next_vy = batch_vy + (F_ry + F_fy * jnp.cos(steer) - batch_vx * batch_omega * self.params.MASS) * self.params.DT / self.params.MASS
        next_omega = batch_omega + (F_fy*self.params.LF*jnp.cos(steer) - F_ry*self.params.LR) * self.params.DT / self.params.Iz
        return next_x, next_y, next_psi, next_vx, next_vy, next_omega
        
    def step_gym(self, state: CarState, action: CarAction):
        batch_x = jnp.array(state.x)
        batch_y = jnp.array(state.y)
        batch_psi = jnp.array(state.psi)
        batch_vx = jnp.array(state.vx)
        batch_vy = jnp.array(state.vy)
        batch_omega = jnp.array(state.omega)
        
        self.batch_buffer[:,1:,:] = self.batch_buffer[:,:-1,:]
        self.batch_buffer[:,0,0] = action.target_vel
        self.batch_buffer[:,0,1] = action.target_steer
        
        batch_target_vel = jnp.array(self.batch_buffer[:,-1,0])
        batch_target_steer = jnp.array(self.batch_buffer[:,-1,1])
        
        next_x, next_y, next_psi, next_vx, next_vy, next_omega = self.step(
            batch_x,
            batch_y,
            batch_psi,
            batch_vx,
            batch_vy,
            batch_omega,
            batch_target_vel,
            batch_target_steer,
            # debug=True
        )
        # print(np.array(next_x), np.array(next_y), np.array(next_psi), np.array(next_vx), np.array(next_vy), np.array(next_omega))
        return CarState(
            x=next_x[0],
            y=next_y[0],
            psi = next_psi[0],
            vx = next_vx[0],
            vy = next_vy[0],
            omega = next_omega[0],
        )



# @jax.jit
def dbm_dxdt(
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
    batch_K_RFY,
    batch_K_FFY,
    batch_Iz,
    batch_mu,
    batch_Cf,
    batch_Cr,
    batch_Bf,
    batch_Br,
    batch_hcom,
    batch_fr,
):
    steer = batch_target_steer * batch_Sa + batch_Sb 
    prev_vel = jnp.sqrt(batch_vx**2 + batch_vy**2)
    throttle = batch_target_vel * batch_Ta - batch_target_vel * batch_Tb * prev_vel
    next_x =  (batch_vx * jnp.cos(batch_psi) - batch_vy * jnp.sin(batch_psi)) 
    next_y = (batch_vx * jnp.sin(batch_psi) + batch_vy * jnp.cos(batch_psi)) 
    next_psi = batch_omega 
    if batch_vx > 0. :
        alpha_f = steer - jnp.atan((batch_LF * batch_omega + batch_vy) / jnp.maximum(batch_vx,batch_vx*.0+.5))
        alpha_r = jnp.atan((batch_LR * batch_omega - batch_vy) / jnp.maximum(batch_vx,batch_vx*.0+.5))
        F_rx = throttle - batch_fr * batch_MASS * GRAVITY * jnp.sign(batch_vx)
        
        F_fz = 0.5 * batch_MASS * GRAVITY * batch_LR / (batch_LF + batch_LR) - 0.5 * batch_hcom / (batch_LF + batch_LR) * F_rx
        F_rz = 0.5 * batch_MASS * GRAVITY * batch_LF / (batch_LF + batch_LR) + 0.5 * batch_hcom / (batch_LF + batch_LR) * F_rx 

        F_fy = 2 * batch_mu * F_fz * jnp.sin(batch_Cf * jnp.arctan(batch_Bf * alpha_f))
        F_ry = 2 * batch_mu * F_rz * jnp.sin(batch_Cr * jnp.arctan(batch_Br * alpha_r))
        next_vx =  (F_rx - F_fy * jnp.sin(steer) + batch_vy * batch_omega * batch_MASS)  / batch_MASS
        next_vy =  (F_ry + F_fy * jnp.cos(steer) - batch_vx * batch_omega * batch_MASS)  / batch_MASS
        next_omega =  (F_fy*batch_LF*jnp.cos(steer) - F_ry*batch_LR)  / batch_Iz
    
    else :
        batch_vx = -batch_vx
        batch_vy = -batch_vy
        # batch_omega = -batch_omega
        steer = -steer
        throttle = -throttle
        alpha_f = steer - jnp.atan((batch_LF * batch_omega + batch_vy) / jnp.maximum(batch_vx,batch_vx*.0+.5))
        alpha_r = jnp.atan((batch_LR * batch_omega - batch_vy) / jnp.maximum(batch_vx,batch_vx*.0+.5))
        F_rx = throttle - batch_fr * batch_MASS * GRAVITY * jnp.sign(batch_vx)
        
        F_fz = 0.5 * batch_MASS * GRAVITY * batch_LR / (batch_LF + batch_LR) - 0.5 * batch_hcom / (batch_LF + batch_LR) * F_rx
        F_rz = 0.5 * batch_MASS * GRAVITY * batch_LF / (batch_LF + batch_LR) + 0.5 * batch_hcom / (batch_LF + batch_LR) * F_rx 

        F_fy = 2 * batch_mu * F_fz * jnp.sin(batch_Cf * jnp.arctan(batch_Bf * alpha_f))
        F_ry = 2 * batch_mu * F_rz * jnp.sin(batch_Cr * jnp.arctan(batch_Br * alpha_r))
        next_vx =  (F_rx - F_fy * jnp.sin(steer) + batch_vy * batch_omega * batch_MASS)  / batch_MASS
        next_vy =  (F_ry + F_fy * jnp.cos(steer) - batch_vx * batch_omega * batch_MASS)  / batch_MASS
        next_omega =  (F_fy*batch_LF*jnp.cos(steer) - F_ry*batch_LR)  / batch_Iz
        next_vx = -next_vx
        next_vy = -next_vy    
    # F_fy = batch_K_FFY * alpha_f
    # F_ry = batch_K_RFY * alpha_r
    # print("haha ",F_rx,F_fy,batch_vy,batch_omega)
    
    # print("dbm part 3", time.time() - st)
    # print(steer, - F_fy * jnp.sin(steer), batch_vy * batch_omega * batch_MASS)
    # print("dbm step time", time.time() - st)
    # print(next_x.shape, next_y.shape, next_psi.shape, next_vx.shape, next_vy.shape, next_omega.shape)
    return next_x, next_y, next_psi, next_vx, next_vy, next_omega

# @jax.jit
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
    batch_mu,
    batch_Cf,
    batch_Cr,
    batch_Bf,
    batch_Br,
    batch_hcom,
    batch_fr, 
):
    steer = batch_target_steer * batch_Sa + batch_Sb 
        
    prev_vel = jnp.sqrt(batch_vx**2 + batch_vy**2)
    # print("vel omega", prev_vel, batch_omega)
    # print(batch_target_vel.shape, batch_Ta.shape, batch_Tb.shape, prev_vel.shape)
    throttle = batch_target_vel * batch_Ta + batch_Tb * prev_vel
    # assert jnp.all(batch_Ta > 0)
    # print(batch_target_vel)
    # print(throttle)
    # print(batch_psi)
    next_x = batch_x + (batch_vx * jnp.cos(batch_psi) - batch_vy * jnp.sin(batch_psi)) * batch_DT
    # print(batch_x.shape, batch_DT.shape)
    next_y = batch_y + (batch_vx * jnp.sin(batch_psi) + batch_vy * jnp.cos(batch_psi)) * batch_DT
    next_psi = batch_psi + batch_omega * batch_DT
    
    # alpha_f = steer - jnp.atan2(batch_LF * batch_omega + batch_vy, jnp.maximum(batch_vx,batch_vx*.0+.5))
    alpha_f = steer - jnp.atan((batch_LF * batch_omega + batch_vy) / jnp.maximum(batch_vx,batch_vx*.0+.5))
    # alpha_r = jnp.atan2(batch_LR * batch_omega - batch_vy, jnp.maximum(batch_vx,batch_vx*.0+.5))
    alpha_r = jnp.atan((batch_LR * batch_omega - batch_vy) / jnp.maximum(batch_vx,batch_vx*.0+.5))
    
    F_rx = throttle
    
    
    
    next_vx = batch_vx + (F_rx - F_fy * jnp.sin(steer) + batch_vy * batch_omega * batch_MASS) * batch_DT / batch_MASS
    next_vy = batch_vy + (F_ry + F_fy * jnp.cos(steer) - batch_vx * batch_omega * batch_MASS) * batch_DT / batch_MASS
    next_omega = batch_omega + (F_fy*batch_LF*jnp.cos(steer) - F_ry*batch_LR) * batch_DT / batch_Iz
    print(steer, - F_fy * jnp.sin(steer), batch_vy * batch_omega * batch_MASS)

    return next_x, next_y, next_psi, next_vx, next_vy, next_omega