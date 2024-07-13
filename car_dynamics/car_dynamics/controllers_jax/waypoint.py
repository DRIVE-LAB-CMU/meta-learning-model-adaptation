import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
import time 

def counter_circle(theta):
    center = (0., 0.)
    radius = 4.5
    return jnp.array([center[0] + radius * jnp.cos(theta),
                      center[1] + radius * jnp.sin(theta)])
    
def counter_oval(theta):
    center = (0., 0.)
    x_radius = 1.2
    y_radius = 1.4

    x = center[0] + x_radius * jnp.cos(theta)
    y = center[1] + y_radius * jnp.sin(theta)

    return jnp.array([x, y])

def custom_fn(theta, traj):
    while theta >= traj[-1, 0]:
        theta -= traj[-1, 0]
    # i = 0
    # j = len(traj) - 1
    # k = (i + j) // 2
    # while i < j:
    #     if theta < traj[k, 0]:
    #         j = k
    #     else:
    #         i = k + 1
    #     k = (i + j) // 2
    # i -= 1
    i = np.sum(traj[:,0]<theta) - 1
    # print("haha")
    ratio = (theta - traj[i, 0]) / (traj[i + 1, 0] - traj[i, 0])
    x, y = traj[i, 1] + ratio * (traj[i + 1, 1] - traj[i, 1]), traj[i, 2] + ratio * (traj[i + 1, 2] - traj[i, 2])
    v = traj[i, 3] + ratio * (traj[i + 1, 3] - traj[i, 3])
    return jnp.array([x, y]), 0.9*v
    
    
def counter_square(theta):
    theta = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))
    ## Generate a square
    center = (1., 2.0)
    x_radius = 4.
    y_radius = 4.
    r = jnp.sqrt(x_radius**2 + y_radius**2)
    
    if -np.pi/4 <= theta <= jnp.pi/4:
        x = center[0] + x_radius
        y = center[1] + r * jnp.sin(theta)
    elif -jnp.pi/4*3 <= theta <= -jnp.pi/4:
        x = center[0] + r * jnp.cos(theta)
        y = center[1] - y_radius
    elif -jnp.pi <= theta <= -jnp.pi/4*3 or jnp.pi/4*3 <= theta <= jnp.pi:
        x = center[0] - x_radius
        y = center[1] + r * jnp.sin(theta)
    else:
        x = center[0] + r * jnp.cos(theta)
        y = center[1] + y_radius

    return jnp.array([x, y])
        
class WaypointGenerator:
    
    def line_point_distance(self, px, py, x1, y1, x2, y2):
        """
        This function calculates the distance between a point and a line segment, also determining side.

        Args:
            px: x coordinate of the point.
            py: y coordinate of the point.
            x1: x coordinate of the first point on the line segment.
            y1: y coordinate of the first point on the line segment.
            x2: x coordinate of the second point on the line segment.
            y2: y coordinate of the second point on the line segment.

        Returns:
            A tuple containing the distance and a value indicating side (positive for left, negative for right).
        """

        # Calculate the vector along the line segment
        dx = x2 - x1
        dy = y2 - y1

        # Calculate the vector from the point to the first point on the line segment
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

        return t

    def __init__(self, waypoint_type: str, dt: float, H: int, speed: float):
        self.waypoint_type = waypoint_type
        
        if waypoint_type == 'counter circle':
            self.fn = counter_circle
        elif waypoint_type == 'counter oval':
            self.fn = counter_oval
        elif waypoint_type == 'counter square':
            self.fn = counter_square
        elif waypoint_type.endswith('.yaml'):
            yaml_content = yaml.load(open(waypoint_type, 'r'), Loader=yaml.FullLoader)
            centerline_file = yaml_content['track_info']['centerline_file'][:-4]
            ox = yaml_content['track_info']['ox']
            oy = yaml_content['track_info']['oy']
            self.fn = custom_fn
            df = pd.read_csv('/home/dvij/lecar-car/ref_trajs/' + centerline_file + '_with_speeds.csv')
            self.path = np.array(df.iloc[:,:]) + np.array([0, ox, oy, 0])
            # print(self.path)
            self.waypoint_type = 'custom'
            self.scale = 6.5*(float(yaml_content['vehicle_params']['Lf']) + float(yaml_content['vehicle_params']['Lr']))
        else :
            self.fn = custom_fn
            self.scale = 1.
            df = pd.read_csv('/home/dvij/lecar-car/ref_trajs/' + self.waypoint_type + '_with_speeds.csv')
            self.path = np.array(df.iloc[:,:])
            self.waypoint_type = 'custom'
        # else:
        #     raise ValueError(f"Unknown waypoint_type: {waypoint_type}")
        self.dt = dt 
        self.H = H
        self.speed = speed
        if self.waypoint_type == 'custom':
            self.waypoint_t_list = jnp.arange(0, self.path[-1,0], 0.1*self.scale)
            self.waypoint_list = jnp.array([self.fn(t,self.path)[0] for t in self.waypoint_t_list])
        else :
            self.waypoint_t_list = jnp.arange(0, jnp.pi*2+dt, dt / speed)
            self.waypoint_list = jnp.array([self.fn(t) for t in self.waypoint_t_list])
        self.waypoint_list_np = np.array(self.waypoint_list)
        self.last_i = -1
        
    
    def generate(self, obs: jnp.ndarray, dt=-1, mu_factor=1.) -> jnp.ndarray:
        if dt < 0.:
            dt = self.dt
        # print(len(self.waypoint_list))
        pos2d = obs[:2]
        psi = obs[2]
        vel2d = obs[3:5]
        ts = time.time()
        if self.waypoint_type != 'custom':
            distance_list = jnp.linalg.norm(self.waypoint_list - pos2d, axis=-1)
            t_idx = jnp.argmin(distance_list)
            t_closed = self.waypoint_t_list[t_idx]
            target_pos_list = []
            
            magic_factor = 1./1.2
            for i in range(self.H+1):
                t = t_closed + i * dt * self.speed * magic_factor
                t_1 = t + dt * self.speed * magic_factor
                pos = self.fn(t)
                pos_next = self.fn(t_1)
                vel = (pos_next - pos) / dt
                speed_ref = jnp.clip(jnp.linalg.norm(vel), .5, 100.)
                # print(speed_ref)
                psi = jnp.arctan2(pos_next[1] - pos[1], pos_next[0] - pos[0])
                target_pos_list.append(jnp.array([pos[0], pos[1], psi, speed_ref]))
            return jnp.array(target_pos_list), None
        else :
            if self.last_i == -1:
                # print("WTF!!!!!!", pos2d, self.waypoint_list)
                distance_list = jnp.linalg.norm(self.waypoint_list - pos2d, axis=-1)
                t_idx = jnp.argmin(distance_list)
                self.last_i = t_idx
            else :
                if self.last_i + 20 < len(self.waypoint_list):
                    distance_list = jnp.linalg.norm(self.waypoint_list[self.last_i:(self.last_i+20)] - pos2d, axis=-1)
                    t_idx = jnp.argmin(distance_list) + self.last_i          
                else :
                    distance_list = jnp.linalg.norm(self.waypoint_list[self.last_i:] - pos2d, axis=-1)
                    t_idx1 = jnp.argmin(distance_list) + self.last_i
                    distance_list2 = jnp.linalg.norm(self.waypoint_list[:20] - pos2d, axis=-1)
                    t_idx2 = jnp.argmin(distance_list2)
                    d1 = distance_list[t_idx1]
                    d2 = distance_list2[t_idx2]
                    if d1 < d2:
                        t_idx = t_idx1
                    else :
                        t_idx = t_idx2 
                self.last_i = t_idx
            # print("init time: ", ts-time.time())
            t_closed = self.waypoint_t_list[t_idx]
            
            d1 = np.sqrt((self.waypoint_list[t_idx-1,0]-self.waypoint_list[t_idx,0])**2 \
                + (self.waypoint_list[t_idx-1,1]-self.waypoint_list[t_idx,1])**2)
            if d1 > 0. : 
                t1 = self.line_point_distance(pos2d[0], pos2d[1], self.waypoint_list[t_idx-1,0], self.waypoint_list[t_idx-1,1], self.waypoint_list[t_idx,0], self.waypoint_list[t_idx,1])
                d1 *= min(1.,max(0.,t1)) - 1.
            N = len(self.waypoint_list)
            d2 = np.sqrt((self.waypoint_list[t_idx,0]-self.waypoint_list[(t_idx+1)%N,0])**2 \
                + (self.waypoint_list[t_idx,1]-self.waypoint_list[(t_idx+1)%N,1])**2)
            if d2 > 0. :
                t2 = self.line_point_distance(pos2d[0], pos2d[1], self.waypoint_list[t_idx,0], self.waypoint_list[t_idx,1], self.waypoint_list[(t_idx+1)%N,0], self.waypoint_list[(t_idx+1)%N,1])
                d2 *= min(1.,max(0.,t2))
            t_closed_refined = t_closed + d1 + d2
            # print("refine time: ", ts-time.time())
            target_pos_list = []
            _, speed = self.fn(t_closed_refined, self.path)
            speed *= np.sqrt(mu_factor)
            kin_pos, _ = self.fn(t_closed_refined+1.2*speed, self.path)
            for i in range(self.H+1):
                # print(speed)
                t = t_closed_refined + i * dt * speed
                t_1 = t + dt * speed
                pos, speed = self.fn(t, self.path)
                speed *= np.sqrt(mu_factor)
                pos_next, _ = self.fn(t_1, self.path)
                vel = (pos_next - pos) / dt
                speed_ref = jnp.clip(jnp.linalg.norm(vel), .5, 100.)
                # speed_ref = jnp.clip(jnp.linalg.norm(vel), .5, 100.)
                psi = jnp.arctan2(pos_next[1] - pos[1], pos_next[0] - pos[0])
                # print(speed, speed_ref)
                target_pos_list.append(jnp.array([pos[0], pos[1], psi, speed]))
            # print("end time: ", time.time()-ts)
            return jnp.array(target_pos_list), kin_pos
        
        
    