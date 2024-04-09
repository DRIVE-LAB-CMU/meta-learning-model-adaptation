import jax
import jax.numpy as jnp
import numpy as np

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
    
    def __init__(self, waypoint_type: str, dt: float, H: int, speed: float):
        self.waypoint_type = waypoint_type
        
        if waypoint_type == 'counter circle':
            self.fn = counter_circle
        elif waypoint_type == 'counter oval':
            self.fn = counter_oval
        elif waypoint_type == 'counter square':
            self.fn = counter_square
        else:
            raise ValueError(f"Unknown waypoint_type: {waypoint_type}")
        self.dt = dt 
        self.H = H
        self.speed = speed
        
        self.waypoint_t_list = jnp.arange(0, jnp.pi*2+dt, dt / speed)
        self.waypoint_list = jnp.array([self.fn(t) for t in self.waypoint_t_list])
        self.waypoint_list_np = np.array(self.waypoint_list)
        
    
    def generate(self, obs: jnp.ndarray) -> jnp.ndarray:
        pos2d = obs[:2]
        psi = obs[2]
        vel2d = obs[3:5]
        distance_list = jnp.linalg.norm(self.waypoint_list - pos2d, axis=-1)
        t_idx = jnp.argmin(distance_list)
        t_closed = self.waypoint_t_list[t_idx]
        target_pos_list = []
        
        speed = jnp.clip(jnp.linalg.norm(vel2d), .5, 100.)
        # speed = self.speed
        magic_factor = 1./1.2
        for i in range(self.H+1):
            t = t_closed + i * self.dt * speed * magic_factor
            pos = self.fn(t)
            pos_next = self.fn(t + self.dt/2)
            psi = jnp.arctan2(pos_next[1] - pos[1], pos_next[0] - pos[0])
            target_pos_list.append(jnp.array([pos[0], pos[1], psi]))
        return jnp.array(target_pos_list)        
        
        
    