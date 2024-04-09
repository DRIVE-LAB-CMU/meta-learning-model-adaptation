import scipy
import scipy.interpolate
import scipy.signal
import numpy as np
import math

def rotate_point(x, y, angle):
    """
    Rotate a point around the origin by a given angle.

    Parameters:
    x (float): The x-coordinate of the point.
    y (float): The y-coordinate of the point.
    angle (float): The rotation angle in degrees.

    Returns:
    (float, float): The new coordinates after rotation.
    """
    # Convert angle from degrees to radians
    theta = math.radians(angle)

    # Apply rotation matrix
    x_new = x * math.cos(theta) - y * math.sin(theta)
    y_new = x * math.sin(theta) + y * math.cos(theta)

    return x_new, y_new

def pos2vel_savgol(p, delta=0.05, window_length=9, polyorder=2):
    """ Given position array (num_samples, 2) denoting (x,y), return velocity (vx, vy)

    Args:
        p (np.ndarray): position array (num_samples, 2) denoting (x,y)
        delta (float, optional): sampling time. Defaults to 0.05 (1 div control frequency).
        window_length (int, optional): window length for savgol filter. Defaults to 9.
        polyorder (int, optional): polynomial order for savgol filter. Defaults to 2.
    Returns:
        v (np.ndarray): velocity array (num_samples, 2) denoting (vx, vy)
    """
    assert len(p.shape) > 1
    assert p.shape[1] == 2
    v_smooth = np.zeros_like(p)
    p_smooth = np.zeros_like(p)
    a_smooth = np.zeros_like(p)
    for i in range(p.shape[1]):
        p_smooth[:,i] = scipy.signal.savgol_filter(p[:,i], window_length=window_length, polyorder=polyorder, \
                                                    deriv=0, delta=delta)
        v_smooth[:,i] = scipy.signal.savgol_filter(p[:,i], window_length=window_length, polyorder=polyorder, \
                                                    deriv=1, delta=delta)
        a_smooth[:,i] = scipy.signal.savgol_filter(p[:,i], window_length=window_length, polyorder=polyorder, \
                                                    deriv=2, delta=delta)
    
    # kappa = np.abs(v_smooth[:,0]*a_smooth[:,1]-v_smooth[:,1]*a_smooth[:,0]) /\
    #                     (v_smooth[:,0]**2. + v_smooth[:,0]**2.) ** 1.5

    v_smooth_vec = v_smooth[:,:2]
    v_smooth_val = np.linalg.norm(v_smooth_vec, axis=1)
    # return p_smooth, v_smooth, kappa
    return p_smooth, v_smooth_val, v_smooth_vec

def calc_delta_v(v_list, yaw_list, LF, LR):
    assert len(v_list.shape) == 2
    # assert v_list.shape == yaw_list.shape
    
    delta_list = []
    beta_list = []
    
    for i, (v, yaw) in enumerate(zip(v_list, yaw_list)):
        beta = np.arctan2(v[1], v[0]) - yaw
        beta = np.arctan2(np.sin(beta), np.cos(beta))
        beta_list.append(beta)
        delta_list.append(np.arctan2(np.tan(beta)*(LF+LR),LR))
    
    return np.array(delta_list), np.array(beta_list)
        
def calc_delta(p, yaw, LF, LR):
    """ Given position array (num_samples, 2) denoting (x,y), return steering angle delta
    """
    assert len(p.shape) == 2
    assert p.shape[0] == yaw.shape[0]
    delta = []
    alpha_list = []
    for i, _ in enumerate(p):
        if i == 0:
            continue
        # ld = np.linalg.norm(state[i,:2] - state[i-1,:2])
        ld = np.linalg.norm(p[i] - p[i - 1])
        
        # R = 1 / kappa[i-1]
        
        # car_dir = np.array([np.cos(state[i-1,2]),np.sin(state[i-1,2])])
        car_dir = np.array([np.cos(yaw[i - 1]),np.sin(yaw[i - 1])])
        vel_dir = np.array(p[i] - p[i - 1])
        alpha = np.math.atan2(np.linalg.det([car_dir, vel_dir]),np.dot(car_dir,vel_dir))
        # alpha = np.math.atan2(vel_dir[1], vel_dir[0]) - yaw[i - 1]
        
        
        alpha_list.append(alpha)
        delta.append(np.arctan2(np.tan(alpha)*(LF+LR),LR))

    return np.array(delta), np.array(alpha_list)

def interpolate(t, x, frequency=20, mode='linear', t_new=None):
    assert len(x.shape) > 1
    # t: scalar
    if t_new is None:
        l = int(np.floor((t[-1]-t[0])*frequency)) + 1
        t_new = np.linspace(t[0], t[0] + (l-1)/frequency, l) 
    else:
        l = len(t_new)
    
    x_new = np.zeros((l, x.shape[1]))
    for i in range(x.shape[1]):
        f = scipy.interpolate.interp1d(t, x[:,i], kind=mode, bounds_error=False, fill_value='extrapolate')
        try:
            x_new[:,i] = f(t_new)
        except:
            print(t.shape, x[:,i].shape, t_new.shape, x_new.shape)
    return t_new, x_new


def quaternion_to_euler(input):
    """ quat: [x,y,z,w] -> [roll, pitch, yaw]
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x-axis, pitch is rotation around y-axis
    and yaw is rotation around z-axis.
    """
    # Normalize the quaternion
    x, y, z, w = input
    magnitude = math.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x / magnitude, y / magnitude, z / magnitude, w / magnitude

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # Use 90 degrees if out of range
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw