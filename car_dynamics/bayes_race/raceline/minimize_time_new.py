""" Calculate minimum lap time and optimal speed profile on a fixed path.
    Friction circle model is used for vehicle dynamics.

    Implementation of ``Minimum-time speed optimisation over a fixed path`` by Lipp and Boyd (2014)
    https://web.stanford.edu/~boyd/papers/pdf/speed_opt.pdf
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import time
import numpy as np
import cvxpy as cv
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt

from bayes_race.utils import odeintRK6
from bayes_race.params import ORCA, F110, CarlaParams
from bayes_race.tracks import ETHZ, ETHZMobil, UCB, CarlaRace
import math


def define_path(x, y, curvs, plot_results=True):
    """ calculate derivatives and double derivatives in path coordinate system
    """
    num_wpts = np.size(x)
    
    path = {
            'x': x, 
            'y': y, 
            'curvs': curvs
            }
    return path

def plots(t, vxys):
    # speed
    flg, ax = plt.subplots(1)
    
    i = 0
    for vxy in vxys :
        plt.plot(vxy, label='speed abs at mu='+str(mus[i]))
        i += 1
    plt.plot(v_raceline, label='gt speeds')
    plt.title('absolute speed vs time')
    plt.xlabel('time [s]')
    plt.ylabel('speed [m/s]')
    plt.grid(True)
    plt.legend(loc=0)

    plt.show()

def friction_circle(Fmax):
    t = np.linspace(0, 2*np.pi, num=100)
    x = Fmax*np.cos(t)
    y = Fmax*np.sin(t)
    return x, y

def get_time_vec(b, theta):
    num_wpts = theta.size
    dtheta = 1/(num_wpts-1)
    bsqrt = np.sqrt(b)
    dt = 2*dtheta/(bsqrt[0:num_wpts-1]+bsqrt[1:num_wpts])
    t = np.zeros([num_wpts])
    for j in range(1, num_wpts):
        t[j] = t[j-1] + dt[j-1]
    return t

def get_dist(x1,x2,y1,y2) :
    return ((x1-x2)**2 + (y1-y2)**2)**(1/2)
    
def optimize(path, params,mu=None):
    """ main function to solve convex optimization
    """
    x = path['x']
    y = path['y']
    curv = path['curvs']
    num_wpts = x.size
    # print(curv)
    # A = cv.Variable((num_wpts-1))
    V = cv.Variable((num_wpts))
    # U = cv.Variable((2, num_wpts-1))

    cost = 0
    constr = []

    # no constr on A[0], U[:,0], defined on mid points
    # B[0].value = 0.1
    if mu is None :
        print('mu = ', (params['Dr'] + params['Df'])/(9.81*params['mass']))
        ay_max = (params['Dr'] + params['Df'])/params['mass']
    else :
        ay_max = mu*9.81
        print('mu = ', mu)

    ay_min = -(params['Dr'] + params['Df'])/params['mass']
    # constr += [V[0] == V[-1]]
    constr += [V[0] == 0]
    for j in range(num_wpts-1):
        ds = get_dist(x[j],x[j+1],y[j],y[j+1])
        # print(ds)
        cost += 2*ds*cv.inv_pos(cv.power(V[j],0.5) + cv.power(V[j+1],0.5))
        a_max = ((params['Cm1']-params['Cm2']*V[j])*params['max_inputs'][0] - params['Cr0'] - params['Cr2']*V[j]**2)/params['mass']
        a_min = ((params['Cm1']-params['Cm2']*V[j])*params['min_inputs'][0] - params['Cr0'])/params['mass']
        
        constr += [V[j] >= 0.]
        constr += [V[j+1] - V[j] <= 2*a_max*ds]
        # constr += [V[j+1] - V[j] <= params['max_acc']*ds]
        constr += [V[j+1] - V[j] >= 2*a_min*ds]
        constr += [V[j]**2*curv[j] <= ay_max]
        # constr += [V[j]**2*curv[j] >= ay_min]
        
    problem = cv.Problem(cv.Minimize(cost), constr)
    solution = problem.solve(solver=cv.ECOS, gp=False)
    V = V.value
    V = abs(V)
    topt = cost.value
    vopt = (V)
    return vopt, topt

def solve(x, y, curvs, params, mu=None, plot_results=False, print_updates=False, **kwargs):
    """ call this wrapper function
    """
    path = define_path(x, y, curvs)
    # params = define_params(mass, lf, lr)
    vopt, topt = optimize(
        path=path, 
        params=params,
        mu=mu
        )
    return vopt, topt

def calcMinimumTimeSpeed(x, y, curvs, params, mu = None, **kwargs):
    """ wrapper function to return minimum time only
    """
    vopt, topt= solve(x, y, curvs, params, mu=mu, **kwargs)
    return vopt, topt

def circle_curvature(x1, y1, x2, y2, x3, y3):
    a = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    b = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    c = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    k = area / s
    c = (4 * area) / (a*b*c)
    return c

def render_curvature(x,y, gap=3) :
    # To get the curvature at each point on the trajectory
    N = len(x)
    curvs = []
    for i in range(N) :
        x1 = x[(i-gap)%N]
        y1 = y[(i-gap)%N]
        x2 = x[(i)%N]
        y2 = y[(i)%N]
        x3 = x[(i+gap)%N]
        y3 = y[(i+gap)%N]
        curvs.append(circle_curvature(x1, y1, x2, y2, x3, y3))
    return np.array(curvs)

if __name__ == "__main__":
    """ example how to use
    """
    global v_raceline, mus
    # define waypoints
    # for example we choose center line but can be any trajectory
    track = CarlaRace(reference='optimal')
    x, y = track.x_raceline, track.y_raceline
    v_raceline = track.v_raceline
    curvs = render_curvature(x,y)
    # define vehicle params
    # params = ORCA()
    params = CarlaParams()
    file_name = '../tracks/src/carla{}_raceline{}.npz'.format('', '')
    npzfile = np.load(file_name)
    print(npzfile.files)
    # call the solver
    start = time.time()
    vopts = []
    mus = np.array([0.5,0.6,0.7,0.8,0.9,1.0])
    for mu in mus :
        # vopt, topt = calcMinimumTimeSpeed(x, y, curvs, params, mu=mu, plot_results=True)
        vopt = track.v_raceline*math.sqrt((mu*params['mass']*9.81)/(params['Df']+params['Dr']))
        vopts.append(vopt)
    vopt_mu = {}
    vopt_mu['0.5'] = vopts[0]
    vopt_mu['0.6'] = vopts[1]
    vopt_mu['0.7'] = vopts[2]
    vopt_mu['0.8'] = vopts[3]
    vopt_mu['0.9'] = vopts[4]
    vopt_mu['1.0'] = vopts[5]
    plots(None,vopts)
    vopts = np.array(vopts)
    end = time.time()
    
    np.savez(file_name,x=npzfile['x'],y=npzfile['y'],time=npzfile['time']\
        ,speed=npzfile['speed'], inputs=npzfile['inputs'], speeds=vopts, mus=mus)
    print("time to solve optimization: {:.2f}".format(end-start))

