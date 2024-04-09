"""	Nonlinear MPC using Kinematic6 and GPs for model correction.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bayes_race.params import ORCA
from bayes_race.models import Dynamic, Kinematic6
from bayes_race.gp.utils import loadGPModel, loadGPModelVars
from bayes_race.tracks import ETHZ
from bayes_race.mpc.planner import ConstantSpeed
from bayes_race.mpc.gpmpc import setupNLP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import os

#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
ERROR_CORR = True
TRACK_CONS = False
SAVE_VIDEO = True
RUN_NO = 4

#####################################################################
# default settings

GP_EPS_LEN = 305
RUN_FOLDER = 'RUN_ONLINE_' + str(RUN_NO) + '_' + str(GP_EPS_LEN) + '/'
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)
model_kin = Kinematic6(**params)

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
SIM_TIME = 26.

#####################################################################
# load GP models

with open('../gp/orca/vxgp.pickle', 'rb') as f:
	(vxmodel, vxxscaler, vxyscaler) = pickle.load(f)
vxgp = loadGPModelVars('vx', vxmodel, vxxscaler, vxyscaler)
with open('../gp/orca/vygp.pickle', 'rb') as f:
	(vymodel, vyxscaler, vyyscaler) = pickle.load(f)
vygp = loadGPModelVars('vy', vymodel, vyxscaler, vyyscaler)
# print(vxmodel.X_train_)
with open('../gp/orca/omegagp.pickle', 'rb') as f:
	(omegamodel, omegaxscaler, omegayscaler) = pickle.load(f)
omegagp = loadGPModelVars('omega', omegamodel, omegaxscaler, omegayscaler)
gpmodels = {
	'vx': vxgp,
	'vy': vygp,
	'omega': omegagp,
	'xscaler': vxxscaler,
	'yscaler': vxyscaler,
	}
x_train = np.zeros((GP_EPS_LEN,2+3+1))

#####################################################################
# load GP model params

k1 = 1.0*RBF(
		length_scale=np.ones(x_train.shape[1]),
		length_scale_bounds=(1e-5, 1e5),
		)
k2 = ConstantKernel(0.1)
vxgp_kernel = k1 + k2
vygp_kernel = k1 + k2
omegagp_kernel = k1 + k2
with open('../gp/orca/vxgp_params.pickle', 'rb') as f:
	vxmodel_params = pickle.load(f)
vxgp_kernel.set_params(**(vxmodel_params))
with open('../gp/orca/vygp_params.pickle', 'rb') as f:
	vymodel_params = pickle.load(f)
vygp_kernel.set_params(**(vymodel_params))
with open('../gp/orca/omegagp_params.pickle', 'rb') as f:
	omegamodel_params = pickle.load(f)
omegagp_kernel.set_params(**(omegamodel_params))
gpmodels_kernel = {
	'vx': vxgp_kernel,
	'vy': vygp_kernel,
	'omega': omegagp_kernel,
	}

data_gp = [[0.,0.,0.,0.,0.,0.]]
out_gp_vx = [0.]
out_gp_vy = [0.]
out_gp_omega = [0.]

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, gpmodels, track, GP_EPS_LEN=GP_EPS_LEN, 
	track_cons=TRACK_CONS, error_correction=ERROR_CORR)

#####################################################################
# define load_data

def load_data(data_dyn, data_kin, VARIDX, xscaler=None, yscaler=None):
	# data_dyn = np.load('../data/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	# data_kin = np.load('../data/KIN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	y_all = data_dyn['states'][:6,1:] - data_kin['states'][:6,1:]
	# print(N_SAMPLES)
	# print(data_kin['inputs'][:,:].T.shape)
	# print(data_kin['states'][6,:].reshape(1,-1).T.shape)
	# print(data_dyn['states'][3:6,:].T.shape)
	x = np.concatenate([
		data_kin['inputs'][:,:-1].T,
		data_kin['states'][6,:-1].reshape(1,-1).T,
		data_dyn['states'][3:6,:-1].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	if xscaler is None or yscaler is None:
		xscaler = StandardScaler()
		yscaler = StandardScaler()
		xscaler.fit(x)
		yscaler.fit(y)
		return xscaler.transform(x), yscaler.transform(y), xscaler, yscaler
	else:
		return xscaler.transform(x), yscaler.transform(y)

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states+1, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
inputs_kin = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

data_dyn = {}
data_kin = {}
data_dyn['time'] = time
data_kin['time'] = time

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:n_states,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))
states_kin = np.zeros([7,n_steps+1])
states_kin[:,0] = states[:,0]
	
# dynamic plot
H = .13
W = .07
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])


plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnFfy, = ax2.plot(0, 0, label='Ffy')
LnFrx, = ax2.plot(0, 0, label='Frx')
LnFry, = ax2.plot(0, 0, label='Fry')
LnSpeeds, = ax2.plot(0, 0, label='Speeds')
plt.xlim([0, SIM_TIME])
plt.ylim([-params['mass']*9.81, params['mass']*9.81])
plt.xlabel('time [s]')
plt.ylabel('force [N]')
plt.legend()
plt.ion()

plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnSpeeds, = ax2.plot(0, 0, label='Speeds')
LnRefSpeeds, = ax2.plot(0, 0, label='Ref Speeds')
plt.xlim([0, SIM_TIME])
plt.ylim([0, 10.])
plt.xlabel('time [s]')
plt.ylabel('force [N]')
plt.legend()

fig = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=0.8)
LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=.5, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'purple', alpha=0.8, label='Current pose')
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=.5, lw=0.5, label="prediction")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

plt.show()

if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
if not os.path.exists(RUN_FOLDER+'Video/'):
    os.makedirs(RUN_FOLDER+'Video/')
# main simulation loop
ref_speeds = []
Drs = []
Dfs = []
for idt in range(n_steps-horizon):

	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
	use_kinematic = True
	Drs.append(model.Dr)
	Dfs.append(model.Df)
	
	if idt > 310 :
		model.Df -= model.Df/2200.
		model.Dr -= model.Dr/2200.
		# load new experience into data_dyn and data_kin
		start = tm.time()	
		# data_dyn['states'] = states[:,:idt+1]
		# data_dyn['dstates'] = dstates[:,:idt+1]
		# data_dyn['inputs'] = inputs[:,:idt+1]
		min_ind = 3#max(idt-GP_EPS_LEN-1,3)
		# data_kin['states'] = states_kin[:,:idt+1]
		# data_kin['inputs'] = inputs_kin[:,:idt+1]
		data_dyn['states'] = states[:,min_ind:GP_EPS_LEN+1+min_ind]
		data_dyn['dstates'] = dstates[:,min_ind:GP_EPS_LEN+1+min_ind]
		data_dyn['inputs'] = inputs[:,min_ind:GP_EPS_LEN+1+min_ind]
		
		# data_dyn['states'][:,idt:] = states[:,idt-1:idt]
		# data_dyn['dstates'][:,idt:] = dstates[:,idt-1:idt]
		# data_dyn['inputs'][:,idt:] = inputs[:,idt-1:idt]
		
		data_kin['states'] = states_kin[:,min_ind:GP_EPS_LEN+1+min_ind]
		data_kin['inputs'] = inputs_kin[:,min_ind:GP_EPS_LEN+1+min_ind]
		
		# data_kin['states'][:,idt:] = states_kin[:,idt-1:idt]
		# data_kin['inputs'][:,idt:] = inputs_kin[:,idt-1:idt]
		
		end = tm.time()
		# print("GP init time : ", end-start)
		# start = tm.time()
		
		# Get model train data
		for VARIDX in [3,4,5] :
			if VARIDX==3 :
				x_train, y_train = load_data(data_dyn, data_kin, VARIDX, xscaler=vxxscaler, yscaler=vxyscaler)
				# idtst = idt
				# while idtst-3-2*min_ind+idt <= 306 :
				# 	x_train[idtst-min_ind-1:idtst-3-2*min_ind+idt,:] = x_train[:idt-min_ind-2,:]
				# 	y_train[idtst-min_ind-1:idtst-3-2*min_ind+idt,:] = y_train[:idt-min_ind-2,:]
				# 	idtst += idt-min_ind-2
				start = tm.time()
				vxmodel = GaussianProcessRegressor(
					alpha=1e-6, 
					kernel=vxgp_kernel, 
					normalize_y=True,
					optimizer=None,
					# n_restarts_optimizer=0,
				)
				vxmodel.fit(x_train, y_train)
				end = tm.time()
				# print("GP vx fit time : ", end-start)
			if VARIDX==4 :
				x_train, y_train = load_data(data_dyn, data_kin, VARIDX, xscaler=vyxscaler, yscaler=vyyscaler)
				# idtst = idt
				# while idtst-3-2*min_ind+idt <= 306 :
				# 	x_train[idtst-min_ind-1:idtst-3-2*min_ind+idt,:] = x_train[:idt-min_ind-2,:]
				# 	y_train[idtst-min_ind-1:idtst-3-2*min_ind+idt,:] = y_train[:idt-min_ind-2,:]
				# 	idtst += idt-min_ind-2
				start = tm.time()
				vymodel = GaussianProcessRegressor(
					alpha=1e-6, 
					kernel=vygp_kernel, 
					normalize_y=True,
					optimizer=None,
					# n_restarts_optimizer=0,
				)
				vymodel.fit(x_train, y_train)
				end = tm.time()
				# print("GP vy fit time : ", end-start)
			if VARIDX==5 :
				x_train, y_train = load_data(data_dyn, data_kin, VARIDX, xscaler=omegaxscaler, yscaler=omegayscaler)
				# idtst = idt
				# while idtst-3-2*min_ind+idt <= 306 :
				# 	x_train[idtst-min_ind-1:idtst-3-2*min_ind+idt,:] = x_train[:idt-min_ind-2,:]
				# 	y_train[idtst-min_ind-1:idtst-3-2*min_ind+idt,:] = y_train[:idt-min_ind-2,:]
				# 	idtst += idt-min_ind-2
				start = tm.time()
				omegamodel = GaussianProcessRegressor(
					alpha=1e-6, 
					kernel=omegagp_kernel, 
					normalize_y=True,
					optimizer=None,
					# n_restarts_optimizer=0,
				)
				omegamodel.fit(x_train, y_train)
				end = tm.time()
				# print("GP omega fit time : ", end-start)
		# end = tm.time()
		# print("GP formation time : ", end-start)
		# start = tm.time()
		# vxgp = loadGPModel('vx', vxmodel, vxxscaler, vxyscaler)
		# vygp = loadGPModel('vy', vymodel, vyxscaler, vyyscaler)
		# omegagp = loadGPModel('omega', omegamodel, omegaxscaler, omegayscaler)
		# gpmodels = {
		# 	'vx': vxgp,
		# 	'vy': vygp,
		# 	'omega': omegagp,
		# 	'xscaler': vxxscaler,
		# 	'yscaler': vxyscaler,
		# }
		# end = tm.time()
		# print("GP load time : ", end-start)
		start = tm.time()
		# nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, gpmodels, track, 
		# 	track_cons=TRACK_CONS, error_correction=ERROR_CORR)
		# nlp.update_gp_models(gpmodels)
		end = tm.time()
		use_kinematic = False
		# print("NLP setup time : ", end-start)
	# else :

	# planner based on BayesOpt
	xref, projidx = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)
	ref_speeds.append(track.v_raceline[projidx])
	if projidx > 656 :
		projidx = 0
	# print(projidx)
	# solve NLP
	start = tm.time()	
	umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev, vxm=vxmodel, vym=vymodel, omegam=omegamodel, use_kinematic=use_kinematic)
	end = tm.time()
	inputs[:,idt] = np.array([umpc[0,0], states[n_states,idt] + Ts*umpc[1,0]])
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))

	# update current position with numerical integration (exact model)
	x_next, dxdt_next = model.sim_continuous(states[:n_states,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
	inputs_kin[:,idt] = inputs[:,idt]
	if (idt!=0) :
		inputs_kin[1,idt] = (inputs[1,idt] - inputs[1,idt-1])/Ts
	else :
		inputs_kin[1,idt] = (inputs[1,idt] - 0.)/Ts
	x_next_kin, dxdt_next_kin = model_kin.sim_continuous(states[:,idt], inputs_kin[:,idt].reshape(-1,1), [0, Ts])
	states_kin[:,idt+1] = x_next_kin[:,-1]
	
	states[:n_states,idt+1] = x_next[:,-1]
	states[n_states,idt+1] = inputs[1,idt]
	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])

	# forward sim to predict over the horizon
	steer = states[n_states,idt]
	hstates[:,0] = x0[:n_states]
	hstates2[:,0] = x0[:n_states]
	for idh in range(horizon):
		steer = steer + Ts*umpc[1,idh]
		hinput = np.array([umpc[0,idh], steer])
		x_next, dxdt_next = model.sim_continuous(hstates[:n_states,idh], hinput.reshape(-1,1), [0, Ts])
		hstates[:,idh+1] = x_next[:n_states,-1]
		hstates2[:,idh+1] = xmpc[:n_states,idh+1]

	# update plot
	LnS.set_xdata(states[0,:idt+1])
	LnS.set_ydata(states[1,:idt+1])
	if SAVE_VIDEO :
		plt.savefig(RUN_FOLDER+'Video/frame'+str(idt)+'.png', dpi=300)
	
	LnR.set_xdata(xref[0,1:])
	LnR.set_ydata(xref[1,1:])

	LnP.set_xdata(states[0,idt] + dims[:,0]*np.cos(states[2,idt]) - dims[:,1]*np.sin(states[2,idt]))
	LnP.set_ydata(states[1,idt] + dims[:,0]*np.sin(states[2,idt]) + dims[:,1]*np.cos(states[2,idt]))
	
	LnH.set_xdata(hstates[0])
	LnH.set_ydata(hstates[1])

	LnH2.set_xdata(hstates2[0])
	LnH2.set_ydata(hstates2[1])
	
	LnFfy.set_xdata(time[:idt+1])
	LnFfy.set_ydata(Ffy[:idt+1])

	LnFrx.set_xdata(time[:idt+1])
	LnFrx.set_ydata(Frx[:idt+1])

	LnFry.set_xdata(time[:idt+1])
	LnFry.set_ydata(Fry[:idt+1])

	LnSpeeds.set_xdata(time[:idt+1])
	LnSpeeds.set_ydata(states[3,:idt+1])

	LnRefSpeeds.set_xdata(time[:idt+1])
	LnRefSpeeds.set_ydata(ref_speeds)
	plt.pause(Ts/1000)

plt.ioff()

#####################################################################
# save data

if SAVE_RESULTS:
	np.savez(
		'../data/DYN-GPMPC-{}{}.npz'.format(SUFFIX, TRACK_NAME),
		time=time,
		states=states,
		dstates=dstates,
		inputs=inputs,
		)

#####################################################################
# plots

# plot speed
plt.figure()
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon], label='abs')
plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
plt.plot(time[:n_steps-horizon], ref_speeds, label='ref_speeds')
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.grid(True)
plt.legend()
plt.savefig(RUN_FOLDER+'speeds.png')

# plot acceleration
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('PWM duty cycle [-]')
plt.grid(True)
plt.savefig(RUN_FOLDER+'pwms.png')

# plot steering angle
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('steering [rad]')
plt.grid(True)
plt.savefig(RUN_FOLDER+'steerings.png')

# plot inertial heading
plt.figure()
plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('orientation [rad]')
plt.grid(True)
plt.savefig(RUN_FOLDER+'orientations.png')

# plot speed
plt.figure()
plt.plot(time[:n_steps-horizon], Dfs[:n_steps-horizon], label='Df')
plt.plot(time[:n_steps-horizon], Drs[:n_steps-horizon], label='Dr')
plt.xlabel('time [s]')
plt.ylabel('mu*N [N]')
plt.grid(True)
plt.legend()
plt.savefig(RUN_FOLDER+'max_friction_forces.png')

plt.show()