"""	Setup NLP with GP models in CasADi.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import casadi as cs

from bayes_race.models import Kinematic6
from bayes_race.mpc.constraints import Boundary
import time as tm


class setupNLP:

	def __init__(self, horizon, Ts, Q, P, R, params, gpmodels, track, GP_EPS_LEN = 305, 
		track_cons=False, error_correction=True, input_acc=False):

		model = Kinematic6(input_acc=input_acc, **params)

		self.horizon = horizon
		self.params = params
		self.model = model
		self.track = track
		self.track_cons = track_cons
		self.error_correction = error_correction
		self.n_states = model.n_states
		self.Ts = Ts
		n_inputs = model.n_inputs
		xref_size = 2

		vxgp = gpmodels['vx']
		vygp = gpmodels['vy']
		omegagp = gpmodels['omega']
		xss = gpmodels['xscaler'].scale_
		xsm = gpmodels['xscaler'].mean_

		# Casadi vaiables
		x0 = cs.SX.sym('x0', self.n_states, 1)
		xref = cs.SX.sym('xref', xref_size, horizon+1)
		X = cs.SX.sym('X', GP_EPS_LEN, 6)
		alpha_vx = cs.SX.sym('alpha_vx', GP_EPS_LEN, 1)
		y_mean_vx = cs.SX.sym('y_mean_vx', 1, 1)
		alpha_vy = cs.SX.sym('alpha_vy', GP_EPS_LEN, 1)
		y_mean_vy = cs.SX.sym('y_mean_vy', 1, 1)
		alpha_omega = cs.SX.sym('alpha_omega', GP_EPS_LEN, 1)
		y_mean_omega = cs.SX.sym('y_mean_omega', 1, 1)
		uprev = cs.SX.sym('uprev', 2, 1)
		kin_use = cs.SX.sym('kin_use', 1, 1)
		x = cs.SX.sym('x', self.n_states, horizon+1)
		u = cs.SX.sym('u', n_inputs, horizon)
		self.dxdtc = cs.SX.sym('dxdt', self.n_states, 1)

		if track_cons:
			eps = cs.SX.sym('eps', 2, horizon)
			Aineq = cs.SX.sym('Aineq', 2*horizon, 2)
			bineq = cs.SX.sym('bineq', 2*horizon, 1)

		# sum problem objectives and concatenate constraints
		cost_tracking = 0
		cost_actuation = 0
		cost_violation = 0

		cost_tracking += (x[:xref_size,-1]-xref[:xref_size,-1]).T @ P @ (x[:xref_size,-1]-xref[:xref_size,-1])
		constraints = x[:,0] - x0
		# start = tm.time()
		# vxgp = vxgp.expand()
		# vygp = vygp.expand()
		# omegagp = omegagp.expand()
		idh = 0
		gpinput = (cs.vertcat(u[:,idh], x[6,idh], x[3:6,idh]).T - xsm.reshape(1,-1)) / xss.reshape(1,-1)
		# print(vxgp.print_options())
		start = tm.time()

		for idh in range(horizon):
			dxdt = model.casadi(x[:,idh], u[:,idh], self.dxdtc)

			if error_correction:
				gpinput = (cs.vertcat(u[:,idh], x[6,idh], x[3:6,idh]).T - xsm.reshape(1,-1)) / xss.reshape(1,-1)
				# a.reset_input()
				# b.reset_input()
				# c.reset_input()
				# print(cs.SX.get_free(vxgp))
				error = cs.vertcat(
					cs.SX(0),
					cs.SX(0),
					cs.SX(0),
					vxgp(gpinput,X,alpha_vx,y_mean_vx)[0],
					vygp(gpinput,X,alpha_vy,y_mean_vy)[0],
				 	omegagp(gpinput,X,alpha_omega,y_mean_omega)[0],
					cs.SX(0),
					)
				# print(a.serialize())
				# print()
				constraints = cs.vertcat( constraints, x[:,idh+1] - x[:,idh] - Ts*dxdt - kin_use*error )
				print(constraints.size(), tm.time()-start)
			else:
				constraints = cs.vertcat( constraints, x[:,idh+1] - x[:,idh] - Ts*dxdt )

		for idh in range(horizon):

			# delta between subsequent time steps
			if idh==0:
				deltaU  = u[0,idh]-uprev[0]
			else:
				deltaU = u[0,idh]-u[0,idh-1]

			cost_tracking += (x[:xref_size,idh+1]-xref[:xref_size,idh+1]).T @ Q @ (x[:xref_size,idh+1]-xref[:xref_size,idh+1])
			cost_actuation += deltaU.T @ R[0,0] @ deltaU + (u[1,idh]*Ts) @ R[1,1] @ (u[1,idh]*Ts)

			if track_cons:
				cost_violation += 1e6 * (eps[:,idh].T @ eps[:,idh])

			constraints = cs.vertcat( constraints, x[6,idh+1] - params['max_inputs'][1] )
			constraints = cs.vertcat( constraints, -x[6,idh+1] + params['min_inputs'][1] )
			constraints = cs.vertcat( constraints, u[0,idh] - params['max_inputs'][0] )
			constraints = cs.vertcat( constraints, -u[0,idh] + params['min_inputs'][0] )
			constraints = cs.vertcat( constraints, u[1,idh] - params['max_rates'][1] )
			constraints = cs.vertcat( constraints, -u[1,idh] + params['min_rates'][1] )

			# track constraints
			if track_cons:
				constraints = cs.vertcat( constraints, Aineq[2*idh:2*idh+2,:] @ x[:2,idh+1] - bineq[2*idh:2*idh+2,:] - eps[:,idh] )

		cost = cost_tracking + cost_actuation + cost_violation

		xvars = cs.vertcat(
			cs.reshape(x,-1,1),
			cs.reshape(u,-1,1),
			)
		if track_cons:
			xvars = cs.vertcat(
				xvars,
				cs.reshape(eps,-1,1),
				)

		pvars = cs.vertcat(
			cs.reshape(x0,-1,1), 
			cs.reshape(xref,-1,1), 
			cs.reshape(uprev,-1,1),
			)
		if track_cons:
			pvars = cs.vertcat(
				pvars,
				cs.reshape(Aineq,-1,1),
				cs.reshape(bineq,-1,1),
				)
		pvars = cs.vertcat(
			pvars,
			cs.reshape(X,-1,1),
			cs.reshape(alpha_vx,-1,1),
			cs.reshape(alpha_vy,-1,1),
			cs.reshape(alpha_omega,-1,1),
			cs.reshape(y_mean_vx,-1,1),
			cs.reshape(y_mean_vy,-1,1),
			cs.reshape(y_mean_omega,-1,1),
			cs.reshape(kin_use,-1,1),
		)
		self.nlp = {
			'x': xvars,
			'p': pvars,
			'f': cost, 
			'g': constraints,
			}
		ipoptoptions = {
			'print_level': 0,
			'print_timing_statistics': 'no',
			'max_iter': 100,
			}
		self.options = {
			'expand': True,
			'print_time': False,
			'ipopt': ipoptoptions,
		}
		self.name = 'gpmpc'
		self.problem = cs.nlpsol(self.name, 'ipopt', self.nlp, self.options)

	# def update_gp_models(self,gpmodels) :
	# 	vxgp = gpmodels['vx']
	# 	vygp = gpmodels['vy']
	# 	omegagp = gpmodels['omega']
	# 	xss = gpmodels['xscaler'].scale_
	# 	xsm = gpmodels['xscaler'].mean_
	# 	xvars = self.nlp['x']
	# 	n_inputs = self.model.n_inputs
	# 	x = cs.reshape(xvars[:((self.horizon+1)*self.n_states)],self.n_states,self.horizon+1)
	# 	u = cs.reshape(xvars[((self.horizon+1)*self.n_states):],n_inputs,self.horizon)
	# 	for idh in range(self.horizon):
	# 		dxdt = self.model.casadi(x[:,idh], u[:,idh], self.dxdtc)

	# 		if self.error_correction:
	# 			gpinput = (cs.vertcat(u[:,idh], x[6,idh], x[3:6,idh]).T - xsm.reshape(1,-1)) / xss.reshape(1,-1)
	# 			# a.reset_input()
	# 			# b.reset_input()
	# 			# c.reset_input()
	# 			# print(cs.SX.get_free(vxgp))
	# 			error = cs.vertcat(
	# 				cs.SX(0),
	# 				cs.SX(0),
	# 				cs.SX(0),
	# 				vxgp(gpinput)[0],
	# 				vygp(gpinput)[0],
	# 			 	omegagp(gpinput)[0],
	# 				cs.SX(0),
	# 				)
	# 			# print(a.serialize())
	# 			# print()
	# 			self.nlp['g'][self.n_states*(idh+1):self.n_states*(idh+2)] = x[:,idh+1] - x[:,idh] - self.Ts*dxdt - error
	# 			# print(constraints.size(), tm.time()-start)
	# 		# else:
	# 		# 	constraints = cs.vertcat( constraints, x[:,idh+1] - x[:,idh] - self.Ts*dxdt )
	# 	self.problem = cs.nlpsol(self.name, 'ipopt', self.nlp, self.options)

	def solve(self, x0, xref, uprev, vxm, vym, omegam, use_kinematic):
		
		n_states = self.model.n_states
		n_inputs = self.model.n_inputs
		horizon = self.horizon
		track_cons = self.track_cons
		if use_kinematic :
			kin_use = np.array([[0.]])
		else :
			kin_use = np.array([[1.]])
			# kin_use = 1.

		# track constraints
		if track_cons:
			Aineq = np.zeros([2*horizon,2])
			bineq = np.zeros([2*horizon,1])
			for idh in range(horizon):
				Ain, bin = Boundary(xref[:2,idh+1], self.track)
				Aineq[2*idh:2*idh+2,:] = Ain
				bineq[2*idh:2*idh+2] = bin
		else:
			Aineq = np.zeros([0,2])
			bineq = np.zeros([0,1])

		arg = {}
		# print(vxm.alpha_)
		arg['p'] = np.concatenate([
			x0.reshape(-1,1), 
			xref.T.reshape(-1,1), 
			uprev.reshape(-1,1),
			vxm.X_train_.T.reshape(-1,1),
			vxm.alpha_.reshape(-1,1),
			vym.alpha_.reshape(-1,1),
			omegam.alpha_.reshape(-1,1),
			vxm._y_train_mean.reshape(-1,1),
			vym._y_train_mean.reshape(-1,1),
			omegam._y_train_mean.reshape(-1,1),
			kin_use,
			])
		arg['lbx'] = -np.inf*np.ones( n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons )
		arg['ubx'] = np.inf*np.ones( n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons )
		arg['lbg'] =  np.concatenate( [np.zeros(n_states*(horizon+1)), -np.inf*np.ones(horizon*(6+2*track_cons))] )
		arg['ubg'] =  np.concatenate( [np.zeros(n_states*(horizon+1)), np.zeros(horizon*(6+2*track_cons))] )
		
		res = self.problem(**arg)
		fval = res['f'].full()[0][0]
		xmpc = res['x'][:n_states*(horizon+1)].full().reshape(horizon+1,n_states).T
		umpc = res['x'][n_states*(horizon+1):n_states*(horizon+1)+n_inputs*horizon].full().reshape(horizon,n_inputs).T
		return umpc, fval, xmpc
