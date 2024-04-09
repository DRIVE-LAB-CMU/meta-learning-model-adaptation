"""	Setup NLP with GP models in CasADi.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import numpy as np
import casadi as cs

from bayes_race.models import Kinematic6
from bayes_race.mpc.constraints import Boundary
import time as tm
import math
import torch

class setupNLP:

	def sigmoid(self,x,L,R) :
		return (((1./(1+cs.exp(-5.*(x-L)))) - .5) + \
			((1./(1+cs.exp(5.*(x-R)))) - .5))*x + (cs.exp(-5.*(x-L))/(1+cs.exp(-5.*(x-L))))*L + \
			(cs.exp(5.*(x-R))/(1+cs.exp(5.*(x-R))))*R

	def __init__(self, horizon, Ts, Q, P, R, params, models, track, GP_EPS_LEN = 305, 
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
		self.prev_x = None
		Rx_model = models['Rx']
		Ry_model = models['Ry']
		Fy_model = models['Fy']
		self.itr = 0
		Rx_inps = [models['Rx'].sx_in(1),models['Rx'].sx_in(2)]
		Ry_inps = [models['Ry'].sx_in(1),models['Ry'].sx_in(2),models['Ry'].sx_in(3),models['Ry'].sx_in(4)]
		Fy_inps = [models['Fy'].sx_in(1),models['Fy'].sx_in(2),models['Fy'].sx_in(3),models['Fy'].sx_in(4)]
		# Casadi vaiables
		x0 = cs.SX.sym('x0', self.n_states, 1)
		xref = cs.SX.sym('xref', xref_size, horizon+1)
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
		
		idh = 0
		vx = x[3,idh]+0.001
		vy = x[4,idh]
		w = x[5,idh]
		pwm = u[0,idh]
		# gpinput = (cs.vertcat(u[:,idh], x[6,idh], x[3:6,idh],w/vx,w*vy,w*vx,pwm*vx,vx*vx,vy/vx).T - xsm.reshape(1,-1)) / xss.reshape(1,-1)
		# print(vxgp.print_options())
		start = tm.time()

		for idh in range(horizon):
			dxdt = model.casadi(x[:,idh], u[:,idh], self.dxdtc)

			if error_correction:
				error = cs.vertcat(
						cs.SX(0),
						cs.SX(0),
						cs.SX(0),
						cs.SX(0),
						cs.SX(0),
						cs.SX(0),
						cs.SX(0),
						)
				
				for i in range(1) :
					# vx = 
					# vx = self.sigmoid(x[3,idh] + error[3],0.1,5.)
					# vx = cs.if_else(vx>0.1,vx,0.1)
					# vx = cs.if_else(vx<5.,vx,5.)
					psi = x[2,idh]
					vy = x[4,idh] + error[4]
					w = x[5,idh] + error[5]
					delta = x[6,idh]
					pwm = u[0,idh]
					vx = x[3,idh] + error[3]
					vmin = 0.05
					vy = cs.if_else(vx<vmin, 0, vy)
					w = cs.if_else(vx<vmin, 0, w)
					# delta = cs.if_else(vx<vmin, 0, delta)
					vx = cs.if_else(vx<vmin, vmin, vx)
					alpha_f = (delta - cs.atan2(w*params['lf']+vy,vx))
					alpha_r = cs.atan2(w*params['lr']-vy,vx)
					
					# alpha_f = self.sigmoid(alpha_f,-0.5,.5)
					# alpha_r = self.sigmoid(alpha_r,-0.5,.5)
					alpha_f = cs.if_else(alpha_f>-0.5,alpha_f,-0.5)
					alpha_r = cs.if_else(alpha_r>-0.5,alpha_r,-0.5)
					alpha_f = cs.if_else(alpha_f<0.5,alpha_f,0.5)
					alpha_r = cs.if_else(alpha_r<0.5,alpha_r,0.5)
					Ffy_ins = [alpha_f] + Fy_inps
					Ffy = Fy_model(*Ffy_ins)[0]
					Fry_ins = [alpha_r] + Ry_inps
					Fry = Ry_model(*Fry_ins)[0]
					Frx_ins = [vx**2] + Rx_inps
					Frx = (params['Cm1'] - params['Cm2']*vx)*pwm + Rx_model(*Frx_ins)[0]
					vx_dot = (Frx-Ffy*cs.sin(delta)+params['mass']*vy*w)/params['mass']
					vy_dot = (Fry+Ffy*cs.cos(delta)-params['mass']*vx*w)/params['mass']
					w_dot = (Ffy*params['lf']*cs.cos(delta)-Fry*params['lr'])/params['Iz']
					x_dot = vx*cs.cos(psi) - vy*cs.sin(psi)
					y_dot = vx*cs.sin(psi) + vy*cs.cos(psi)
					psi_dot = w
		
					error += cs.vertcat(
						x_dot,
						y_dot,
						psi_dot,
						vx_dot[0],
						vy_dot[0],
						w_dot[0],
						u[1,idh],
						) *Ts
				
				constraints = cs.vertcat( constraints, x[:,idh+1] - x[:,idh] - (1-kin_use)*Ts*dxdt - kin_use*error )
				# print(constraints.size(), tm.time()-start, "haha")
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
		# print(pvars.shape)
		# print(pvars.shape)
		
		for inp in Rx_inps :
			pvars = cs.vertcat(
				pvars,
				cs.reshape(inp,-1,1)
			)	
		for inp in Ry_inps :
			pvars = cs.vertcat(
				pvars,
				cs.reshape(inp,-1,1)
			)	
		for inp in Fy_inps :
			pvars = cs.vertcat(
				pvars,
				cs.reshape(inp,-1,1)
			)	
		pvars = cs.vertcat(
			pvars,
			cs.reshape(kin_use,-1,1)
		)
		if track_cons:
			pvars = cs.vertcat(
				pvars,
				cs.reshape(Aineq,-1,1),
				cs.reshape(bineq,-1,1),
				)
		# print(pvars.shape)
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

	def solve(self, x0, xref, uprev, use_kinematic,models):
		
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
		p_list = [
			x0.reshape(-1,1), 
			xref.T.reshape(-1,1), 
			uprev.reshape(-1,1),
			]
		for i,layer in enumerate(models.Rx):
			if isinstance(layer, torch.nn.Linear):
				weights = layer.weight.detach().numpy().T.reshape(-1,1)
				p_list.append(weights)
		for i,layer in enumerate(models.Rx):
			if isinstance(layer, torch.nn.Linear):
				biases = np.expand_dims(layer.bias.detach().numpy(),0).reshape(-1,1)
				p_list.append(biases)
		
		for i,layer in enumerate(models.Ry):
			if isinstance(layer, torch.nn.Linear):
				weights = layer.weight.detach().numpy().T.reshape(-1,1)
				p_list.append(weights)
		for i,layer in enumerate(models.Ry):
			if isinstance(layer, torch.nn.Linear):
				biases = np.expand_dims(layer.bias.detach().numpy(),0).reshape(-1,1)
				p_list.append(biases)
		
		for i,layer in enumerate(models.Fy):
			if isinstance(layer, torch.nn.Linear):
				weights = layer.weight.detach().numpy().T.reshape(-1,1)
				p_list.append(weights)
		for i,layer in enumerate(models.Fy):
			if isinstance(layer, torch.nn.Linear):
				biases = np.expand_dims(layer.bias.detach().numpy(),0).reshape(-1,1)
				p_list.append(biases)
		p_list.append(kin_use)
		p_list.append(Aineq.T.reshape(-1,1))
		p_list.append(bineq.T.reshape(-1,1))
		arg['p'] = np.concatenate(p_list)
		arg['lbx'] = -np.inf*np.ones( n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons )
		arg['ubx'] = np.inf*np.ones( n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons )
		arg['lbg'] =  np.concatenate( [np.zeros(n_states*(horizon+1)), -np.inf*np.ones(horizon*(6+2*track_cons))] )
		arg['ubg'] =  np.concatenate( [np.zeros(n_states*(horizon+1)), np.zeros(horizon*(6+2*track_cons))] )
		if self.prev_x is not None and self.itr>30:
			# print("Opted for this 1")
			arg['x0'] = self.prev_x
		elif self.prev_x is not None :
			# print("Opted for this 2")
			arg['x0'] = np.zeros_like(self.prev_x)

		res = self.problem(**arg)
		fval = res['f'].full()[0][0]
		self.prev_x = res['x']
		self.itr += 1
		# vx = cs.if_else(vx>0.1,vx,0.1)
		# vy = x[4,idh] + error[4]
		# w = x[5,idh] + error[5]
		# delta = x[6,idh]+error[6]
		# alpha_f = (delta - cs.atan2(w*params['lf']+vy,vx))
		# alpha_r = cs.atan2(w*params['lr']-vy,vx)
				
		xmpc = res['x'][:n_states*(horizon+1)].full().reshape(horizon+1,n_states).T
		umpc = res['x'][n_states*(horizon+1):n_states*(horizon+1)+n_inputs*horizon].full().reshape(horizon,n_inputs).T
		# if res['cost'] > 10. :
		# print(xmpc[3:6,:])
		return umpc, fval, xmpc
