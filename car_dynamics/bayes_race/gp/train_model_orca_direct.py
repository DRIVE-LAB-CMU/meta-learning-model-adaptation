"""	Train a GP model for error discrepancy between kinematic and dynamic models.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from bayes_race.params import ORCA
from bayes_race.models import Kinematic6, Dynamic

from bayes_race.utils.plots import plot_true_predicted_variance
import torch

#####################################################################
# load data

SAVE_MODELS = False
SAVE_PARAMS = False
N_ITERS = 100
VARIDXs = [3,4,5]
state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']
lf = 0.029
lr = 0.033

class DynamicModel(torch.nn.Module):
	def __init__(self, model, deltat = 0.01):
		"""
		In the constructor we instantiate four parameters and assign them as
		member parameters.
		"""
		super().__init__()
		# self.Iz = torch.nn.Parameter(torch.randn(()))
		# self.Bf = torch.nn.Parameter(torch.randn(()))
		# self.Cf = torch.nn.Parameter(torch.randn(()))
		# self.Df = torch.nn.Parameter(torch.randn(()))
		# self.Br = torch.nn.Parameter(torch.randn(()))
		# self.Cr = torch.nn.Parameter(torch.randn(()))
		# self.Dr = torch.nn.Parameter(torch.randn(()))
		# self.Cr0 = torch.nn.Parameter(torch.randn(()))
		# self.Cr2 = torch.nn.Parameter(torch.randn(()))
		# self.Iz = torch.nn.Parameter(torch.Tensor([model.Iz]))
		self.Bf = torch.nn.Parameter(torch.Tensor([model.Bf]))
		self.Cf = torch.nn.Parameter(torch.Tensor([model.Cf]))
		self.Df = torch.nn.Parameter(torch.Tensor([3.*model.Df]))
		self.Br = torch.nn.Parameter(torch.Tensor([model.Br]))
		self.Cr = torch.nn.Parameter(torch.Tensor([model.Cr]))
		self.Dr = torch.nn.Parameter(torch.Tensor([3.*model.Dr]))
		self.Cr0 = torch.nn.Parameter(torch.Tensor([model.Cr0]))
		self.Cr2 = torch.nn.Parameter(torch.Tensor([model.Cr2]))
		self.deltat = deltat
		self.model = model
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		# print(x.shape)
		# out = X
		deltatheta = x[:,1]
		theta = x[:,2]
		pwm = x[:,0]
		out = torch.zeros_like(x[:,3:6])
		# print(out)
		for i in range(2) :
			# print(x[274,3:6])
			# print(out[274])
			vx = x[:,3] + out[:,0]
			vy = x[:,4] + out[:,1]
			w = x[:,5] + out[:,2]
			alpha_f = theta - torch.atan2(w*self.model.lf+vy,self.relu(vx-0.1)+0.1)
			alpha_r = torch.atan2(w*self.model.lr-vy,self.relu(vx-0.1)+0.1)
			# print(alpha_f,alpha_r)
			Ffy = self.Df*torch.sin(self.Cf*torch.atan(self.Bf*alpha_f))
			Fry = self.Dr*torch.sin(self.Cr*torch.atan(self.Br*alpha_r))
			Frx = (self.model.Cm1-self.model.Cm2*vx)*pwm - self.Cr0 - self.Cr2*vx**2
			Frx_kin = (self.model.Cm1-self.model.Cm2*vx)*pwm
			vx_dot = (Frx-Ffy*torch.sin(theta)+self.model.mass*vy*w)/self.model.mass
			vy_dot = (Fry+Ffy*torch.cos(theta)-self.model.mass*vx*w)/self.model.mass
			w_dot = (Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/self.model.Iz
			# print(vx_dot[274],Frx[274],(self.Cr2*vx**2)[274],self.Cr2)
			# print(vy_dot.shape)
			# print(w_dot.shape)
			out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat
		out2 = (out)
		return out2

	def string(self):
		"""
		Just like any class in Python, you can also define custom method on PyTorch modules
		"""
		return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

def load_data(CTYPE, TRACK_NAME, VARIDX, xscaler=None, yscaler=None):
	data_dyn = np.load('../data/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	data_kin = np.load('../data/KIN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	y_all = (data_dyn['states'][:6,1:N_SAMPLES+1]-data_dyn['states'][:6,:N_SAMPLES]) #- data_kin['states'][:6,1:N_SAMPLES+1]
	x = np.concatenate([
		data_kin['inputs'][:,:N_SAMPLES].T,
		data_dyn['inputs'][1,:N_SAMPLES].reshape(1,-1).T,
		data_dyn['states'][3:6,:N_SAMPLES].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	if xscaler is None or yscaler is None:
		xscaler = StandardScaler()
		yscaler = StandardScaler()
		xscaler.fit(x)
		yscaler.fit(y)
		return x, y, xscaler, yscaler
	else:
		return xscaler.transform(x), yscaler.transform(y)

first = True
x_trains = []
y_trains = []
xscalers = []
yscalers = []
for VARIDX in VARIDXs :
	N_SAMPLES = 305
	filename = 'orca/{}gp.pickle'.format(state_names[VARIDX])
	filename_params = 'orca/{}gp_params.pickle'.format(state_names[VARIDX])

	x_train, y_train, xscaler, yscaler = load_data('PP', 'ETHZMobil', VARIDX)
	x_trains.append(torch.tensor(x_train))
	y_trains.append(torch.tensor(y_train))
	xscalers.append(xscaler)
	yscalers.append(yscaler)


#####################################################################
# load vehicle parmaeters (Use only geometric parameters)

params = ORCA(control='pwm')
vehicle_model = Dynamic(**params)


#####################################################################
# train GP model

x_train = x_trains[0]
y_train = torch.cat(y_trains,axis=1)
model = DynamicModel(vehicle_model)
start = time.time()

#####################################################################
# Train the model

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for i in range(500) :
	# Zero your gradients for every batch!
	optimizer.zero_grad()
	# print(x_train[:5,:4])
	# print(x_train[1:6,2]-x_train[:5,2])
	# Make predictions for this batch
	outputs = model(x_train[30:])
	if i==499 or i==0:
		print(outputs[:,2],y_train[:,2])
	loss = loss_fn(outputs, y_train[30:])
	loss.backward()
	print("Iter " + str(i) + " loss : ", loss.item())
	# Adjust learning weights
	optimizer.step()
	model.Cr2.data = model.Cr2.data.clamp(0,0.001) 
	model.Cr0.data = model.Cr0.data.clamp(0,0.2) 
	model.Bf.data = model.Bf.data.clamp(0,10.) 
	model.Cf.data = model.Cf.data.clamp(0,10.) 
	model.Df.data = model.Df.data.clamp(0,10.) 
	model.Br.data = model.Br.data.clamp(0,10.) 
	model.Cr.data = model.Cr.data.clamp(0,10.) 
	model.Dr.data = model.Dr.data.clamp(0,10.) 

# print("Iz : ", model.Iz.item(),params['Iz'])
print("Bf : ", model.Bf.item(),params['Bf'])
print("Cf : ", model.Cf.item(),params['Cf'])
print("Df : ", model.Df.item(),params['Df'])
print("Br : ", model.Br.item(),params['Br'])
print("Cr : ", model.Cr.item(),params['Cr'])
print("Dr : ", model.Dr.item(),params['Dr'])
print("Cr0 : ", model.Cr0.item(),params['Cr0'])
print("Cr2 : ", model.Cr2.item(),params['Cr2'])

print("Training done")
end = time.time()
print('training time: %ss' %(end - start))		

