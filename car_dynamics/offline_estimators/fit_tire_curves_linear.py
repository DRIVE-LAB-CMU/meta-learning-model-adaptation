""" Train an ELM model for tires and alo predict aerodynamic parameters given the data
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import sys

# print(sys.path)
sys.path.append('./')
import time
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from bayes_race.params import ORCA, CarlaParams, RCParams
from bayes_race.models import Kinematic6, Dynamic
import random
from bayes_race.utils.plots import plot_true_predicted_variance
import torch
import argparse

#####################################################################
# load data

parser = argparse.ArgumentParser(description='Arguments for offline fitting of tire curves')
# parser.add_argument('--file', required=False, default='processed_data/vicon-circle-data-20240302-143549-1_data.npz',help='Specify the file path')
parser.add_argument('--file', required=False, default='processed_data/vicon-circle-data-20240306-001713_data.npz',help='Specify the file path')
# parser.add_argument('--file', required=False, default='processed_data/vicon-circle-data-20240225-215833_data.npz',help='Specify the file path')
parser.add_argument('--model_path', required=False, default='orca/semi_mlp-v1.pickle',help='Specify the model save file path')
parser.add_argument('--dt', required=False, default=0.05,help='Specify the time step')
parser.add_argument('--ignore_first', required=False, default=30,help='Specify the no of initial time steps to ignore')
parser.add_argument('--min_v', required=False, default=1.,help='Min velocity for masking out')
parser.add_argument('--max_v', required=False, default=3.5,help='Max velocity for masking out')
parser.add_argument('--resolution', required=False, default=3,help='Resolution for finite difference')
parser.add_argument('--n_iters', required=False, default=2000,help='No of iters for training')
parser.add_argument('--save', required=False, action='store_true',help='Whether to save trained model')
parser.add_argument('--seed', required=False, default=1,help='seed for random number generator')

# Parse the command-line arguments
args = parser.parse_args()

print(args.file)
SAVE_MODELS = args.save
MODEL_PATH = args.model_path
N_ITERS = int(args.n_iters)
FILE_NAME = args.file
RES = int(args.resolution)
ignore_first = args.ignore_first
min_v = float(args.min_v)
max_v = float(args.max_v)

state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']

torch.manual_seed(args.seed)
random.seed(0)
np.random.seed(0)

alpha_f_distribution_y = np.zeros(2000)
alpha_f_distribution_x = np.arange(-1.,1.,2./2000)

alpha_r_distribution_y = np.zeros(2000)
alpha_r_distribution_x = np.arange(-1.,1.,2./2000)

class ResidualModel(torch.nn.Module):
    def __init__(self, model, deltat = args.dt):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.Rx = torch.nn.Linear(1,1).to(torch.float64)
        self.Rx.weight.data.fill_(0.)
        
        self.Ry = torch.nn.Linear(1,1,bias=False).to(torch.float64)
        self.Ry.weight.data.fill_(10.)
        
        self.Fy = torch.nn.Linear(1,1,bias=False).to(torch.float64)
        self.Fy.weight.data.fill_(10.)
        
        self.deltat = deltat
        self.model = model
        self.b = torch.arange(0.,.6,(0.6)/12.).to(torch.float64).unsqueeze(0)

    
    def get_force_F(self,alpha_f) :
        return self.Fy(alpha_f)
        
    def get_force_R(self,alpha_r) :
        return self.Ry(alpha_r)
        
    def forward(self, x, debug=False,n_divs=2):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # print(x.shape)
        # out = X
        pitch = torch.tensor(0.)#x[:,7]
        theta = x[:,2]
        pwm = x[:,0]
        out = torch.zeros_like(x[:,3:6])
        for i in range(n_divs) :
            vx = (x[:,3] + out[:,0]).unsqueeze(1)
            vy = x[:,4] + out[:,1]
            w = x[:,5] + out[:,2]
            alpha_f = (theta - torch.atan2(w*self.model.lf+vy,vx[:,0])).unsqueeze(1)
            alpha_r = torch.atan2(w*self.model.lr-vy,vx[:,0]).unsqueeze(1)
            # print(self.model.lf,self.model.lr,vx.shape,vy.shape,w.shape,alpha_f.shape,alpha_r.shape)
            # print(torch.mean(vx),torch.mean(torch.atan2(w*self.model.lf+vy,vx[:,0])),torch.mean(alpha_f),torch.mean(alpha_r))
            if debug :
                for alpha in alpha_f[:,0] :
                    alpha_f_distribution_y[int((alpha+1.)*1000)] += 1
                for alpha in alpha_r[:,0] :
                    alpha_r_distribution_y[int((alpha+1.)*1000)] += 1
            # print(alpha_f.shape,alpha_r.shape)
            Ffy = self.Fy(alpha_f)[:,0]
            Fry = self.Ry(alpha_r)[:,0]
            Frx = self.Rx(vx)[:,0]
            # print(Ffy.shape)
            a_pred = (pwm>0)*self.model.Cm1*pwm*(3.45*0.919)/(0.34*1265) \
                + (pwm<=0)*self.model.Cm2*pwm*(3.45*0.919)/(0.34*1265)
            # Frx_kin = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm
            Frx = 0.#a_pred + self.model.Cm1*pwm
            vx_dot = (Frx-Ffy*torch.sin(theta)+vy*w-9.8*torch.sin(pitch))
            vy_dot = (Fry+Ffy*torch.cos(theta)-vx[:,0]*w)
            # print(self.model.mass/self.model.Iz)
            w_dot = self.model.mass*(Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/self.model.Iz
            out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat/n_divs
        out2 = (out)
        return out2


def load_data(file_name):
    data_dyn = np.load(file_name)
    y_all = (data_dyn['states'].T[:,RES:]-data_dyn['states'].T[:,:-3])/RES
    # print(data_dyn['inputs'].shape)    
    # print(data_dyn['states'].shape)    
    x = np.concatenate([
        data_dyn['inputs'].T[:,:-(RES-1)].T,
        data_dyn['inputs'].T[1,:-(RES-1)].reshape(1,-1).T,
        data_dyn['states'].T[3:6,:-RES].T,
        data_dyn['states'].T[:,:-RES].T],
        axis=1)
    y = y_all[3:6].T
    mask = x[:,-1:]
    return torch.tensor(x), torch.tensor(y), torch.tensor(mask)
    
    
x_train, y_train, mask = load_data(FILE_NAME)


#####################################################################
# load vehicle parmaeters (Use only geometric parameters)

params = RCParams(control='pwm')
vehicle_model = Dynamic(**params)


#####################################################################
# train GP model

model = ResidualModel(vehicle_model)
start = time.time()

#####################################################################
# Train the model

# Optimizers specified in the torch.optim package
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=50.,momentum=0.9)
loss_fn = torch.nn.MSELoss()

for i in range(N_ITERS) :
    # Zero your gradients for every batch!
    for param in model.Fy.parameters():
        param.requires_grad = True
    optimizer.zero_grad()
    outputs = model(x_train[ignore_first:],debug=False)
    # if i==4999 :
    #     outputs = model(x_train[30:],debug=False)
    # else :
    #     outputs = model(x_train[30:])
    # print(y_train[30:].shape[0])
    # model.Fy.requires_grad_ = False
    diff_vx = np.abs(x_train[ignore_first:,3:4]-x_train[ignore_first-1:-1,3:4])
    # mask = (x_train[ignore_first:,3:4]>min_v)*(x_train[ignore_first:,3:4]<max_v)*(diff_vx<1.5)
    loss = loss_fn(outputs[:,1:]*mask[ignore_first:,:], y_train[ignore_first:,1:]*mask[ignore_first:,:])
    loss.backward()
    print("Iter " + str(i) + " loss1 : ", loss.item())
    
    # Adjust learning weights
    optimizer.step()
    
    
print("Training done")
end = time.time()
print('training time: %ss' %(end - start))        
print('Rx trained params : ', model.Rx.weight.data, model.Rx.bias.data)
# plt.plot(alpha_f_distribution_x,alpha_f_distribution_y)
# plt.plot(alpha_r_distribution_x,alpha_r_distribution_y)
# plt.show()
alpha_f = torch.tensor(np.arange(-.25,.25,0.004)).unsqueeze(1)
Ffy = model.get_force_F(alpha_f).detach().numpy()
Ffy_true = params['Df']*torch.sin(params['Cf']*torch.atan(params['Bf']*alpha_f))
plt.plot(alpha_f,Ffy)
# plt.plot(alpha_f,Ffy_true)
plt.show()
model.eval()

alpha_r = torch.tensor(np.arange(-.25,.25,0.004)).unsqueeze(1)
Fry = model.get_force_R(alpha_r).detach().numpy()
Fry_true = params['Dr']*torch.sin(params['Cr']*torch.atan(params['Br']*alpha_r))

plt.plot(alpha_r[:,0],Fry)
plt.show()
# print("Iz : ", model.Iz.item(),params['Iz'])
print("Bf : ", params['Bf'])
print("Cf : ", params['Cf'])
print("Df : ", params['Df'])
print("Br : ", params['Br'])
print("Cr : ", params['Cr'])
print("Dr : ", params['Dr'])
print("Cr0 : ", params['Cr0'])
print("Cr2 : ", params['Cr2'])


y_train_mu = model(x_train).detach()

MSE = mean_squared_error(y_train, y_train_mu, multioutput='raw_values')
R2Score = r2_score(y_train, y_train_mu, multioutput='raw_values')
EV = explained_variance_score(y_train, y_train_mu, multioutput='raw_values')

print('root mean square error: %s' %(np.sqrt(MSE)))
print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_train.mean()))))
print('R2 score: %s' %(R2Score))
print('explained variance: %s' %(EV))

# plot results
for VARIDX in [3,4,5] :
    y_train_std = np.zeros_like(y_train_mu)
    plot_true_predicted_variance(
        y_train[:,VARIDX-3].numpy(), y_train_mu[:,VARIDX-3].numpy(), y_train_std[:,VARIDX-3], 
        ylabel='{} '.format(state_names[VARIDX]), xlabel='sample index'
        )


    plt.show()

if SAVE_MODELS :
    torch.save(model.state_dict(), MODEL_PATH)