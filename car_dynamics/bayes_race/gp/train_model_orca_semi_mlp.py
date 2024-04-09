"""    Train a GP model for error discrepancy between kinematic and dynamic models.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from bayes_race.params import ORCA, CarlaParams
from bayes_race.models import Kinematic6, Dynamic
import random
from bayes_race.utils.plots import plot_true_predicted_variance
import torch

#####################################################################
# load data

SAVE_MODELS = True
SAVE_PARAMS = False
MODEL_PATH = 'orca/semi_mlp-v1.pickle'
N_ITERS = 100
VARIDXs = [3,4,5]
state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']
torch.manual_seed(1)
random.seed(0)
np.random.seed(0)

alpha_f_distribution_y = np.zeros(2000)
alpha_f_distribution_x = np.arange(-1.,1.,2./2000)

alpha_r_distribution_y = np.zeros(2000)
alpha_r_distribution_x = np.arange(-1.,1.,2./2000)

class DynamicModel(torch.nn.Module):
    def __init__(self, model, deltat = 0.05):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.Rx = torch.nn.Sequential(torch.nn.Linear(1,1).to(torch.float64))
        
        self.Ry = torch.nn.Sequential(torch.nn.Linear(1,12).to(torch.float64), \
                    self.relu, \
                    torch.nn.Linear(12,1,bias=False).to(torch.float64))
        self.Ry[0].weight.data.fill_(1.)
        self.Ry[2].weight.data.fill_(0.)
        # self.Ry[2].bias.data.fill_(0.)
        self.Ry[0].bias.data = torch.arange(0.,.6,(0.6)/12.).to(torch.float64)
        self.Fy = torch.nn.Sequential(torch.nn.Linear(1,12).to(torch.float64), \
                    self.relu, \
                    torch.nn.Linear(12,1,bias=False).to(torch.float64))
        
        self.Fy[0].weight.data.fill_(1.)
        self.Fy[2].weight.data.fill_(0.)
        # self.Fy[2].bias.data.fill_(0.)
        self.Fy[0].bias.data = torch.arange(-.6,.6,(1.2)/12.).to(torch.float64)
        self.deltat = deltat
        self.model = model
        self.b = torch.arange(0.,.6,(0.6)/12.).to(torch.float64).unsqueeze(0)

    def first_layer(self,x):
        # y = torch.zeros_like(x)
        # for b in torch.arange(0.,.6,(0.6)/12.).to(torch.float64) :
        y = (x>self.b)*(x<self.b+(0.6)/12.)*(x-self.b) + (x>self.b+(0.6)/12.)*((0.6)/12.)
        return y
    
    def get_force_F(self,alpha_f) :
        yf = (alpha_f>0.)*self.first_layer(alpha_f) + (alpha_f<=0.)*self.first_layer(-alpha_f)
        Ffy = self.Fy[2](yf)[:,0]*(alpha_f[:,0]>0.) - self.Fy[2](yf)[:,0]*(alpha_f[:,0]<=0.)
        return Ffy
    
    def get_force_R(self,alpha_r) :
        yr = (alpha_r>0.)*self.first_layer(alpha_r) + (alpha_r<=0.)*self.first_layer(-alpha_r)
        Fry = self.Ry[2](yr)[:,0]*(alpha_r[:,0]>0.) - self.Ry[2](yr)[:,0]*(alpha_r[:,0]<=0.)
        return Fry
        
    def forward(self, x, debug=False):
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
        # print(torch.mean(pwm))
        out = torch.zeros_like(x[:,3:6])
        # print(out)
        for i in range(2) :
            vx = (x[:,3] + out[:,0]).unsqueeze(1)
            vy = x[:,4] + out[:,1]
            w = x[:,5] + out[:,2]
            alpha_f = (theta - torch.atan2(w*self.model.lf+vy,vx[:,0])).unsqueeze(1)
            alpha_r = torch.atan2(w*self.model.lr-vy,vx[:,0]).unsqueeze(1)
            # print(torch.mean(w),torch.mean(torch.atan2(w*self.model.lf+vy,vx[:,0])),torch.mean(alpha_f),torch.mean(alpha_r))
            if debug :
                for alpha in alpha_f[:,0] :
                    alpha_f_distribution_y[int((alpha+1.)*1000)] += 1
                for alpha in alpha_r[:,0] :
                    alpha_r_distribution_y[int((alpha+1.)*1000)] += 1
            # print(torch.max(alpha_r))
            yf = (alpha_f>0.)*self.first_layer(alpha_f) + (alpha_f<=0.)*self.first_layer(-alpha_f)
            yr = (alpha_r>0.)*self.first_layer(alpha_r) + (alpha_r<=0.)*self.first_layer(-alpha_r)
            # print(alpha_f[2000,:],yf[2000,:])
            Ffy = self.Fy[2](yf)[:,0]*(alpha_f[:,0]>0.) - self.Fy[2](yf)[:,0]*(alpha_f[:,0]<=0.)
            Fry = self.Ry[2](yr)[:,0]*(alpha_r[:,0]>0.) - self.Ry[2](yr)[:,0]*(alpha_r[:,0]<=0.)
            
            # Ffy = self.Fy(alpha_f)[:,0]*(alpha_f[:,0]>0.) - self.Fy(-alpha_f)[:,0]*(alpha_f[:,0]<=0.)
            # Fry = self.Ry(alpha_r)[:,0]*(alpha_r[:,0]>0.) - self.Ry(-alpha_r)[:,0]*(alpha_r[:,0]<=0.)
            Frx = self.Rx(vx)[:,0]
            
            # if debug :
            #     print(Ffy,Fry,Frx)
            # rpms = (vx[:,0]*60*3.45*0.919/(2*math.pi*0.34))
            # max_torques = (rpms<1700)*((rpms-1000)*550/700 + (1700-rpms)*380/700) \
            #     + (rpms>=1700)*(550)
            a_pred = (pwm>0)*self.model.Cm1*pwm*(3.45*0.919)/(0.34*1265) \
                + (pwm<=0)*self.model.Cm2*pwm*(3.45*0.919)/(0.34*1265)
            # Frx_kin = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm
            Frx = a_pred + self.model.Cm1*pwm
            vx_dot = (Frx-Ffy*torch.sin(theta)+vy*w-9.8*torch.sin(pitch))
            vy_dot = (Fry+Ffy*torch.cos(theta)-vx[:,0]*w)
            w_dot = self.model.mass*(Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/self.model.Iz
            out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat/2.
        out2 = (out)
        return out2

    
def load_data(CTYPE, TRACK_NAME, VARIDX, xscaler=None, yscaler=None):
    FILE_NAME = 'C:/Users/dvijk/vicon/processed_data/exp2_data.npz'
    data_dyn = np.load(FILE_NAME)
    # data_kin = np.load('../data/KIN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
    N_SAMPLES = data_dyn['states'].shape[1]-1
    y_all = (data_dyn['states'].T[:,3:]-data_dyn['states'].T[:,:-3])/3. #- data_kin['states'][:6,1:N_SAMPLES+1]
    # print(y_all)
    print(data_dyn['inputs'].shape)    
    print(data_dyn['states'].shape)    
    x = np.concatenate([
        data_dyn['inputs'].T[:,:-2].T,
        data_dyn['inputs'].T[1,:-2].reshape(1,-1).T,
        data_dyn['states'].T[3:6,:-3].T,
        data_dyn['states'].T[:,:-3].T],
        axis=1)
    y = y_all[VARIDX].reshape(-1,1)

    if xscaler is None or yscaler is None:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(x)
        yscaler.fit(y)
        return x, y
    
first = True
x_trains = []
y_trains = []
for VARIDX in VARIDXs :
    N_SAMPLES = 305
    
    x_train, y_train = load_data('PP', 'Carla', VARIDX)
    x_trains.append(torch.tensor(x_train))
    y_trains.append(torch.tensor(y_train))
    

#####################################################################
# load vehicle parmaeters (Use only geometric parameters)

params = CarlaParams(control='pwm')
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
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=20.,momentum=0.9)
# optimizer2 = torch.optim.SGD(model.parameters(), lr=.001,momentum=0.9)
loss_fn = torch.nn.MSELoss()

for i in range(10000) :
    # Zero your gradients for every batch!
    for param in model.Fy.parameters():
        param.requires_grad = True
    optimizer.zero_grad()
    # print(x_train[30:].shape)
    if i==4999 :
        outputs = model(x_train[30:],debug=False)
    else :
        outputs = model(x_train[30:])
    # print(y_train[30:].shape[0])
    # model.Fy.requires_grad_ = False
    
    # print(outputs.shape, y_train[30:].shape)
    loss = loss_fn(outputs[:,1:]*(x_train[30:,3:4]>1.), y_train[30:,1:]*(x_train[30:,3:4]>1.))
    loss.backward()
    print("Iter " + str(i) + " loss1 : ", loss.item())
    # Adjust learning weights
    optimizer.step()
    
    # for param in model.Fy.parameters():
    #     param.requires_grad = False

    # optimizer2.zero_grad()
    # outputs = model(x_train[30:])
    # loss2 = loss_fn(outputs[:,:1], y_train[30:,:1])
    # loss2.backward()
    # print("Iter " + str(i) + " loss2 : ", loss2.item())
    
    # # Adjust learning weights
    # optimizer2.step()

print("Training done")
end = time.time()
print('training time: %ss' %(end - start))        
print('Rx trained params : ', model.Rx[0].weight.data, model.Rx[0].bias.data)
# plt.plot(alpha_f_distribution_x,alpha_f_distribution_y)
# plt.plot(alpha_r_distribution_x,alpha_r_distribution_y)
# plt.show()
alpha_f = torch.tensor(np.arange(-.25,.25,0.004)).unsqueeze(1)
# Ffy = (alpha_f[:,0]>0)*model.Fy(alpha_f)[:,0].detach().numpy() - (alpha_f[:,0]<=0)*model.Fy(-alpha_f)[:,0].detach().numpy()
Ffy = model.get_force_F(alpha_f).detach().numpy()
Ffy_true = params['Df']*torch.sin(params['Cf']*torch.atan(params['Bf']*alpha_f))
# print(Ffy)
plt.plot(alpha_f,Ffy)
# plt.plot(alpha_f,Ffy_true)
plt.show()
model.eval()
alpha_r = torch.tensor(np.arange(-.25,.25,0.004)).unsqueeze(1)
# Fry = (alpha_r[:,0]>0)*model.Ry(alpha_r)[:,0].detach().numpy() - (alpha_r[:,0]<=0)*model.Ry(-alpha_r)[:,0].detach().numpy()
Fry = model.get_force_R(alpha_r).detach().numpy()
Fry_true = params['Dr']*torch.sin(params['Cr']*torch.atan(params['Br']*alpha_r))

start = time.time()
temperature = 1.
wt = np.exp(-temperature*np.abs(alpha_r[:,0]))
p_fit = np.polyfit(alpha_r[:,0],Fry,deg=3,w=wt)
print(p_fit)
Fry_fit = p_fit[3]
# X = alpha_r[:,0]
for i in range(3) :
    Fry_fit += p_fit[3-i-1]*alpha_r[:,0]**(i+1)
    # X *= alpha_r[:,0]
end = time.time()
print("Poly fit time : ", (start-end))

# print(Ffy)
plt.plot(alpha_r[:,0],Fry)
# plt.plot(alpha_r[:,0],Fry_true)
# plt.plot(alpha_r[:,0],Fry_fit)
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