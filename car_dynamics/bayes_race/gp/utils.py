""" GP model in CasADi.
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import time
import numpy as np
import casadi as cs
from scipy.linalg import solve_triangular
from sklearn.neural_network import MLPRegressor
import torch

def CasadiRBF(X, Y, model):
    """ RBF kernel in CasADi
    """
    sX = X.shape[0]
    sY = Y.shape[0]    
    length_scale = model.kernel_.get_params()['k1__k2__length_scale'].reshape(1,-1)
    constant = model.kernel_.get_params()['k1__k1__constant_value']
    X = X / cs.repmat(length_scale, sX , 1)
    Y = Y / cs.repmat(length_scale, sY , 1)
    dist = cs.repmat(cs.sum1(X.T**2).T,1,sY) + cs.repmat(cs.sum1(Y.T**2),sX,1) - 2*cs.mtimes(X,Y.T)
    K = constant*cs.exp(-.5 * dist)
    return K

def CasadiConstant(X, Y, model):
    """ Constant kernel in CasADi
    """
    constant = model.kernel_.get_params()['k2__constant_value']
    sX = X.shape[0]
    sY = Y.shape[0]
    K = constant*cs.SX.ones((sX, sY))
    return K

def CasadiMatern(X, Y, model):
    """ Matern kernel in CasADi
    """
    length_scale = model.kernel_.get_params()['k2__length_scale'].reshape(1,-1)
    constant = model.kernel_.get_params()['k1__constant_value']
    nu = model.kernel_.get_params()['k2__nu']

    sX = X.shape[0]
    sY = Y.shape[0]
    X = X / cs.repmat(length_scale, sX , 1)
    Y = Y / cs.repmat(length_scale, sY , 1)
    dist = cs.repmat(cs.sum1(X.T**2).T,1,sY) + cs.repmat(cs.sum1(Y.T**2),sX,1) - 2*cs.mtimes(X,Y.T)

    if nu == 0.5:
        K = constant*cs.exp(-dist**0.5)
    elif nu == 1.5:
        K = np.sqrt(3)*dist**0.5
        K = constant*(1. + K) * cs.exp(-K)
    elif nu == 2.5:
        K = np.sqrt(5)*dist**0.5
        K = constant*(1. + K + 5/3*dist) * cs.exp(-K)
    else:
        raise NotImplementedError
    return K
    
def loadGPModel(name, model, xscaler, yscaler, kernel='RBF'):
    """ GP mean and variance as casadi.SX variable
    """
    X = model.X_train_
    x = cs.SX.sym('x', 1, X.shape[1])

    # mean
    if kernel == 'RBF':
        K1 = CasadiRBF(x, X, model)
        K2 = CasadiConstant(x, X, model)
        K = K1 + K2
    elif kernel == 'Matern':
        K = CasadiMatern(x, X, model)
    else:
        raise NotImplementedError

    y_mu = cs.mtimes(K, model.alpha_) + model._y_train_mean
    y_mu = y_mu * yscaler.scale_ + yscaler.mean_

    # variance
    L_inv = solve_triangular(model.L_.T,np.eye(model.L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    if kernel == 'RBF':
        K1_ = CasadiRBF(x, x, model)
        K2_ = CasadiConstant(x, x, model)
        K_ = K1_ + K2_
    elif kernel == 'Matern':
        K_ = CasadiMatern(x, x, model)

    y_var = cs.diag(K_) - cs.sum2(cs.mtimes(K, K_inv)*K)
    y_var = cs.fmax(y_var, 0)
    y_std = cs.sqrt(y_var)
    y_std *= yscaler.scale_

    gpmodel = cs.Function(name, [x], [y_mu, y_mu])
    # print(y_std)
    return gpmodel


def loadGPModelVars(name, model, xscaler, yscaler, kernel='RBF'):
    """ GP mean and variance as casadi.SX variable
    """
    
    X = cs.SX.sym('X', 305, 6)
    x = cs.SX.sym('x', 1, X.shape[1])

    # mean
    if kernel == 'RBF':
        K1 = CasadiRBF(x, X, model)
        K2 = CasadiConstant(x, X, model)
        K = K1 + K2
    elif kernel == 'Matern':
        K = CasadiMatern(x, X, model)
    else:
        raise NotImplementedError

    alpha = cs.SX.sym('alpha', 305, 1)
    y_mean = cs.SX.sym('y_mean', 1, 1)
    y_mu = cs.mtimes(K, alpha) + y_mean
    y_mu = y_mu * yscaler.scale_ + yscaler.mean_

    # variance
    L_inv = solve_triangular(model.L_.T,np.eye(model.L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    if kernel == 'RBF':
        K1_ = CasadiRBF(x, x, model)
        K2_ = CasadiConstant(x, x, model)
        K_ = K1_ + K2_
    elif kernel == 'Matern':
        K_ = CasadiMatern(x, x, model)

    y_var = cs.diag(K_) - cs.sum2(cs.mtimes(K, K_inv)*K)
    y_var = cs.fmax(y_var, 0)
    y_std = cs.sqrt(y_var)
    y_std *= yscaler.scale_

    gpmodel = cs.Function(name, [x,X,alpha,y_mean], [y_mu, y_mu])
    # print(y_std)
    return gpmodel


def loadMLPModel(name, mlp_model: MLPRegressor, yscaler):
    # Get the number of input and output layers from the model
    n_inputs = mlp_model.coefs_[0].shape[0]
    n_outputs = mlp_model.coefs_[-1].shape[1]
    
    # Define the CasADi variables for the inputs and outputs
    inputs = cs.MX.sym("inputs", 1, n_inputs)
    outputs = cs.MX.sym("outputs", 1, n_outputs)
    
    # Define the hidden layers using the weights and biases from the scikit-learn model
    hidden = inputs
    for i in range(len(mlp_model.coefs_)-1):
        weights = mlp_model.coefs_[i]
        biases = np.expand_dims(mlp_model.intercepts_[i],0)
        hidden_pre = cs.mtimes(hidden, weights) + biases
        hidden = cs.tanh(hidden_pre)
        print(i,hidden.shape)
    
    # Define the output layer using the final weights and biases
    output_weights = mlp_model.coefs_[-1]
    output_biases = np.expand_dims(mlp_model.intercepts_[-1],0)
    print(output_weights.shape)
    outputs = cs.mtimes(hidden, output_weights) + output_biases
    outputs = outputs * yscaler.scale_ + yscaler.mean_

    # Create the CasADi function
    f = cs.Function(name, [inputs], [outputs])
    
    return f

def loadTorchModel(name, nn_model):
    # Get the number of input and output layers from the model
    n_inputs = nn_model[0].in_features
    n_outputs = nn_model[-1].out_features
    
    # Define the CasADi variables for the inputs and outputs
    inputs = cs.MX.sym("inputs", 1, n_inputs)
    outputs = cs.MX.sym("outputs", 1, n_outputs)
    
    # Define the hidden layers using the weights and biases from the scikit-learn model
    hidden = inputs
    for i,layer in enumerate(nn_model):
        if isinstance(layer, torch.nn.Linear):
            weights = layer.weight.detach().numpy().T
            biases = np.expand_dims(layer.bias.detach().numpy(),0)
            hidden = cs.mtimes(hidden, weights) + biases
        if isinstance(layer,torch.nn.ReLU) :
            hidden = cs.if_else(hidden>0.,hidden,0.)
        if isinstance(layer,torch.nn.Tanh) :
            hidden = cs.tanh(hidden)
  
    outputs = hidden
    f = cs.Function(name, [inputs], [outputs])
    return f

def loadTorchModelEq(name, nn_model):
    # Get the number of input and output layers from the model
    alpha_r = torch.tensor(np.arange(-.6,.6,0.004)).unsqueeze(1)
    temperature = 1.
    Fy = nn_model(alpha_r)[:,0].detach().numpy()
    wt = np.exp(-temperature*np.abs(alpha_r[:,0]))
    p_fit = np.polyfit(alpha_r[:,0],Fy,deg=3,w=wt)

    # Define the CasADi variables for the inputs and outputs
    inputs = cs.MX.sym("inputs", 1, 1)
    outputs = cs.MX.sym("outputs", 1, 1)
    
    outputs = p_fit[3] + p_fit[2]*inputs + p_fit[1]*inputs**2 + p_fit[0]*inputs**3

    # Create the CasADi function
    f = cs.Function(name, [inputs], [outputs])
    return f

def loadTorchModelImplicit(name, nn_model):
    # Get the number of input and output layers from the model
    n_inputs = nn_model[0].in_features
    n_outputs = nn_model[-1].out_features
    
    # Define the CasADi variables for the inputs and outputs
    inputs = cs.MX.sym("inputs", 1, n_inputs)
    outputs = cs.MX.sym("outputs", 1, n_outputs)
    
    # Define the hidden layers using the weights and biases from the scikit-learn model
    hidden = inputs
    wts = []
    bss = []
    for i,layer in enumerate(nn_model):
        if isinstance(layer, torch.nn.Linear):
            weights = cs.MX.sym("weights_"+str(i), layer.in_features, layer.out_features)
            wts.append(weights)
            biases = cs.MX.sym("biases_"+str(i), 1, layer.out_features)
            bss.append(biases)
            hidden = cs.mtimes(hidden, weights) + biases
        if isinstance(layer,torch.nn.ReLU) :
            hidden = cs.if_else(hidden>0.,hidden,0.)
        if isinstance(layer,torch.nn.Tanh) :
            hidden = cs.tanh(hidden)
    
    inputs = [inputs]
    outputs = hidden
    inputs += wts
    inputs += bss
    f = cs.Function(name, inputs, [outputs])
    return f
