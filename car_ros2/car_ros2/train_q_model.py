import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

data = np.load('dataset.pkl', allow_pickle=True)
data = np.concatenate((np.expand_dims(data[0],axis=0),np.array(data[1:])[:,:-1,:]),axis = 0)
print(data.shape)

discount_factor = 0.98
n_iters = 1000
learning_rate = 0.001

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = []
        curr_h = input_size
        for h in hidden_size:
            self.fc.append(nn.Linear(curr_h, h))
            self.fc.append(nn.ReLU())
            curr_h = h
        self.fc.append(nn.Linear(curr_h, output_size))
        self.fc = nn.Sequential(*self.fc)
        
    def forward(self, x):        
        out = self.fc(x)
        return out

model = SimpleModel(21,[50,50],1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


dist_opp = np.sqrt((data[:,:-1,1]-data[:,:-1,0])**2 + (data[:,:-1,3]-data[:,:-1,2])**2)
Y = data[:,1:,0] - data[:,:-1,0] #- 0.5*(np.abs(data[:,:-1,2])>0.5)*(np.abs(data[:,:-1,2])-0.5) - 0.3*(dist_opp<0.25)
X = data[:,:,[1,2,3,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
X[:,:,0] -= data[:,:,0]
X[:,:,2] -= data[:,:,2]
X[:,:,0] = (X[:,:,0]>75.)*(X[:,:,0]-150.087) + (X[:,:,0]<-75.)*(X[:,:,0]+150.087) + (X[:,:,0]<=75.)*(X[:,:,0]>=-75.)*X[:,:,0]
Y = (Y>75.)*(Y-150.087) + (Y<-75.)*(Y+150.087) + (Y<=75.)*(Y>=-75.)*Y

# print(Y)
# print(np.argmax(Y))
# print(np.argmin(Y))
# exit(0)
Y = torch.tensor(Y).float()
mask = torch.tensor((Y<0.6)*(Y>=0.))
X = torch.tensor(X).float()
X[torch.isnan(X)] = 0.
# print(mask.sum()/(Y.shape[0]*Y.shape[1]))
# exit(0)
for i in range(n_iters) :
    total_loss = 0.
    for j in range(X.shape[0]) :
        # Calculate loss
        preds = model(torch.tensor(X[j,:-1]).float()) - discount_factor*model(torch.tensor(X[j,1:]).float())
        # print(preds.shape, Y[j].shape, mask[j].shape)
        # mask[j]*=(torch.isnan(preds.squeeze()))
        loss = loss_fn(preds.squeeze()*mask[j], torch.tensor(torch.tensor(Y[j])*mask[j]).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        total_loss += loss.item()
    print("Iteration: ", i, " Loss: ", total_loss)

torch.save(model.state_dict(), 'model.pth')
    


