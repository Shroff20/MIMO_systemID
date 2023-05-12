# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:02:36 2023

@author: ssmee
"""

import numpy as np
import matplotlib.pyplot as plt
import torch 
import scipy.signal as signal
import torch.nn as nn


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


#%%


def make_data(N_inputs, N_outputs, N_loadcases):

    np.random.seed(0)
    
    G = np.empty((N_outputs, N_inputs), dtype = object)
    for i in range(N_outputs):
        for j in range(N_inputs):
            wn = 1 + np.random.rand()
            z = 2*np.random.rand()+.5
            if i>=j:
                A = np.random.rand()-.5
            else:
                A = 1e-9
            G[i, j] = signal.TransferFunction([A*wn**2], [1, 2*z*wn, wn**2])
            
            
            
    t = np.linspace(0, 100, 1000)
    Nt = len(t)
    
    U = np.zeros((N_loadcases, Nt, N_inputs))
    Y = np.zeros((N_loadcases, Nt, N_outputs))
    
    for p in range(N_loadcases):
        for k in range(2):
            for j in range(N_inputs):
                A = np.random.rand(3)-.5
                tc  = np.random.rand(3)*t[-1]
                wn = np.random.rand(3)
                U[p,:, j] += A[0]*np.heaviside(t-tc[0], 0)
                U[p,:, j] += A[1]*np.heaviside(t-tc[1], 0)*np.sin(wn[1]*(t-tc[1]))*np.exp(-.2*(t-tc[1]))
    
    for p in range(N_loadcases):
        for i in range(N_outputs):
            for j in range(N_inputs):
                t, dY, _ = signal.lsim(G[i, j], U[p, :, j], t)
                
                Y[p, :, i] += dY
            
    U = torch.from_numpy(U).float()
    Y = torch.from_numpy(Y).float()
            
            
    return U, Y
    

def train_test_split(X, test_frac = .3):
    
    N_test = int(np.ceil(X.shape[0]*test_frac))
    
    X_test = X[:N_test, :, :]
    X_train = X[N_test:, :, :]
    
    return X_train, X_test
    



class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.relu = nn.ReLU()
        
    def forward(self, x):
        
        h = self.init_hidden(x.shape[0])
        h = h.data
        out, h = self.gru(x, h)
        #out = self.fc(self.relu(out[:,-1]))
        out = self.fc(out)
        #out = self.relu(out)
        
        
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


def normalize(X):
    return .5*X/(X.abs().max())






#%%
N_inputs = 4
N_outputs = 10
N_loadcases = 200

N_epochs = 2000
N_layers = 1
N_hidden_dim = 100
learn_rate = .01
jaggedness_penalty = 0#1e-3



if torch.cuda.is_available():
    device = torch.device("cuda")
    print('using GPU')

else:
    device = torch.device("cpu")
    print('using CPU')
    


U, Y =  make_data(N_inputs, N_outputs, N_loadcases)


U = normalize(U)
Y = normalize(Y)

U_train, U_test =  train_test_split(U)
Y_train, Y_test =  train_test_split(Y)

U_train = U_train.to(device)
U_test = U_test.to(device)
Y_train = Y_train.to(device)
Y_test = Y_test.to(device)



print(f'train inputs: {U_train.shape}, {type(U_train)}')
print(f'train targets: {Y_train.shape}, {type(Y_train)}')
print(f'test inputs: {U_test.shape}, {type(U_test)}')
print(f'test targets: {Y_test.shape}, {type(Y_test)}')
      


model = GRUNet(input_dim = N_inputs, hidden_dim = N_hidden_dim, output_dim = N_outputs, n_layers = N_layers)
model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
model.train()



loss_vec = np.empty(N_epochs)


for epoch in range(N_epochs):
   
    optimizer.zero_grad()
    
    out, _ = model(U_train.to(device))
    
    if jaggedness_penalty>0:
        penalty =jaggedness_penalty* (out-Y_train.to(device)).diff(axis = 1).abs().sum()
    else:
        penalty = 0
    
    
    loss = criterion(out, Y_train) + penalty
    loss.backward()
    optimizer.step()  
    
    loss_val = loss.item()
    
    loss_vec[epoch] = loss_val
    
    if epoch%10 == 0:
    
        print(f'epoch {epoch}: loss = {loss_val}, penalty = {100*penalty/loss_val :2f}%')
    
    
#%%    
    
Y_pred, _ = model(U_test.to(device))    
loss = criterion(Y_pred, Y_test)  
    
print(f'test data: loss = {loss.item()}')


for p in range(5):
    fig, ax = plt.subplots(3, 1)
    fig.set_dpi(200)
    ax[0].plot(U_test[p, :, :].cpu().detach().numpy())
    ax[0].set_title(f'inputs ({U_test.shape[2]})')
    
    ax[1].plot(Y_test[p, :, :].cpu().detach().numpy())
    ax[1].plot(Y_pred[p, :, :].cpu().detach().numpy(), linestyle = '--', color = 'k', linewidth = .5)
    ax[1].set_title(f'outputs ({Y_test.shape[2]})')

    
    ax[2].plot(Y_pred[p, :, :].cpu().detach().numpy() - Y_test[p, :, :].cpu().detach().numpy())
    ax[2].set_title(f'error ({Y_test.shape[2]})')
    
    fig.tight_layout()

fig, ax = plt.subplots(1,1)
ax.plot(loss_vec)
plt.yscale('log')
fig.set_dpi(200)
#%%


    