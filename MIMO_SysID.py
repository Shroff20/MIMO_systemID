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

def print_header(x, lvl = 1):
    
    print('\n'*2)
    print('='*100)
    print(x.upper())
    print('='*100)


class FakeDataMaker():
      
    def make_system(N_inputs, N_outputs):
        
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
                
        print(f' * done making transfer function matrix: (outputs, intputs) = {G.shape}')
         
        return G        
    
    def make_inputs(N_inputs, N_loadcases):
        
        t = np.linspace(0, 100, 1000)
        Nt = len(t)
        U = np.zeros((N_loadcases, Nt, N_inputs))
        
        for p in range(N_loadcases):
            for k in range(2):
                for j in range(N_inputs):
                    A = np.random.rand(3)-.5
                    tc  = np.random.rand(3)*t[-1]
                    wn = np.random.rand(3)
                    U[p,:, j] += A[0]*np.heaviside(t-tc[0], 0)
                    U[p,:, j] += A[1]*np.heaviside(t-tc[1], 0)*np.sin(wn[1]*(t-tc[1]))*np.exp(-.2*(t-tc[1]))
        
        print(f' * done making inputs: (loadcases, timesteps, features) = {U.shape}')
        
        return U, t
    
    
    def compute_real_output(G, U, t):
        
        Nt = len(t)
        N_outputs = G.shape[0]
        N_inputs = G.shape[1]
        N_loadcases = U.shape[0]
        
        Y = np.zeros((N_loadcases, Nt, N_outputs))
        for p in range(N_loadcases):
            for i in range(N_outputs):
                for j in range(N_inputs):
                    t, dY, _ = signal.lsim(G[i, j], U[p, :, j], t)
                    Y[p, :, i] += dY
        print(f' * done computing actual output: (loadcases, timesteps, features) = {Y.shape}')
                
        return Y
    
    
    def generate_fake_data(N_inputs, N_outputs, N_loadcases):
        
        print_header('generating data')
        G = FakeDataMaker.make_system(N_inputs, N_outputs)
        U, t = FakeDataMaker.make_inputs(N_inputs, N_loadcases)
        Y = FakeDataMaker.compute_real_output(G, U, t)
        
        return G, U, Y, t
    
    pass


class NeuralNetworkTimeSeries():
    
    def __init__(self):
        self.device = None
        self.X_train = None
        self.X_test= None
        self.Y_train = None
        self.Y_test = None
        self.X_normalization = None
        self.Y_normalization = None
        self.autoencodersX = {}
        self.autoencodersY = {}
        self.X_train_encoded = None
        self.X_test_encoded = None
        self.Y_train_encoded = None
        self.Y_test_encoded = None
        
        
        print_header('initialization')
        
        # set cpu or gpu as device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(' * cuda avalible, using GPU')
        else:
            device = torch.device("cpu")
            print(' * cuda not avalible, using CPU (warning: will result in slow training)')
        self.device = device
        
        pass
    
    
    def load_data(self, inputs, outputs):
        
        
        print_header('load data')
        
        device = self.device
        
        X = torch.from_numpy(inputs).float().to(device)
        Y = torch.from_numpy(outputs).float().to(device)
        
        X, X_normalization = NeuralNetworkTimeSeries._normalize(X)
        Y, Y_normalization = NeuralNetworkTimeSeries._normalize(Y)
        
        X_train, X_test = NeuralNetworkTimeSeries._train_test_split(X, test_frac = .3)
        Y_train, Y_test = NeuralNetworkTimeSeries._train_test_split(Y, test_frac = .3)

        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        
        
        self.X_normalization = X_normalization
        self.Y_normalization = Y_normalization        
        
        
        print(f' * loaded inputs: {inputs.shape}')
        print(f' * loaded outputs: {outputs.shape}')
        
       
        
        
        #U = normalize(U)
        #Y = normalize(Y)
        
        
        pass
    


    def make_prediction():
        
        pass
    
    def _normalize(X):
        
        data_min = X.min()
        data_max = X.max()
        
        X = (X-data_min)/(data_max-data_min)
        
        return X, (float(data_min), float(data_max))


    def __str__(self):
        
        NeuralNetworkTimeSeries._dict_print(vars(self), name = type(self))
        
        return ''
    
    def _dict_print(d, name = ''):
        print('\n')
        print(name)
        for key, val in d.items():
            try:
                print(f' * {key}: {val.shape}')
            except:
                print(f' * {key}: {val}')
   
    
    def _train_test_split(X, test_frac = .3):
        
        N_test = int(np.ceil(X.shape[0]*test_frac))
        X_test = X[:N_test, :, :]
        X_train = X[N_test:, :, :]
        
        return X_train, X_test     

    


    def _train_autoencoder(device, inputs, compressed_dim, N_epochs, N_layers_autoencoder, learn_rate = .01, verbose = True):
        
        N_features = inputs.shape[-1]       
        
        
        model = AutoEncoder(input_dim = N_features, compressed_dim = compressed_dim, n_layers = N_layers_autoencoder)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        
        # if inputs.shape[1] !=1: 
        #     inputs = torch.reshape(inputs, (-1, 1, inputs.shape[2]))
        
        model.to(device)
        inputs = inputs.to(device)
        model.train()
        loss_vec = np.empty(N_epochs)
                
        for epoch in range(N_epochs):
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, inputs) 
            loss.backward(retain_graph=True)
            optimizer.step()  
            loss_val = loss.item()
            loss_vec[epoch] = loss_val
            if epoch%10 == 0 and verbose:
                print(f' * epoch {epoch}: loss = {loss_val}')
            
        model.losses = loss_vec
        
        return model


    def autoencoder_sweep(self, X_or_Y, N_trial_dims, N_epochs, N_layers_autoencoder):
        
        print_header(f'autoencoder sweep for {X_or_Y}')
        
        device = self.device
        
        if X_or_Y == 'X':
            inputs = self.X_train
        elif X_or_Y == 'Y':
            inputs = self.Y_train    
        else:
            raise(Exception('must be X or Y'))
        
        print(f' * training {N_layers_autoencoder} layer autoencoder for {N_epochs} epochs')
        
        for compressed_dim in N_trial_dims:
            autoencoder_model  = NeuralNetworkTimeSeries._train_autoencoder(device, inputs, compressed_dim, N_epochs, N_layers_autoencoder, verbose = False)           
            print(f' * {compressed_dim} dimensions: loss = {autoencoder_model.losses[-1]:.4e}')

            if X_or_Y == 'X':
                self.autoencodersX[compressed_dim]  = autoencoder_model
            elif X_or_Y == 'Y':
                self.autoencodersY[compressed_dim]  = autoencoder_model
            else:
                raise(Exception('must be X or Y'))
            
                    

    def reduce_dimensionality(self, X_or_Y, N):
        
        print_header(f'Reduce {X_or_Y} dimensionality')
        
        if X_or_Y == 'X':
            model = self.autoencodersX[N]
            self.X_train_encoded = model.encoder(self.X_train)
            self.X_test_encoded = model.encoder(self.X_test)
            
        elif X_or_Y == 'Y':
            model = self.autoencodersY[N]
            self.Y_train_encoded = model.encoder(self.Y_train)
            self.Y_test_encoded = model.encoder(self.Y_test)        
        else:
            raise(Exception('must be X or Y'))
        
        
        
        pass 
        


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, compressed_dim, n_layers = 1):
        super().__init__()
        
        layer_dims = np.ceil(np.exp(np.linspace(np.log(input_dim), np.log(compressed_dim), n_layers+1))).astype(int)
        
        encodelist = []
        decodelist = []
        for i in range(len(layer_dims)-1):
            # if i !=0:  # put relu between layers
            #     encodelist.append(nn.Sigmoid())
            #     decodelist.append(nn.Sigmoid())
            encodelist.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            decodelist.append(nn.Linear(layer_dims[-i-1], layer_dims[-i-2]))
            
        #print(encodelist)
        #print(decodelist)
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.fc_encode =   nn.Sequential(*encodelist)
        self.fc_decode =   nn.Sequential(*decodelist)    
        self.losses = []
        self.max_error = None
        self.mean_error = None
        
        #print(self.layer_dims)
        
    def encoder(self, x):
        x = self.fc_encode(x)
        return x
    
    def decoder(self, x):
        x = self.fc_decode(x)
        return x        
        
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
# dataset
N_inputs = 2
N_outputs = 5
N_loadcases = 20

#autoencoder
N_epochs_autoencoderX = 50
trial_dims_autoencoderX = range(1, N_inputs+1)
N_layers_autoencoderX = 1

N_epochs_autoencoderY = 50
trial_dims_autoencoderY = range(1, N_outputs+1)
N_layers_autoencoderY = 1


N_dim_X_autoencoder = 2
N_dim_Y_autoencoder = 4






G, inputs, outputs, t = FakeDataMaker.generate_fake_data(N_inputs, N_outputs, N_loadcases)



NNTS = NeuralNetworkTimeSeries()
NNTS.load_data(inputs, outputs)


NNTS.autoencoder_sweep('X', trial_dims_autoencoderX, N_epochs_autoencoderX, N_layers_autoencoderX)
NNTS.autoencoder_sweep('Y', trial_dims_autoencoderY, N_epochs_autoencoderY, N_layers_autoencoderY)

NNTS. reduce_dimensionality('X', N_dim_X_autoencoder)
NNTS. reduce_dimensionality('Y', N_dim_Y_autoencoder)

print_header('done')


print(NNTS)

    
#%%
import sqlite3



#%%
   

    



# class GRUNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
#         super(GRUNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
        
#         self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.losses = []
#         self.max_error = None
#         self.mean_error = None
#         #self.relu = nn.ReLU()
        
#     def forward(self, x):
#         h = self.init_hidden(x.shape[0])
#         h = h.data
#         out, h = self.gru(x, h)
#         #out = self.fc(self.relu(out[:,-1]))
#         out = self.fc(out)
#         #out = self.relu(out)
#         return out, h
    
#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
#         return hidden


# def normalize(X):
#     return .5*X/(X.abs().max())






# def autoencoder_error_analysis(Y, autoencoder_model_Y, plot = True):
#     Y_encoded = autoencoder_model_Y.encoder(Y)
#     Y_decoded = autoencoder_model_Y.decoder(Y_encoded)
#     print(f' * original shape: {Y.shape}')
#     print(f' * encoded shape: {Y_encoded.shape}')
#     print(f' * decoded shape: {Y_decoded.shape}')
    
#     encoding_error = Y - Y_decoded
#     max_error = torch.max(torch.abs(encoding_error))
#     mean_error = torch.mean(torch.abs(encoding_error))
    
#     print(f' * max_error: {max_error}')
#     print(f' * mean_error: {mean_error}')
    
#     if plot:
#         fig, ax = plt.subplots()
#         for k in range(N_loadcases):
#             ax.plot(encoding_error[k, :, :].cpu().detach().numpy())
            
#     return max_error, mean_error

# def train_autoencoder(inputs, compressed_dim, N_epochs, N_layers_autoencoder, learn_rate = .01, verbose = True):
    
#     N_features = inputs.shape[-1]
    
#     model = AutoEncoder(input_dim = N_features, compressed_dim = compressed_dim, n_layers = N_layers_autoencoder)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
#     # if inputs.shape[1] !=1: 
#     #     inputs = torch.reshape(inputs, (-1, 1, inputs.shape[2]))
    
#     model.to(device)
#     inputs = inputs.to(device)
#     model.train()
#     loss_vec = np.empty(N_epochs)
    
#     for epoch in range(N_epochs):
#         optimizer.zero_grad()
#         out = model(inputs)
#         loss = criterion(out, inputs) 
#         loss.backward(retain_graph=True)
#         optimizer.step()  
#         loss_val = loss.item()
#         loss_vec[epoch] = loss_val
#         if epoch%10 == 0 & verbose == True:
#             print(f' * epoch {epoch}: loss = {loss_val}')
    
    
#     max_error, mean_error = autoencoder_error_analysis(inputs, model, plot = False)
    
#     model.loss_final = float(loss_val)
#     model.max_error = float(max_error)
#     model.mean_error = float(mean_error)   
    
#     return model


# def plot_autoencoder_sensitivity(all_models):
    
#     fig, ax = plt.subplots(3, 1)
#     ax[0].plot(all_models.keys(), [x.loss_final for x in all_models.values()])
#     ax[0].set_title('loss')
#     ax[0].set_xlabel('latent dimension')

#     ax[1].plot(all_models.keys(), [x.max_error for x in all_models.values()])
#     ax[1].set_title('max error')
#     ax[1].set_xlabel('latent dimension')

#     ax[2].plot(all_models.keys(), [x.mean_error for x in all_models.values()])
#     ax[2].set_title('mean error')
#     ax[2].set_xlabel('latent dimension')

#     fig.tight_layout()
#     fig.set_dpi(200)


# def autoencoder_sweep(X, N_trial_dims, N_epochs, N_layers_autoencoder = 1):
    
#     all_models = {}
#     for compressed_dim in N_trial_dims:
        
#         print(f'N = {compressed_dim} latent dimension autoencoder')
#         autoencoder_model  = train_autoencoder(X, compressed_dim, N_epochs, N_layers_autoencoder, verbose = False)
#         print(f' * loss = {autoencoder_model.loss_final}')
#         print('\n')
        
#         all_models[compressed_dim] = autoencoder_model
        
#     plot_autoencoder_sensitivity(all_models)
    
#     return all_models


# def train_RNN(inputs,targets, N_hidden_dim, N_layers, N_epochs, learn_rate = .01, verbose = True, jaggedness_penalty = 0):
    
#     N_inputs = inputs.shape[-1]
#     N_outputs = targets.shape[-1]
    
#     model = GRUNet(input_dim = N_inputs, hidden_dim = N_hidden_dim, output_dim = N_outputs, n_layers = N_layers)
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
#     model.train()
    
#     inputs = inputs.to(device)
#     targets = targets.to(device)
    
#     for epoch in range(N_epochs):
#         optimizer.zero_grad()
#         out, _ = model(inputs)
        
#         if jaggedness_penalty>0:
#             penalty =jaggedness_penalty* (out-targets).diff(axis = 1).abs().sum()
#         else:
#             penalty = 0
        
#         loss = criterion(out, targets) + penalty
#         loss.backward(retain_graph=True)
#         optimizer.step()  
#         loss_val = loss.item()
        
#         model.losses.append(loss_val)
        
#         if epoch%10 == 0 and verbose == True:
#             print(f' * epoch {epoch}: loss = {loss_val}, penalty = {100*penalty/loss_val :2f}%')
    
#     return model



# #%% INPUTS

# # dataset
# N_inputs = 2
# N_outputs = 5
# N_loadcases = 10

# #autoencoder
# N_epochs_autoencoder = 500
# N_trial_dims = [1, 4, 7, 20] # range(1, N_outputs+1)
# N_layers_autoencoder = 1
# N_latent_dim_input = 4
# N_latent_dim_output = 7

# # RNN
# N_epochs = 500
# N_layers = 1
# N_hidden_dim = 100
# jaggedness_penalty = 0#1e-3


# #%%


# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('using GPU')

# else:
#     device = torch.device("cpu")
#     print('using CPU')
    





# U = torch.from_numpy(U).float().to(device)
# Y = torch.from_numpy(Y).float().to(device)

# U = normalize(U)
# Y = normalize(Y)


# U_train, U_test =  train_test_split(U)
# Y_train, Y_test =  train_test_split(Y)

# U_train = U_train.to(device)
# U_test = U_test.to(device)
# Y_train = Y_train.to(device)
# Y_test = Y_test.to(device)

# all_autoencoder_models_Y = autoencoder_sweep(Y_train, N_trial_dims, N_epochs_autoencoder, N_layers_autoencoder)
# all_autoencoder_models_U = autoencoder_sweep(U_train, N_trial_dims, N_epochs_autoencoder, N_layers_autoencoder)


# autoencoder_model_Y = all_autoencoder_models_Y[N_latent_dim_output]
# autoencoder_model_U = all_autoencoder_models_U[N_latent_dim_input]


# #%%

# Y_train_encoded = autoencoder_model_Y.encoder(Y_train)
# Y_test_encoded = autoencoder_model_Y.encoder(Y_test)

# U_train_encoded = autoencoder_model_U.encoder(U_train)
# U_test_encoded = autoencoder_model_U.encoder(U_test)


# print(f'train inputs: {U_train_encoded.shape}, {type(U_train_encoded)}')
# print(f'train targets: {Y_train_encoded.shape}, {type(Y_train_encoded)}')
# print(f'test inputs: {U_test_encoded.shape}, {type(U_test_encoded)}')
# print(f'test targets: {Y_test_encoded.shape}, {type(Y_test_encoded)}')
      


# RNNmodel = train_RNN(inputs = U_train_encoded, targets = Y_train_encoded, N_hidden_dim = N_hidden_dim, N_layers = N_layers, N_epochs = N_epochs)
    
# fig, ax = plt.subplots(1,1)
# ax.plot(RNNmodel.losses)
# plt.yscale('log')
# fig.set_dpi(200)    


# #%% Testing
    
# Y_pred_encoded, _ = RNNmodel(U_test_encoded)    

# Y_pred = autoencoder_model_Y.decoder(Y_pred_encoded)


# for p in range(np.min([5, U_test.shape[0]])):
#     fig, ax = plt.subplots(3, 1)
#     fig.set_dpi(200)
#     ax[0].plot(U_test[p, :, :].cpu().detach().numpy())
#     ax[0].set_title(f'inputs ({U_test.shape[2]})')
    
#     ax[1].plot(Y_test[p, :, :].cpu().detach().numpy())
#     ax[1].plot(Y_pred[p, :, :].cpu().detach().numpy(), linestyle = '--', color = 'k', linewidth = .5)
#     ax[1].set_title(f'outputs ({Y_test.shape[2]})')

#     error = Y_pred.cpu().detach().numpy() - Y_test.cpu().detach().numpy()
#     ax[2].plot(error[p, :, :])
#     ax[2].set_title(f'error ({Y_test.shape[2]})')
    
#     fig.tight_layout()



# #%%

# model =  torch.nn.Sequential(
#             torch.nn.Linear(10, 2),
#             torch.nn.ReLU(),
#             )
    