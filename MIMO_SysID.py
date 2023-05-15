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
import time

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
    
    def __init__(self, working_dir):
        self.device = None
        self.X_train = None
        self.X_test= None
        self.Y_train = None
        self.Y_test = None
        self.X_normalization = None
        self.Y_normalization = None
        self.autoencodersX = {}
        self.autoencodersY = {}
        self.autoencoderX = None
        self.autoencoderY = None          
        self.X_train_encoded = None
        self.X_test_encoded = None
        self.Y_train_encoded = None
        self.Y_test_encoded = None
        self.model = None
        self.working_dir = working_dir
        
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
        
        
        f_normalizeX, f_unnormalizeX, norm_limsX =  NeuralNetworkTimeSeries._create_normalization(X)
        f_normalizeY, f_unnormalizeY, norm_limsY =  NeuralNetworkTimeSeries._create_normalization(Y)

        X_train_raw, X_test_raw = NeuralNetworkTimeSeries._train_test_split(X, test_frac = .3)
        Y_train_raw, Y_test_raw = NeuralNetworkTimeSeries._train_test_split(Y, test_frac = .3)

        

        
        X_train = f_normalizeX(X_train_raw)
        X_test = f_normalizeX(X_test_raw)
        Y_train = f_normalizeY(Y_train_raw)
        Y_test = f_normalizeY(Y_test_raw)
        
        
        
        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw
        self.Y_train_raw = Y_train_raw
        self.Y_test_raw = Y_test_raw       
        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        
        self.f_normalizeX = f_normalizeX
        self.f_normalizeY = f_normalizeY        
        self.f_unnormalizeX = f_unnormalizeX
        self.f_unnormalizeY = f_unnormalizeY       
        self.norm_limsX = norm_limsX           
        self.norm_limsY = norm_limsY            
        
        print(f' * loaded inputs: {inputs.shape}')
        print(f' * loaded outputs: {outputs.shape}')
        
    
   
    
    

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

    


    def _train_autoencoder(device, input_train, input_test, compressed_dim, N_epochs, N_layers_autoencoder, learn_rate = .01, verbose = True):
        
        N_features = input_train.shape[-1]       
        model = AutoEncoder(input_dim = N_features, compressed_dim = compressed_dim, n_layers = N_layers_autoencoder)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        
        # if inputs.shape[1] !=1: 
        #     inputs = torch.reshape(inputs, (-1, 1, inputs.shape[2]))
        
        model.to(device)
        input_train = input_train.to(device)
        input_test = input_test.to(device)
        
        model.train()      
        
        t0 = time.perf_counter()
        
        for epoch in range(N_epochs):
            optimizer.zero_grad()
            
            out_train = model(input_train)
            loss_train = criterion(out_train, input_train) 
            loss_train.backward(retain_graph=True)
            loss_train_val = loss_train.item()
            optimizer.step()  
            
            out_test = model(input_test)
            loss_test = criterion(out_test, input_test)          
            loss_test_val = loss_test.item()
            
            model.losses_train.append(float(loss_train_val))
            model.losses_test.append(float(loss_test_val))
                        
            t1 = time.perf_counter()
            dt = t1-t0

            if (dt>=1.0 and verbose) or (epoch == 0) or (epoch == N_epochs-1):
                print(f'      * epoch {epoch: >6}: train loss = {loss_train_val : .4e}, test loss = {loss_test_val : .4e}')
                t0 = t1
                    
        return model


    def autoencoder_sweep(self, X_or_Y, N_trial_dims, N_epochs, N_layers_autoencoder):
        
        print_header(f'autoencoder sweep for {X_or_Y}')
        
        device = self.device
        
        if X_or_Y == 'X':
            input_train = self.X_train
            input_test = self.X_test
        elif X_or_Y == 'Y':
            input_train = self.Y_train
            input_test = self.Y_test 
        else:
            raise(Exception('must be X or Y'))
        
        for compressed_dim in N_trial_dims:
            
            print(f' * training autoencoder with {compressed_dim} compressed dimensions and {N_layers_autoencoder} layers for {N_epochs} epochs')

            autoencoder_model  = NeuralNetworkTimeSeries._train_autoencoder(device, input_train, input_test, compressed_dim, N_epochs, N_layers_autoencoder)           
            #print(f' * {compressed_dim} dimensions: loss = {autoencoder_model.losses[-1]:.4e}')

            if X_or_Y == 'X':
                self.autoencodersX[compressed_dim]  = autoencoder_model
            elif X_or_Y == 'Y':
                self.autoencodersY[compressed_dim]  = autoencoder_model
            else:
                raise(Exception('must be X or Y'))
        
        if X_or_Y == 'X':
            NeuralNetworkTimeSeries._plot_autoencoder_sweep_loss(self.autoencodersX, 'X', output_folder = self.working_dir)
        elif X_or_Y == 'Y':
            NeuralNetworkTimeSeries._plot_autoencoder_sweep_loss(self.autoencodersY, 'Y', output_folder = self.working_dir)
        else:
            raise(Exception('must be X or Y'))    
        

    def reduce_dimensionality(self, X_or_Y, N):
        
        print_header(f'Reduce {X_or_Y} dimensionality')
        if X_or_Y == 'X':
            model = self.autoencodersX[N]
            self.autoencoderX = model
            self.X_train_encoded = model.encoder(self.X_train)
            self.X_test_encoded = model.encoder(self.X_test)
            print(f' * X training:   {self.X_train.shape} -> {self.X_train_encoded.shape}')
            print(f' * X testing:   {self.X_test.shape} -> {self.X_test_encoded.shape}')       
            
        elif X_or_Y == 'Y':
            model = self.autoencodersY[N]
            self.autoencoderY = model
            self.Y_train_encoded = model.encoder(self.Y_train)
            self.Y_test_encoded = model.encoder(self.Y_test)        
            print(f' * Y training:   {self.Y_train.shape} -> {self.Y_train_encoded.shape}')
            print(f' * Y testing:   {self.Y_test.shape} -> {self.Y_test_encoded.shape}')

        else:
            raise(Exception('must be X or Y'))

        
    def train(self, N_hidden_dim, N_layers, N_epochs, learn_rate = .01, verbose = True, gradiant_clip = False):
        
        print_header('train RNN')
        
        N_inputs = self.X_train_encoded.shape[-1]
        N_outputs =  self.Y_train_encoded.shape[-1]
                
        model = GRUNet(input_dim = N_inputs, hidden_dim = N_hidden_dim, output_dim = N_outputs, n_layers = N_layers, device = self.device,)
        

        
        model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        model.train()
        
        t0 = time.perf_counter()
        
        
        
        
        for epoch in range(N_epochs):
            optimizer.zero_grad()
            out_train, _ = model(self.X_train_encoded)
            
            
            loss_train = criterion(out_train, self.Y_train_encoded) 
            loss_train.backward(retain_graph=True)
            
            if gradiant_clip == True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            optimizer.step()  
            loss_train_val = loss_train.item()
            
            out_test, _ = model(self.X_test_encoded)
            loss_test = criterion(out_test, self.Y_test_encoded) 
            loss_test_val = loss_test.item()
            
            model.losses_train.append(float(loss_train_val))
            model.losses_test.append(float(loss_test_val))
            
            t1 = time.perf_counter()
            dt = t1 -t0
            
            if (dt >= 1 and verbose == True) or (epoch == N_epochs-1) or (epoch == 0):
                print(f' * epoch {epoch: >6}: train loss = {loss_train_val : .4e}, test loss = {loss_test_val : .4e}')
                t0 = t1
        
        
        model.plot_losses(self.working_dir)

        self.model = model
    
    
    
    def _error_plot(U, Y_actual, Y_predicted, output_folder = '.'):
        
       error = Y_actual-Y_predicted
        
       
       for p in range(np.min([5, error.shape[0]])):
           fig, ax = plt.subplots(3, 1)
           fig.set_dpi(200)
           ax[0].plot(U[p, :, :].cpu().detach().numpy())
           ax[0].set_title(f'inputs ({U.shape[-1]})')
           
           ax[1].plot(Y_actual[p, :, :].cpu().detach().numpy())
           ax[1].plot(Y_predicted[p, :, :].cpu().detach().numpy(), linestyle = '--', color = 'k', linewidth = .5)
           ax[1].set_title(f'outputs ({Y_predicted.shape[-1]})')

           ax[2].plot(error[p, :, :].cpu().detach().numpy())
           ax[2].set_title(f'error ({error.shape[-1]})')
           
           fig.suptitle(f'test case {p}')
           
           fig.tight_layout()
           
           fn = os.path.join(output_folder, f'signal_plot_test_{p}.pdf')
           fig.savefig(fn)
           
           print(f' * saved {fn}')
    
       
       fig, ax = plt.subplots(2, 1)
       ax[0].scatter(Y_actual.cpu().detach().numpy().ravel(), Y_predicted.cpu().detach().numpy().ravel())
       ax[0].scatter(Y_actual.cpu().detach().numpy().ravel(), Y_predicted.cpu().detach().numpy().ravel())
       ax[0].set_xlabel('actual values')
       ax[0].set_ylabel('predicted values)')    
       ax[0].grid(c = [.9, .9, .9])
       ax[0].set_axisbelow(True)
       ax[1].hist(error.cpu().detach().numpy().ravel())
       ax[1].set_title(f'error')
       ax[1].grid(c = [.9, .9, .9])
       ax[1].set_axisbelow(True)
       fig.set_dpi(200)
       fig.tight_layout()
       
       
          
       return None
        
   
    
   
    def assess_fit(self):
        
            print_header('assess fit on training data')
        
            X_test = self.X_test
            Y_test = self.Y_test
    
            X_test_encoded = self.autoencoderX.encoder(X_test)
            Y_pred_encoded, _ = self.model(X_test_encoded)
    
            Y_pred = self.autoencoderY.decoder(Y_pred_encoded)
    
            NeuralNetworkTimeSeries._error_plot(X_test, Y_test, Y_pred, output_folder = self.working_dir)

        
    def _create_normalization(X):
                
        Xmax = X.amax(dim = (0, 1), keepdim = True)
        Xmin = X.amin(dim = (0, 1), keepdim = True)
        
        f_normalize = lambda y:  2*(y-Xmin)/(Xmax-Xmin) - 1
        f_unnormalize = lambda y:  (y + 1)*(Xmax-Xmin)/2 + Xmin
    
        return f_normalize, f_unnormalize, (Xmin, Xmax)
        
    
    def predict(self, X):
        if type(X) == np.ndarray:
            X = torch.from_numpy(inputs).float().to(self.device)
        
        X = self.f_normalizeX(X)
        X = self.autoencoderX.encoder(X)
        X, _ = self.model(X)
        X = self.autoencoderY.decoder(X)
        X = self.f_unnormalizeY(X)
        
        return X     
    

    def wrapup(self):
        
        print_header('done')

    def _plot_autoencoder_sweep_loss(autoencoders, varname = '', output_folder = '.'):
    
        fig, ax = plt.subplots()
        fig.set_dpi(200)
        for key, data in autoencoders.items():
            
            h = ax.plot(autoencoders[key].losses_train, label = f'train, {key} dimensions')
            ax.plot(autoencoders[key].losses_test, linestyle = '--', label = f'test, {key} dimensions', color = h[0].get_color())
            
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")       
            ax.set_yscale('log')
            ax.set_xlabel ('epoch')
            ax.set_ylabel ('loss')
            ax.set_title (f'{varname} autoencoder losses')    
            ax.grid(c = [.9, .9, .9])
            ax.set_axisbelow(True)
        
        fig.tight_layout()
        fn = os.path.join(output_folder, f'losses_autoencoder_{varname}.pdf')
        fig.savefig(fn)
        print(f' * saved {fn}')
    
    
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.losses_train = []
        self.losses_test = []
        self.max_error = None
        self.mean_error = None
        self.device = device
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
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
    
    
    def plot_losses(self, output_folder = '.'):
        
        losses_train = self.losses_train
        losses_test = self.losses_test
        fig, ax = plt.subplots()
        fig.set_dpi(200)       
        ax.plot(losses_train, label = 'train loss')
        ax.plot(losses_test, label = 'test loss')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")       
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        ax.set_title('timeseries model loss')
        ax.grid(c = [.9, .9, .9])
        ax.set_axisbelow(True)
        fig.tight_layout()
        fn = os.path.join(output_folder, 'losses_timeseries_model.pdf')
        fig.savefig(fn)
        print(' * saved {fn}')

        return None
    
    

class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim, compressed_dim, n_layers = 1):
        super().__init__()
        layer_dims = np.ceil(np.exp(np.linspace(np.log(input_dim), np.log(compressed_dim), n_layers+1))).astype(int)
        layer_dims[0] = input_dim  # sometimes there are numerical issues with floats
        layer_dims[-1] = compressed_dim # sometimes there are numerical issues with floats

        
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
        self.losses_train = []
        self.losses_test = []
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


    def plot_losses(self):
        
        losses_train = self.losses_train
        losses_test = self.losses_test
        fig, ax = plt.subplots()
        fig.set_dpi(200)       
        ax.plot(losses_train, label = 'train loss')
        ax.plot(losses_test, label = 'test loss')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        ax.set_title('autoencoder model loss')

        return None
#%%


if __name__ == '__main__':

    
    working_dir = '.\output_folder'


    # dataset
    N_inputs = 3
    N_outputs = 5
    N_loadcases = 20
    
    #autoencoder
    N_epochs_autoencoderX = 50
    trial_dims_autoencoderX = range(1, N_inputs+1)
    N_layers_autoencoderX = 1
    
    N_epochs_autoencoderY = 50
    trial_dims_autoencoderY = range(1, N_outputs+1)
    N_layers_autoencoderY = 1
    
    N_dim_X_autoencoder = 3
    N_dim_Y_autoencoder = 5
    
    # RNN
    N_epochs_RNN = 100
    N_layers_RNN = 1
    N_hidden_dim_RNN = 100
    
    
    
    G, inputs, outputs, t = FakeDataMaker.generate_fake_data(N_inputs, N_outputs, N_loadcases)
    
    NNTS = NeuralNetworkTimeSeries(working_dir = working_dir)
    NNTS.load_data(inputs, outputs)
    
    NNTS.autoencoder_sweep('X', trial_dims_autoencoderX, N_epochs_autoencoderX, N_layers_autoencoderX)
    NNTS.autoencoder_sweep('Y', trial_dims_autoencoderY, N_epochs_autoencoderY, N_layers_autoencoderY)
    
    NNTS.reduce_dimensionality('X', N_dim_X_autoencoder)
    NNTS.reduce_dimensionality('Y', N_dim_Y_autoencoder)
    
    NNTS.train(N_hidden_dim_RNN, N_layers_RNN, N_epochs_RNN)
    NNTS.assess_fit()
    
    NNTS.wrapup()








