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
import pandas as pd
import pickle

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # fix install to be able to remove this
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
        
        np.random.seed(0)
        Y_offset = np.random.rand(5)[None, None, :]-.5
        Y = Y + Y_offset        
        
        
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
        self.autoencodersX = {}
        self.autoencodersY = {}
        self.autoencoderX = None
        self.autoencoderY = None          
        self.model = None
        self.working_dir = working_dir
        self.folders = {}
        self.normalization_data = None
        self.df_loadcases = pd.DataFrame(columns = ['name', 'fn_raw_data', 'fn_encoded_data'])
        self.train_test_indicies = None
        
        
        print_header('initialization')
        self._make_directory_system()

        # set cpu or gpu as device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(' * cuda avalible, using GPU')
        else:
            device = torch.device("cpu")
            print(' * cuda not avalible, using CPU (warning: will result in slow training)')
        self.device = device
        
        pass
    
    
    def _make_directory_system(self):
        
        working_dir = self.working_dir
        
        dirs = {}
        dirs['data_raw_dir'] = os.path.join(working_dir, 'data', 'raw_data')
        dirs['data_encoded_dir'] = os.path.join(working_dir, 'data', 'encoded_data')
        dirs['plots_losses_dir'] = os.path.join(working_dir, 'plots', 'losses')
        dirs['plots_signals_dir'] = os.path.join(working_dir, 'plots', 'signal_predictions')
        dirs['models_dir'] = os.path.join(working_dir, 'models')

        for path in dirs.values():
            os.makedirs(path, exist_ok = True)
        
        self.folders = dirs
        
    def add_loadcase_data(self, name, inputs, outputs, overwrite_existing = False):
        
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs).float()
        if type(outputs) == np.ndarray:
            outputs = torch.from_numpy(outputs).float()
        
        assert inputs.shape[0] == outputs.shape[0]
        
        def _get_min_max(x):
            
            min_vals, _ = torch.min(x, axis = 0)
            max_vals, _ = torch.max(x, axis = 0)
            
            return min_vals.cpu().detach().numpy(), max_vals.cpu().detach().numpy()
            
        input_min_vals, input_max_vals = _get_min_max(inputs)
        output_min_vals, output_max_vals = _get_min_max(outputs)
        
        fn = os.path.join(self.folders['data_raw_dir'],  f'{name}.pkl')
        
        if os.path.exists(fn) and (overwrite_existing == False):
            print(f' * {name} data already exists and will not be re-added')
        else:
            torch.save((inputs, outputs), fn)
        
        # update the maxes and mins seen over all loadcases
        if self.normalization_data == None:
            self.normalization_data = {}
            self.normalization_data['input_min_vals'] = input_min_vals
            self.normalization_data['input_max_vals'] = input_max_vals
            self.normalization_data['output_min_vals'] = output_min_vals
            self.normalization_data['output_max_vals'] = output_max_vals
        else:
            self.normalization_data['input_min_vals']  = np.min([self.normalization_data['input_min_vals'], input_min_vals], axis = 0)
            self.normalization_data['input_max_vals']  = np.max([self.normalization_data['input_max_vals'], input_max_vals], axis = 0)
            self.normalization_data['output_min_vals'] = np.min([self.normalization_data['output_min_vals'], output_min_vals], axis = 0)
            self.normalization_data['output_max_vals'] = np.max([self.normalization_data['output_max_vals'], output_max_vals], axis = 0)        
        
        data = {}
        data['name'] = name
        data['fn_raw_data'] = fn
        
        self.df_loadcases.loc[name, :] = data
        
        print(f' * {name}: {inputs.shape[1]} inputs, {outputs.shape[1]} outputs, {inputs.shape[0]} timesteps')
    
    
    
    def _normalize(self, data, X_or_Y, normalize_or_unnormalize):
        
        if X_or_Y == 'X':
            min_vals = self.normalization_data['input_min_vals'].reshape(1, 1, -1)
            max_vals = self.normalization_data['input_max_vals'].reshape(1, 1, -1)
        elif X_or_Y == 'Y':
            min_vals = self.normalization_data['output_min_vals'].reshape(1, 1, -1)
            max_vals = self.normalization_data['output_max_vals'].reshape(1, 1, -1)
        else: 
            raise Exception('must be X or Y')
        
        if type(min_vals) == np.ndarray:
            min_vals = torch.from_numpy(min_vals).float().to(self.device)
        if type(max_vals) == np.ndarray:
            max_vals = torch.from_numpy(max_vals).float().to(self.device)      
        
        # # -1 to 1 normalization and reverse normalization
        # if normalize_or_unnormalize == 'normalize':
        #     transformed_data = 2*(data-min_vals)/(max_vals-min_vals) - 1
        # elif normalize_or_unnormalize == 'unnormalize':
        #     transformed_data = (data + 1)*(max_vals-min_vals)/2 + min_vals
        # else:
        #     raise Exception('must be normalize or unnormalize')
        
        # 0 to 1 normalization and reverse normalization
        if normalize_or_unnormalize == 'normalize':
            transformed_data = (data-min_vals)/(max_vals-min_vals)
        elif normalize_or_unnormalize == 'unnormalize':
            transformed_data = data*(max_vals-min_vals) + min_vals
        else:
            raise Exception('must be normalize or unnormalize')
        
        
        return transformed_data
    

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

    


    def _train_autoencoder(device, train_test_indicies, f_load, compressed_dim, N_epochs, N_layers_autoencoder, N_loadcases, learn_rate = .01,  verbose = True, gradiant_clip = True):
        
        
        input_train = f_load(train_test_indicies['train'][0])    
        
        N_features = input_train.shape[-1]       
        model = AutoEncoder(input_dim = N_features, compressed_dim = compressed_dim, n_layers = N_layers_autoencoder)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        model.to(device)
        model.train()      
        
        t0 = time.perf_counter()
        
        
        N_times_through_batches = 5  # time
        N_batches = len(train_test_indicies['train'])
        N_loadcases_per_batch = np.average([len(x) for x in train_test_indicies['train']])
                
        epoch = 0
        total_loadcases_trained_on = 0
        
        for j in range(N_times_through_batches):        
            
            for train_batch in train_test_indicies['train']:
                
                input_train = f_load(train_batch)
            
                n_lim = np.ceil(N_epochs * N_loadcases/(N_times_through_batches * N_batches * N_loadcases_per_batch)).astype(int)
                for n in range(n_lim):
                    optimizer.zero_grad()
                    
                    out_train = model(input_train)
                    loss_train = criterion(out_train, input_train) 
                    loss_train.backward(retain_graph=True)
                    loss_train_val = loss_train.item()
                    
                                        
                    if gradiant_clip == True:
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=.01)
                    
                    
                    optimizer.step()  
                    
                    total_loadcases_trained_on += input_train.shape[0]
                                
                    t1 = time.perf_counter()
                    dt = t1-t0
                    
                    loss_test_val = 0
                    
                    epoch = total_loadcases_trained_on/N_loadcases

                    if (dt>=1.0 and verbose) or (epoch == 0) or (epoch == N_epochs-1):
                        print(f'      * epoch {epoch: >6}: train loss = {loss_train_val : .4e}')
                        t0 = t1
                        
                    model.losses_train.append((float(epoch), float(loss_train_val)))
      
                    
            test_losses_per_batch = []
            for test_batch in train_test_indicies['test']:   
                input_test = f_load(test_batch)
                out_test = model(input_test)
                loss_test = criterion(out_test, input_test)          
                test_losses_per_batch.append(loss_test.item())
            loss_test_val = np.average(test_losses_per_batch)
            model.losses_test.append((float(epoch), float(loss_test_val)))
            print(f'      * epoch {epoch: >6}: train loss = {loss_train_val : .4e}, test loss = {loss_test_val : .4e}')

        return model


    def autoencoder_sweep(self, X_or_Y, N_trial_dims, N_epochs, N_layers_autoencoder):
        
        print_header(f'autoencoder sweep for {X_or_Y}')
        
        N_loadcases = len(self.df_loadcases)
        
        if X_or_Y == 'X':
            f_normalize = lambda data: self._normalize(data, 'X', 'normalize')
            f_load = lambda indicies: f_normalize(NeuralNetworkTimeSeries.load_data_from_disk(self.df_loadcases, indicies, 'fn_raw_data', self.device)['X'])
        elif X_or_Y == 'Y':
            f_normalize = lambda data: self._normalize(data, 'Y', 'normalize')
            f_load = lambda indicies: f_normalize(NeuralNetworkTimeSeries.load_data_from_disk(self.df_loadcases, indicies, 'fn_raw_data', self.device)['Y'])
        else:
            raise(Exception('must be X or Y'))
        
        for compressed_dim in N_trial_dims:
            print(f' * training autoencoder with {compressed_dim} compressed dimensions and {N_layers_autoencoder} layers for {N_epochs} epochs')
            autoencoder_model  = NeuralNetworkTimeSeries._train_autoencoder(self.device,  self.train_test_indicies, f_load, compressed_dim, N_epochs, N_layers_autoencoder, N_loadcases)           

            if X_or_Y == 'X':
                self.autoencodersX[compressed_dim]  = autoencoder_model
            elif X_or_Y == 'Y':
                self.autoencodersY[compressed_dim]  = autoencoder_model
            else:
                raise(Exception('must be X or Y'))
        
        if X_or_Y == 'X':
            NeuralNetworkTimeSeries._plot_autoencoder_sweep_loss(self.autoencodersX, 'X', output_folder = self.folders['plots_losses_dir'])
        elif X_or_Y == 'Y':
            NeuralNetworkTimeSeries._plot_autoencoder_sweep_loss(self.autoencodersY, 'Y', output_folder = self.folders['plots_losses_dir'])
        else:
            raise(Exception('must be X or Y'))    
        

    def normalize_and_reduce_dimensionality(self, Nx, Ny):
        
        print_header('Normalize and reduce dimensionality via autoencoder')
   
        df_loadcases = self.df_loadcases
   
    
        modelX = self.autoencodersX[Nx]
        modelY = self.autoencodersY[Ny]
        self.autoencoderX = modelX
        self.autoencoderY = modelY
        
        f_normalizeX = lambda data: self._normalize(data, 'X', 'normalize')
        f_normalizeY = lambda data: self._normalize(data, 'Y', 'normalize')

        print(f' * reducing inputs to {Nx} dimensions')
        print(f' * reducing outputs to {Ny} dimensions')

        for idx in range(len(df_loadcases)):
            encoded_dataX = f_normalizeX(NeuralNetworkTimeSeries.load_data_from_disk(df_loadcases, [idx,], 'fn_raw_data', self.device)['X'])
            encoded_dataX = modelX.encoder(encoded_dataX)
            encoded_dataX = encoded_dataX[0, :, :]
            
            encoded_dataY = f_normalizeY(NeuralNetworkTimeSeries.load_data_from_disk(df_loadcases, [idx,], 'fn_raw_data', self.device)['Y'])
            encoded_dataY = modelY.encoder(encoded_dataY)
            encoded_dataY = encoded_dataY[0, :, :]          
            
            name = df_loadcases.loc[:, 'name'].iloc[idx]

            fn = os.path.join(self.folders['data_encoded_dir'],  f'{name}.pkl')
            torch.save((encoded_dataX, encoded_dataY), fn)
            
            self.df_loadcases.loc[name, 'fn_encoded_data'] = fn
            
            print(f' * saved {fn}')
        
        
        
    def train_timeseries_model(self, N_hidden_dim, N_layers, N_epochs, learn_rate = .01, verbose = True, gradiant_clip = True):
        
        print_header('train timeseries model')

        indicies = [0, ]
        f_load = lambda indicies: NeuralNetworkTimeSeries.load_data_from_disk(self.df_loadcases, indicies, 'fn_encoded_data', self.device)
        data = f_load(indicies)
        
        N_inputs = data['X'].shape[-1]
        N_outputs =  data['Y'].shape[-1]
                
        model = GRUNet(input_dim = N_inputs, hidden_dim = N_hidden_dim, output_dim = N_outputs, n_layers = N_layers, device = self.device,)
        
        print(f' * initialized GRU model with {N_inputs} inputs, {N_outputs} outputs, {N_layers} layers, {N_hidden_dim} hidden dims')
        
        model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        model.train()
        
        t0 = time.perf_counter()
        
        train_test_indicies = self.train_test_indicies
        N_times_through_batches = 5  # time
        
        N_batches = len(train_test_indicies['train'])
        N_loadcases_per_batch = np.average([len(x) for x in train_test_indicies['train']])
        N_loadcases = len(self.df_loadcases)               
        epoch = 0
        total_loadcases_trained_on = 0 
        
        
        for j in range(N_times_through_batches):        
            
            for train_batch in train_test_indicies['train']:
                data = f_load(train_batch)
                input_train = data['X']
                output_train = data['Y']
                
                n_lim = np.ceil(N_epochs * N_loadcases/(N_times_through_batches * N_batches * N_loadcases_per_batch)).astype(int)
                
                for n in range(n_lim):
                    optimizer.zero_grad()
                    out_train_predicted, _ = model(input_train)
                    
                    loss_train = criterion(out_train_predicted, output_train) 
                    loss_train.backward(retain_graph=True)
                    
                    if gradiant_clip == True:
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=.01)
                    
                    optimizer.step()  
                    loss_train_val = loss_train.item()
                    
                    total_loadcases_trained_on += input_train.shape[0]
                    epoch = total_loadcases_trained_on/N_loadcases

                    model.losses_train.append((float(epoch), float(loss_train_val)))

                    t1 = time.perf_counter()
                    dt = t1 -t0
                    
                    if (dt >= 1 and verbose == True) or (epoch == N_epochs-1) or (epoch == 0):
                        print(f' * epoch {epoch: >6}: train loss = {loss_train_val : .4e}')
                        t0 = t1
        
            test_losses_per_batch = []
            for test_batch in train_test_indicies['test']:   
                data = f_load(test_batch)
                input_test = data['X']
                output_test = data['Y']
                out_test_predicted, _ = model(input_test)
                loss_test = criterion(out_test_predicted, output_test)          
                test_losses_per_batch.append(loss_test.item())
            loss_test_val = np.average(test_losses_per_batch)
            model.losses_test.append((float(epoch), float(loss_test_val)))
            print(f' * epoch {epoch: >6}: train loss = {loss_train_val : .4e}, test loss = {loss_test_val : .4e}')
       
    
        model.plot_losses(self.folders['plots_losses_dir'])

        self.model = model
    
    
    
    def _error_plot(U, Y_actual, Y_predicted, name, output_folder = '.'):
        
        error = Y_actual-Y_predicted
        
        fig, ax = plt.subplots(3, 1)
        fig.set_dpi(200)
        ax[0].plot(U[0, :, :].cpu().detach().numpy())
        ax[0].set_title(f'inputs ({U.shape[-1]})')
        
        ax[1].plot(Y_actual[0, :, :].cpu().detach().numpy())
        ax[1].plot(Y_predicted[0, :, :].cpu().detach().numpy(), linestyle = '--', color = 'k', linewidth = .5)
        ax[1].set_title(f'outputs ({Y_predicted.shape[-1]})')
        
        ax[2].plot(error[0, :, :].cpu().detach().numpy())
        ax[2].set_title(f'error ({error.shape[-1]})')
        
        for i in range(3):
            ax[i].grid(c = [.9, .9, .9])
            ax[i].set_axisbelow(True)
        
        fig.suptitle(f'{name}')
        
        fig.tight_layout()
        
        fn = os.path.join(output_folder, f'{name}.pdf')
        fig.savefig(fn)
        
        print(f' * saved {fn}')
    
        return None
        
   
    def load_data_from_disk(df_loadcases, indicies, col, device):
        
        files = df_loadcases.loc[:, col].iloc[indicies].to_list()
        
        inputs = []
        outputs = []
        
        for file in files:
            (X, Y) = torch.load(file)
            inputs.append(X)
            outputs.append(Y)
            
        inputs = torch.stack(inputs).to(device)
        outputs = torch.stack(outputs).to(device)
        
        return {'X':inputs, 'Y':outputs}
        
            
        
        
    
   
    def assess_fit(self, plot = True, plot_normalized = True):
        
        print_header('assess fit on training data')
    
        test_indicies = np.concatenate(self.train_test_indicies['test'])

        df_loadcases = self.df_loadcases

        for idx in test_indicies:
            f_load = lambda indicies: NeuralNetworkTimeSeries.load_data_from_disk(self.df_loadcases, indicies, 'fn_raw_data', self.device)
            data = f_load([idx,])
            X = data['X']
            Y_actual = data['Y']
            Y_pred = self.predict(X)
            name = df_loadcases['name'].iloc[idx]
            
            fx = lambda data: self._normalize(data, 'X', 'normalize')
            fy = lambda data: self._normalize(data, 'Y', 'normalize')
        
            if plot_normalized == True and plot == True:
                NeuralNetworkTimeSeries._error_plot(fx(X), fy(Y_actual), fy(Y_pred), name, output_folder = self.folders['plots_signals_dir'])
            elif plot == True:
                NeuralNetworkTimeSeries._error_plot(X, Y_actual, Y_pred, name, output_folder = self.folders['plots_signals_dir'])
            else:
                print(' * no plots generated because plot = False')
    
    
    def predict(self, X):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float().to(self.device)
            
        X = self._normalize(X, 'X', 'normalize')
        X = self.autoencoderX.encoder(X)
        X, _ = self.model(X)  # this is really Y as output, but reassigning to save memory
        X = self.autoencoderY.decoder(X)
        X = self._normalize(X, 'Y', 'unnormalize')
        
        return X     
    
    
    def  train_test_split(self, test_frac, batch_size):
        
         print_header('train test split with batches')
        
         N_loadcases = len(self.df_loadcases)
        
         train_test_indicies  = NeuralNetworkTimeSeries._train_test_split_with_batches(batch_size, test_frac, N_loadcases)
         
         self.train_test_indicies = train_test_indicies
         
    
    def _train_test_split_with_batches(batch_size, test_frac, N_loadcases):

        N_test = np.ceil(N_loadcases*test_frac).astype(int)
        idx = list(range(N_loadcases))
        idx_test = idx[:N_test]
        idx_train = idx[N_test:]
        idx_test_batches = [idx_test[x:x+batch_size] for x in range(0, len(idx_test), batch_size)]
        idx_train_batches = [idx_train[x:x+batch_size] for x in range(0, len(idx_train), batch_size)]
        
        batches = {}
        batches['train'] = idx_train_batches
        batches['test'] = idx_test_batches
        #print(batches)
        
        print(f' * train ({100-100*test_frac:.0f}%): {len(idx_train)} loadcases, {len(idx_train_batches)} batches, up to {batch_size} loadcases/batch')
        print(f' * test ({100*test_frac:.0f}%): {len(idx_test)} loadcases, {len(idx_test_batches)} batches, up to {batch_size} loadcases/batch')

        return batches
    

    def save_model(self):
        print_header('save model')
        fn = os.path.join(self.folders['models_dir'], 'all_models.pkl')
        with open(fn, 'wb') as h:
            pickle.dump(self, h, pickle.HIGHEST_PROTOCOL)
        print(f' * saved model to {fn}')
        
    def wrapup(self):
        
        
        
        
        print_header('done')

    def _plot_autoencoder_sweep_loss(autoencoders, varname = '', output_folder = '.'):
    
        fig, ax = plt.subplots()
        fig.set_dpi(200)
        for key, data in autoencoders.items():
            
            epoch = [x[0] for x in autoencoders[key].losses_test]
            loss = [x[1] for x in autoencoders[key].losses_test]
            h = ax.plot(epoch, loss, label = f'test, {key} dimensions')
            
            epoch = [x[0] for x in autoencoders[key].losses_train]
            loss = [x[1] for x in autoencoders[key].losses_train]
            ax.plot(epoch, loss, linestyle = '--', label = f'train, {key} dimensions', color = h[0].get_color())
            
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
        if n_layers ==1:
            drop_prob = 0  # cant have dropout with 1 layer only
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
        
        fig, ax = plt.subplots()
        fig.set_dpi(200)   
        
        epoch = [x[0] for x in self.losses_train]
        loss = [x[1] for x in self.losses_train]
        ax.plot(epoch, loss, label = 'train loss')
        
        epoch = [x[0] for x in self.losses_test]
        loss = [x[1] for x in self.losses_test]
        ax.plot(epoch, loss, label = 'test loss')    
        
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
        print(f' * saved {fn}')

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


