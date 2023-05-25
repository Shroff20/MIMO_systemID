
from shroff_timeseries_toolbox import NeuralNetworkTimeSeries, FakeDataMaker


#%% INPUTS

working_dir = 'demo_output\\' # will store all model data

# fake dataset creation
N_inputs = 3                                    # number of input features
N_outputs = 5                                   # number of output features
N_loadcases = 20                                # number of cases (runs)

# training options
batch_size = 10                                  # will load up to this many loadcases into memory at a time (RAM usage vs. disk read tradeoff)
test_frac = .3                                  # fraction of data to use for testing

# input autoencoder
N_epochs_autoencoderX = 50                    # epochs to train autoencoder
trial_dims_autoencoderX = [1, N_inputs]         # list of compressed dimensions to try
N_layers_autoencoderX = 1                       # autoencoder layers (start with 1, then increase if a more complicated model is needed)

# output autoencoder
N_epochs_autoencoderY = 10                   # epochs to train autoencoder
trial_dims_autoencoderY = [1, N_outputs]        # list of compressed dimensions to try
N_layers_autoencoderY = 1                       # autoencoder layers (start with 1, then increase if a more complicated model is needed)

# autoencoder final choices
N_dim_X_autoencoder = N_inputs                  # chosen number of dimensions to encode inputs
N_dim_Y_autoencoder = N_outputs                 # chosen number of dimensions to encode outputs

# timeseries model
N_epochs_RNN = 100                              # epochs to train timeseries model (RNN)
N_layers_RNN = 1                                # RNN layers (start with 1, then increase if a more complicated model is needed)
N_hidden_dim_RNN = 10                          # RNN memory length: increase to lengthen memory of past input values (should be at least time constant of the system)


#%% PIPELINE

G, inputs, outputs, t = FakeDataMaker.generate_fake_data(N_inputs, N_outputs, N_loadcases) # generate fake data

NNTS = NeuralNetworkTimeSeries(working_dir = working_dir) # initialize model

# add loadcases one at a time.  Data must me uniformly spaced in time with the same time increment over all loadcases.  
for i in range(inputs.shape[0]):
    NNTS.add_loadcase_data(f'loadcase_{i}', inputs[i, :, :], outputs[i, :, :]) # loadcase name, [N_timesteps, N_inputs],   [N_timesteps, N_outputs]

NNTS.train_test_split(test_frac, batch_size) # splits into train and text batches

NNTS.autoencoder_sweep('X', trial_dims_autoencoderX, N_epochs_autoencoderX, N_layers_autoencoderX)  # trains autoencoders for the input of various dimensionality for review
NNTS.autoencoder_sweep('Y', trial_dims_autoencoderY, N_epochs_autoencoderY, N_layers_autoencoderY) # trains autoencoders for the output of various dimensionality for review

NNTS.normalize_and_reduce_dimensionality(N_dim_X_autoencoder, N_dim_Y_autoencoder) # normalizes data and compresses input and output based on chosen dimensionality
NNTS.train_timeseries_model(N_hidden_dim_RNN, N_layers_RNN, N_epochs_RNN) # trains timeseries model
NNTS.save_model() # save model to pkl file
NNTS.assess_fit() # assses fit of the final model
NNTS.plot_detailed_predictions() # assses fit of the final model

NNTS.plot_predictions() # assses fit of the final model
NNTS.wrapup() # wrap up study

#%%





