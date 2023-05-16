
from shroff_timeseries_toolbox import NeuralNetworkTimeSeries, FakeDataMaker


if __name__ == '__main__':

    
    #%% INPUTS

    working_dir = 'demo_output\\'

    # dataset
    N_inputs = 3
    N_outputs = 5
    N_loadcases = 20
    
    batch_size = 3   # will load up to this many loadcases into memory at a time (RAM usage vs. disk read tradeoff)
    test_frac = .3   # fraction of data to use for testing
    
    #autoencoder
    N_epochs_autoencoderX = 50
    trial_dims_autoencoderX = range(1, N_inputs+1)
    N_layers_autoencoderX = 1
    
    N_epochs_autoencoderY = 100
    trial_dims_autoencoderY = range(1, N_outputs+1)
    N_layers_autoencoderY = 1
    
    N_dim_X_autoencoder = N_inputs
    N_dim_Y_autoencoder = N_outputs
    
    # RNN
    N_epochs_RNN = 100
    N_layers_RNN = 1
    N_hidden_dim_RNN = 100
    
    
    #%% PIPELINE
    
    G, inputs, outputs, t = FakeDataMaker.generate_fake_data(N_inputs, N_outputs, N_loadcases)
    
    NNTS = NeuralNetworkTimeSeries(working_dir = working_dir)
    
    for i in range(inputs.shape[0]):
        NNTS.add_loadcase_data(f'loadcase_{i}', inputs[i, :, :], outputs[i, :, :])
    
    NNTS.generate_normalization_functions()
    NNTS.train_test_split(test_frac, batch_size)
    
    NNTS.autoencoder_sweep('X', trial_dims_autoencoderX, N_epochs_autoencoderX, N_layers_autoencoderX)
    NNTS.autoencoder_sweep('Y', trial_dims_autoencoderY, N_epochs_autoencoderY, N_layers_autoencoderY)
    
    NNTS.normalize_and_reduce_dimensionality(N_dim_X_autoencoder, N_dim_Y_autoencoder)
    NNTS.train_timeseries_model(N_hidden_dim_RNN, N_layers_RNN, N_epochs_RNN)
    NNTS.assess_fit()
    NNTS.wrapup()



