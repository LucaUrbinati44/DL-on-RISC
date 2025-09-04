# %%
import os
import random
import numpy as np

import tensorflow as tf

# Imposta il seed globale
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

#from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.saving import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, ReLU, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
# Load the LiteRT model and allocate tensors.
from ai_edge_litert.interpreter import Interpreter # LiteRT (Lite Runtime) successore del runtime tflite_runtime

import h5py
import time
import subprocess
import psutil


labels = [str(i) for i in range(1024)]
num_classes = len(labels)

# Imposta float32 come tipo predefinito in Keras
tf.keras.backend.set_floatx('float32')

# Tipo da usare esplicitamente (es. in numpy)
force_datatype = np.float32

use_gpu = 1

# ------------------ Configurazione GPU ------------------ #
if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disabilita la GPU

print(f"\nUsing GPU: {tf.config.list_physical_devices('GPU')}")
print(tf.config.list_physical_devices('GPU'))

tf.config.optimizer.set_jit(False)  # XLA off

# %% Define functions

def model_predict(xdataset, Y_dataset, 
                  xval, Y_val, 
                  xtest, Y_test, 
                  YValidation_un_val, YValidation_un_test, 
                  model_py, 
                  network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load,
                  mean_array_filepath, variance_array_filepath, test_set_size, 
                  test, save_files=1):

    print(f"\n### model_predict {t}")

    if test == 4: # entire dataset prediction (for figC)
        t = '_tflite'
        x = xdataset
        y = Y_dataset
    elif test == 3:
        t = '_test_tflite'
        #x = xtest
        # Choose the number of test samples
        if test_set_size == 'small':
            xtest_size = 10
        else:
            xtest_size = xtest.shape[0]
        x = xtest[:xtest_size,:]
        y = Y_test
        YValidation_un = YValidation_un_test
    elif test == 2: # entire dataset prediction (for figC)
        t = ''
        x = xdataset
        y = Y_dataset
    elif test == 1: # test set FP prediction
        t = '_test'
        x = xtest
        y = Y_test
        YValidation_un = YValidation_un_test
    elif test == 0: # validation set prediction
        t = '_val'
        x = xval
        y = Y_val
        YValidation_un = YValidation_un_val
    
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)

    filename_Indmax_OPT_py = network_folder_out_RateDLpy + 'Indmax_OPT_py'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    filename_Indmax_DL_py = network_folder_out_RateDLpy + 'Indmax_DL_py'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'

    filename_MaxR_OPT_py = network_folder_out_RateDLpy + 'MaxR_OPT_py'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    filename_MaxR_DL_py = network_folder_out_RateDLpy + 'MaxR_DL_py'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    filename_Rate_OPT_py = network_folder_out_RateDLpy + 'Rate_OPT_py'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    filename_Rate_DL_py = network_folder_out_RateDLpy + 'Rate_DL_py'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'

    print(f"\nStart DL prediction {t}...")

    Indmax_OPT_py = np.argmax(y, axis=1) # ATTENZIONE: ricordarsi di fare +1 in Matlab per ripristinare indici da zero-based a one-based
    print(f'Indmax_OPT_py.shape: {Indmax_OPT_py.shape}')
    print(f'np.min(Indmax_OPT_py): {np.min(Indmax_OPT_py)}')
    print(f'np.max(Indmax_OPT_py): {np.max(Indmax_OPT_py)}')
    print(Indmax_OPT_py[0:5])

    if test == 0 or test == 1 or test == 2:
        YPredicted = model_py.predict(x, verbose=1, batch_size=128)
    else:
        ## Gets metadata from the model file.
        # Load scalers
        #mean_array_filepath = mcu_profiling_folder_scaler + 'mean' + end_folder_Training_Size_dd + '.npy'
        #with open(mean_array_filepath, 'r') as f:
        #    mean_array = f.read()
        mean_array = np.load(mean_array_filepath)
                
        #variance_array_filepath = mcu_profiling_folder_scaler + 'variance' + end_folder_Training_Size_dd + '.npy'
        #with open(variance_array_filepath, 'w') as f:
        #    variance_array = f.read()
        variance_array = np.load(variance_array_filepath)

        #tflite_model = schema_fb.Model.GetRootAsModel(tflite_quant_model, 0)
        #for i in range(tflite_model.MetadataLength()):
        #    meta = tflite_model.Metadata(i)
        #    if meta.Name().decode("utf-8") == "min_runtime_version":
        #        buffer_index = meta.Buffer()
        #        metadata = tflite_model.Buffers(buffer_index)
        #        min_runtime_version_bytes = metadata.DataAsNumpy().tobytes()
        #        print(min_runtime_version_bytes)

        #Loading and running a LiteRT model involves the following steps:
        #1) Loading the model into memory.
        #2) Building an Interpreter based on an existing model.
        # Load the LiteRT model and allocate tensors.
        interpreter = Interpreter(model_content=model_py)
        interpreter.allocate_tensors()
        details = interpreter.get_tensor_details()
        
        # Stampa l'elenco degli operatori
        for op in interpreter._get_ops_details():
            print(op['op_name'])

        #3) Setting input tensor values.
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        #input_dtype = input_details[0]['dtype'] #int8
        #output_dtype = input_details[0]['dtype']
        #print('input_dtype:', input_dtype) # [   1 1024]
        #print('output_dtype:', output_dtype)
        #input_shape = input_details[0]['shape']
        #output_shape = output_details[0]['shape']
        #print('input_shape:', input_shape)
        #print('output_shape:', output_shape)

        input_scale, input_zero_point = input_details[0]['quantization']
        #print('input_scale:', input_scale)
        #print('input_zero_point:', input_zero_point)
        
        input_float = x
        #print('input_float.shape:', input_float.shape) # (3100, 1024)
        #print('np.max(input_float):', np.max(input_float))
        #print('np.min(input_float):', np.min(input_float))

        num_samples = input_float.shape[0]
        output_float = []

        for i in range(num_samples):
        #for i in range(3):
            #print('i:', i)

            sample = input_float[i]  # shape: (1024,)
            #print('sample.shape:', sample.shape)
            print('\nfloat_input:', ' '.join([f"{x:.6f}" for x in sample[0:5]]))
            sample = np.expand_dims(sample, axis=0)  # shape: (1, 1024)

            sample_normalized = np.array((sample - mean_array) / np.sqrt(variance_array), dtype=np.float32)
            
            input_int8 = np.round(sample / input_scale + input_zero_point).astype(np.int8)
            print('quantized_input:', ' '.join([f"{x}" for x in input_int8.flatten()[0:5]]))
            #print('input_int8.shape:', input_int8.shape)
            #print('np.max(input_int8):', np.max(input_int8))
            #print('np.min(input_int8):', np.min(input_int8))

            #4) Invoking inferences.
            # Test the model on input data.
            #input_float = np.array(np.random.random_sample(input_shape), dtype=np.float32) # random input data

            interpreter.set_tensor(input_details[0]['index'], input_int8)
            interpreter.invoke()

            #5) Outputting tensor values
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_int8 = interpreter.get_tensor(output_details[0]['index'])
            #print('output_int8.shape:', output_int8.shape)
            #print('np.max(output_int8):', np.max(output_int8))
            #print('np.min(output_int8):', np.min(output_int8))
            output_scale, output_zero_point = output_details[0]['quantization']
            #print(f'output_scale: {output_scale:.6f}')
            #print(f'output_zero_point: {output_zero_point:.6f}')
            output_deq = (output_int8.astype(np.float32) - output_zero_point) * output_scale
            print('quantized_output:', ' '.join([f"{x}" for x in output_int8.flatten()[0:5]]))
            print('dequantized_output:', ' '.join([f"{x:.6f}" for x in output_deq[0][0:5]]))            #print('output_deq.shape:', output_deq.shape)
            #print('np.max(output_deq):', np.max(output_deq))
            #print('np.min(output_deq):', np.min(output_deq))
            output_float.append(output_deq[0])  # output_deq.shape: (1, 1024) -> prendi [0]
            
            #print(f"\ncodebook_index {i+1}: {np.argmax(output_deq[0])}\n")

        YPredicted = np.stack(output_float, axis=0)  # shape: (3100, 1024)

    #print(f'x.shape: {x.shape}')  # (3100, 1024)
    #print(f'YPredicted.shape: {YPredicted.shape}')
    #print('np.max(YPredicted):', np.max(YPredicted))
    #print('np.min(YPredicted):', np.min(YPredicted))

    Indmax_DL_py = np.argmax(YPredicted, axis=1)
    #print(f'Indmax_DL_py.shape: {Indmax_DL_py.shape}')
    # Questi devono essere numeri interi
    print(f'np.min(Indmax_DL_py): {np.min(Indmax_DL_py)}')
    print(f'np.max(Indmax_DL_py): {np.max(Indmax_DL_py)}')
    print(f'max - min = {np.max(Indmax_DL_py) - np.min(Indmax_DL_py)}')
    print(f'Indmax_DL_py[0:5]: {Indmax_DL_py[0:5]}')

    if test == 0 or test == 1 or test == 3:
        #validation_accuracy = 0
        MaxR_OPT_py = np.zeros((Indmax_OPT_py.shape[0],), dtype=np.float32)
        MaxR_DL_py = np.zeros((Indmax_DL_py.shape[0],), dtype=np.float32)

        # Ciclo di confronto
        for b in range(Indmax_DL_py.shape[0]):
            MaxR_OPT_py[b] = YValidation_un[Indmax_OPT_py[b], b] # YValidation_un.shape: (1024, 3100)
            MaxR_DL_py[b]  = YValidation_un[Indmax_DL_py[b],  b]

            #if MaxR_DL[b] == MaxR_OPT[b]:
            #    validation_accuracy += 1

        Rate_OPT_py = MaxR_OPT_py.mean()
        Rate_DL_py = MaxR_DL_py.mean()
        print(f'Rate_OPT: {Rate_OPT_py}')
        print(f'Rate_DL_py: {Rate_DL_py}')
        #validation_accuracy = validation_accuracy / Indmax_DL_py.shape[0]

        #print(f"size(MaxR_DL): {MaxR_DL.shape}")
    else:
        Rate_OPT_py = 0
        Rate_DL_py = 0


    #if test == 1 or test == 2 or test == 3:
       
    # Scrittura in formato HDF5 (compatibile MATLAB v7.3)
    if save_files == 1:

        #if 0 == 1: # TODO: temporaneamente non voglio salvarli perchè già salvati in precedenza
        with h5py.File(filename_Indmax_OPT_py, 'w') as f:
            f.create_dataset('Indmax_OPT_py', data=Indmax_OPT_py)
            print(f"\n-->Indmax_OPT_py saved in {filename_Indmax_OPT_py}")

        with h5py.File(filename_MaxR_OPT_py, 'w') as f:
            f.create_dataset('MaxR_OPT_py', data=MaxR_OPT_py)
            print(f"---->MaxR_OPT_py saved in {filename_MaxR_OPT_py}")

        with h5py.File(filename_Rate_OPT_py, 'w') as f:
            f.create_dataset('Rate_OPT_py', data=Rate_OPT_py)
            print(f"------>Rate_OPT_py saved in {filename_Rate_OPT_py}")
        
        #if test == 3:
        with h5py.File(filename_Indmax_DL_py, 'w') as f:
            f.create_dataset('Indmax_DL_py', data=Indmax_DL_py)
            print(f"-------->Indmax_DL_py saved in {filename_Indmax_DL_py}")

        with h5py.File(filename_MaxR_DL_py, 'w') as f:
            f.create_dataset('MaxR_DL_py', data=MaxR_DL_py)
            print(f"--------->MaxR_DL_py saved in {filename_MaxR_DL_py}")

        with h5py.File(filename_Rate_DL_py, 'w') as f:
            f.create_dataset('Rate_DL_py', data=Rate_DL_py)
            print(f"------------>Rate_DL_py saved in {filename_Rate_DL_py}")
        
    return Indmax_OPT_py, Indmax_DL_py, Rate_OPT_py, Rate_DL_py

def mse_keras(y_true, y_pred): # Not working
    squared_error = tf.square(y_true - y_pred)  # shape: (batch_size, output_dim)=6200,1024
    loss = tf.reduce_mean(squared_error) # scalar
    return loss

def mse_custom(y_true, y_pred):
    # Calcola l'errore quadratico tra vero e predetto
    squared_error = tf.square(y_true - y_pred)  # shape: (batch_size, output_dim)=6200,1024

    # Somma degli errori lungo l'ultima dimensione (output_dim)
    sum_squared_error = tf.reduce_sum(squared_error, axis=-1)  # shape: (batch_size,)=6200

    # Media su tutto il batch
    loss = 0.5 * tf.reduce_mean(sum_squared_error)  # scalar
    return loss

def mse_matlab(y_true, y_pred): # Not working
    # A regression layer computes the half-mean-squared-error loss for regression tasks:
    # https://www.mathworks.com/help//releases/R2021a/deeplearning/ref/regressionlayer.html?searchHighlight=regressionLayer&searchResultIndex=1
    squared_error = tf.square(y_true - y_pred)  # shape: (batch_size, output_dim)=6200,1024
    
    loss = 0.5 * tf.reduce_mean(squared_error)
    return loss

# %% [markdown]
# ## Define variables

# %%
base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'

input_folder = base_folder + 'Output Matlab/'

#DeepMIMO_dataset_folder = input_folder + 'DeepMIMO Dataset/'
DL_dataset_folder = input_folder + 'DL Dataset/'
network_folder_in = input_folder + 'Neural Network/'

output_folder = base_folder + 'Output_Python/'
network_folder_out = output_folder + 'Neural_Network/'
network_folder_out_RateDLpy = output_folder + 'Neural_Network/RateDLpy/'
network_folder_out_RateDLpy_TFLite = output_folder + 'Neural_Network/RateDLpy_TFLite/'
saved_models_keras = network_folder_out + 'saved_models_keras/'
saved_models_tflite = network_folder_out + 'saved_models_tflite/'
figure_folder = output_folder + 'Figures/'
mcu_profiling_folder = output_folder + 'Profiling_Search_MCU/'
mcu_profiling_folder_model = mcu_profiling_folder + 'model/'
mcu_profiling_folder_scaler = mcu_profiling_folder + 'scaler/'
mcu_profiling_folder_test_data_normalized = mcu_profiling_folder + 'test_data_normalized/'
mcu_profiling_folder_test_data = mcu_profiling_folder + 'test_data/'

folders = [
    output_folder,
    network_folder_out,
    network_folder_out_RateDLpy,
    network_folder_out_RateDLpy_TFLite,
    saved_models_keras,
    saved_models_tflite,
    figure_folder,
    mcu_profiling_folder,
    mcu_profiling_folder_model,
    mcu_profiling_folder_scaler,
    mcu_profiling_folder_test_data_normalized,
    mcu_profiling_folder_test_data
]

for folder in folders:
    if not os.path.exists(folder):  # Controlla se la cartella esiste
        os.makedirs(folder, exist_ok=True)  # Crea la cartella se non esiste
        print(f"\nCartella creata: {folder}")
    #else:
    #    print(f"La cartella esiste già: {folder}")

# ----- Costruzione del modello MLP parametrico -----
def build_mlp(input_features, output_dim, num_layers, hidden_units_list):
    model = tf.keras.Sequential([tf.keras.Input(shape=(input_features,), name='input')])
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(hidden_units_list[i], activation='relu', name=f'hidden_{i}'))
    model.add(tf.keras.layers.Dense(output_dim, activation=None, name='output'))
    return model

def main(My, Mz, load_model_flag, max_epochs, max_epochs_load, 
        train_model_flag, predict_loaded_model_flag,
        convert_model_flag, Training_Size_dd, 
        input_features, output_dim, num_layers, hidden_units_list,
        mean_array_filepath, variance_array_filepath, test_set_size,
        end_folder, end_folder_Training_Size_dd, 
        model_name_suffix, end_folder_Training_Size_dd_max_epochs_load, model_type_load, model_path_tflite,
        save_files_flag_master, save_files_flag_master_once):

        print("### DL_training_4_v3_test")

        # %%
        print(f"\nRIS: {My}x{Mz}")
        print(f"\nTraining_Size_dd: {Training_Size_dd}")
                
        ## Load Dataset DL_input_reshaped

        filename_DL_input_reshaped = DL_dataset_folder + 'DL_input_reshaped' + end_folder + '.mat'
        filename_DL_output_reshaped = DL_dataset_folder + 'DL_output_reshaped' + end_folder + '.mat'
        filename_RandP_all = DL_dataset_folder + 'RandP_all' + end_folder + '.mat'

        # Load the data using h5py for MATLAB v7.3 files
        with h5py.File(filename_DL_input_reshaped, 'r') as f:
            DL_input_reshaped = np.array(f['DL_input_reshaped'][:], dtype=force_datatype)
        with h5py.File(filename_DL_output_reshaped, 'r') as f:
            DL_output_reshaped = np.array(f['DL_output_reshaped'][:], dtype=force_datatype)
        with h5py.File(filename_RandP_all, 'r') as f:
            RandP_all = np.array(f['RandP_all'][:], dtype=force_datatype)

        print(DL_input_reshaped.shape)
        print(DL_output_reshaped.shape)
        print(RandP_all.shape)

        #print(np.min(DL_input_reshaped))
        #print(np.max(DL_input_reshaped))
        #print(np.min(DL_output_reshaped))
        #print(np.max(DL_output_reshaped))


        # %%
        ## Load Rates

        # Costruzione del nome file
        #filename_DL_output_un_reshaped = DL_dataset_folder + 'DL_output_un_reshaped' + end_folder + '.mat'
        filename_DL_output_un_complete_reshaped= DL_dataset_folder + 'DL_output_un_complete_reshaped' + end_folder + '.mat'

        # Load the data using h5py for MATLAB v7.3 files
        with h5py.File(filename_DL_output_un_complete_reshaped, 'r') as f:
            # Accesso alla variabile (nome del dataset = nome della variabile in MATLAB)
            YValidation_un = np.array(f['DL_output_un_complete_reshaped'], dtype=force_datatype)

        print(f'YValidation_un.shape: {YValidation_un.shape}') #YValidation_un.shape: (36200, 1024, 1, 1)
        YValidation_un = np.transpose(YValidation_un, (3, 2, 1, 0))  # conversione a (b, z, y, x)
        print(f'YValidation_un.shape: {YValidation_un.shape}') #YValidation_un.shape: (1, 1, 1024, 36200)


        # %%
        ## Dataset split nuovo con test set, ipotesi di split 90-10 (per avere val+test = Validazion_Size = 6200)
        # L'importante è garantire un sufficiente numero di sample di test che rappresenti bene la distribuzione dei dati

        dataset_size = 36200

        Validation_Size_old = 6200
        Test_Size = int(Validation_Size_old/2)
        Validation_Size = Test_Size
        Dev_Size = dataset_size - Validation_Size - Test_Size
        Train_Size = Dev_Size - Validation_Size

        #print(Dev_Size)
        #print(Train_Size)
        #print(Validation_Size)
        #print(Test_Size)

        perc_train = Train_Size/Dev_Size
        perc_val = Validation_Size/Dev_Size
        print(perc_train)
        print(perc_val)

        RandP_all2 = np.squeeze(np.array(RandP_all.astype(int))) - 1 
        # -1 serve per fare andare tutto perchè altrimenti il valore 0 di DL_input_reshaped (per esempio) non verrebbe mai indicizzato/selezionato 

        Training_Ind = RandP_all2[0:Training_Size_dd]
        Test_Ind = RandP_all2[-Test_Size:]
        Validation_Ind = RandP_all2[-(Validation_Size+Test_Size):-Test_Size]

        #print(Training_Ind.shape)
        #print(Validation_Ind.shape)
        #print(Test_Ind.shape)


        # %%
        X_train = np.array(DL_input_reshaped[Training_Ind, :, :, :], dtype=force_datatype).squeeze()
        Y_train = np.array(DL_output_reshaped[Training_Ind, :, :, :], dtype=force_datatype).squeeze()
        X_val   = np.array(DL_input_reshaped[Validation_Ind, :, :, :], dtype=force_datatype).squeeze()
        Y_val   = np.array(DL_output_reshaped[Validation_Ind, :, :, :], dtype=force_datatype).squeeze()
        X_test  = np.array(DL_input_reshaped[Test_Ind, :, :, :], dtype=force_datatype).squeeze()
        Y_test  = np.array(DL_output_reshaped[Test_Ind, :, :, :], dtype=force_datatype).squeeze()       
        X_dataset = np.array(DL_input_reshaped[:, :, :, :], dtype=force_datatype).squeeze()
        Y_dataset = np.array(DL_output_reshaped[:, :, :, :], dtype=force_datatype).squeeze()
        
        # YValidation_un.shape: (1, 1, 1024, 36200)
        YValidation_un_val  = np.array(YValidation_un[:, :, :, Validation_Ind], dtype=force_datatype).squeeze()
        YValidation_un_test = np.array(YValidation_un[:, :, :, Test_Ind], dtype=force_datatype).squeeze()

        print(f"\nY_train.dtype: {Y_train.dtype}")

        print(X_train.shape) # (Training_Size_dd, 1024)
        print(Y_train.shape) 
        print(X_val.shape) # (3100, 1024)
        print(Y_val.shape)
        print(X_test.shape) # (3100, 1024)
        print(Y_test.shape)
        print(YValidation_un_val.shape)  # (1024, 3100)
        print(YValidation_un_test.shape) # (1024, 3100)
        print(X_dataset.shape) # (36200, 1024)
        print(Y_dataset.shape) 

        # %%
        ## Recreate the same network in Python

        ### Load normalization parameters from Matlab trained model
        #filename_trainedNet_scaler = network_folder_in + 'trainedNet_scaler' + end_folder_Training_Size_dd + '.mat'
        #with h5py.File(filename_trainedNet_scaler, 'r') as f:
        #    trainedNet_scaler = f['trainedNet_scaler'][:][0][0]

        ### Compute normalization parameters from Training Dataset
        trainedNet_scaler = np.mean(np.mean(X_train, axis=1))    
        print(f"\ntrainedNet_scaler: {trainedNet_scaler}")

        # This layer will shift and scale inputs into a distribution centered around 0 with standard deviation 1. 
        # It accomplishes this by precomputing the mean and variance of the data, and calling (input - mean) / sqrt(var) at runtime.
        # The mean and variance values for the layer must be either supplied on construction or learned via adapt(). 
        # adapt() will compute the mean and variance of the data and store them as the layer's weights. 
        # adapt() should be called before fit(), evaluate(), or predict().

        mean_array = np.array([trainedNet_scaler]*X_train.shape[1], dtype=force_datatype)
        variance_array =  np.array([1]*X_train.shape[1], dtype=force_datatype)
        #print(mean_array.shape)
        #print(mean_array[0])
        #print(variance_array.shape)
        #print(variance_array[0])

        ### Normalize data
        # Normalizzazione manuale se già hai mean_array e variance_array da MATLAB
        X_train_normalized = np.array((X_train - mean_array) / np.sqrt(variance_array), dtype=force_datatype)
        X_val_normalized = np.array((X_val - mean_array) / np.sqrt(variance_array), dtype=force_datatype)
        X_test_normalized = np.array((X_test - mean_array) / np.sqrt(variance_array), dtype=force_datatype)
        X_dataset_normalized = np.array((X_dataset - mean_array) / np.sqrt(variance_array), dtype=force_datatype)

        #print(mean_array)
        #print(variance_array)
        
        xtest_npy_filename = mcu_profiling_folder_test_data + 'test_set' + end_folder_Training_Size_dd + '.npy'
        xtestnorm_npy_filename = mcu_profiling_folder_test_data_normalized + 'test_set_normalized' + end_folder_Training_Size_dd + '.npy'
        xtestmean_npy_filename = mcu_profiling_folder_scaler + 'mean' + end_folder_Training_Size_dd + '.npy'
        xtestvar_npy_filename = mcu_profiling_folder_scaler + 'variance' + end_folder_Training_Size_dd + '.npy'

        if save_files_flag_master_once == 1:

            print(f"### Save test data to: {xtest_npy_filename} \nand to: {xtestnorm_npy_filename}")
            np.save(xtest_npy_filename, X_test)
            np.save(xtestnorm_npy_filename, X_test_normalized)

            print(f"### Save mean into: {xtestmean_npy_filename}")
            print(f"### Save variance into: {xtestvar_npy_filename}")
            np.save(xtestmean_npy_filename, mean_array)
            np.save(xtestvar_npy_filename, variance_array)

            #print(X_train[0][0:5])
            #print(mean_array[0:5])
            #print(X_train_normalized[0][0:5])

        normalized = 1

        if normalized == 1:
            xtrain = X_train_normalized
            xval = X_val_normalized
            xtest = X_test_normalized
            xdataset = X_dataset_normalized
        else:
            xtrain = X_train
            xval = X_val
            xtest = X_test

        #os._exit(0)

        # %%
        ## DL Model Training and Prediction

        if train_model_flag == 1:
            if load_model_flag == 1:
                max_epochs_new = max_epochs_load + max_epochs
                factor = 0.5
                patience = 3
                min_delta = 0.025
            else:
                max_epochs_new = max_epochs
                factor = 0.5
                patience = 3
                min_delta = 0.05

        # %%
        ################################ Load Model ################################
        if load_model_flag == 1:

            #end_folder_Training_Size_dd_max_epochs_load = end_folder_Training_Size_dd + '_' + str(max_epochs_load)
            #model_type_load = 'model_py_test' + end_folder_Training_Size_dd_max_epochs_load + model_name_suffix

            model_path_keras = saved_models_keras + model_type_load + '.keras'
            model_py = load_model(model_path_keras, custom_objects={'mse_custom': mse_custom})
            model_py.summary()
            print(f"\nLoad model: {model_path_keras}")

            model_path_tflite = saved_models_tflite + model_type_load + '_quant.tflite'
        
            if convert_model_flag == 1:
                print('\n### convert_model')

                #### V4 ####
                def representative_dataset():
                    #for i in range(min(100, len(x_sample))):
                    for i in range(xval.shape[0]): # (3100, 1024)
                        #yield [x_sample[i:i+1].astype(np.float32)]
                        data = xval[i,:]
                        yield [data.astype(np.float32)]                        

                # Enforce full integer quantization for all ops including the input and output
                converter = tf.lite.TFLiteConverter.from_keras_model(model_py)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8  # or tf.uint8
                converter.inference_output_type = tf.int8  # or tf.uint8
                tflite_quant_model = converter.convert()
            
                with open(model_path_tflite, 'wb') as f:
                    f.write(tflite_quant_model)

            else:
                with open(model_path_tflite, 'rb') as f:
                    tflite_quant_model = f.read()
                print('TFLite model loaded')

            
            # %%
            ################################ Predict with Loaded Model ################################
            Rate_OPT_py_load_val = 0
            Rate_DL_py_load_val = 0
            Rate_OPT_py_load_test = 0
            Rate_DL_py_load_test = 0
            Rate_OPT_py_load = 0
            Rate_DL_py_load = 0
            Rate_OPT_py_load_test_tflite = 0
            Rate_DL_py_load_test_tflite = 0

            if predict_loaded_model_flag == 1:
                print("\n### predict_loaded_model")

                save_files_flag = save_files_flag_master

                #save_files_flag = 0
                test = 0 # Predict with val set
                _, _, Rate_OPT_py_load_val, Rate_DL_py_load_val   = model_predict(xdataset, Y_dataset, 
                                                                            xval, Y_val, 
                                                                            xtest, Y_test,
                                                                            YValidation_un_val, YValidation_un_test, 
                                                                            model_py, 
                                                                            network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, 
                                                                            mean_array_filepath, variance_array_filepath, test_set_size, 
                                                                            test=test, save_files=save_files_flag)
                #save_files_flag = 1
                test = 1 # Predict with test set
                Indmax_OPT_py_load_test, Indmax_DL_py_load_test, Rate_OPT_py_load_test, Rate_DL_py_load_test = model_predict(xdataset, Y_dataset, 
                                                                            xval, Y_val, 
                                                                            xtest, Y_test, 
                                                                            YValidation_un_val, YValidation_un_test, 
                                                                            model_py, 
                                                                            network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, 
                                                                            mean_array_filepath, variance_array_filepath, test_set_size, 
                                                                            test=test, save_files=save_files_flag)
                #save_files_flag = 0
                #test = 2 # Predict with all dataset
                #_, _, Rate_OPT_py_load, Rate_DL_py_load = model_predict(xdataset, Y_dataset, 
                #                                                  xval, Y_val, 
                #                                                  xtest, Y_test, 
                #                                                  YValidation_un_val, YValidation_un_test, 
                #                                                  model_py, 
                #                                                  network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, 
                #                                                  test=test, save_files=save_files_flag)
                #save_files_flag = 0
                test = 3 # Predict with TF-Lite Model
                _, Indmax_DL_py_load_test_tflite, _, Rate_DL_py_load_test_tflite = model_predict(xdataset, Y_dataset, 
                                                                                          xval, Y_val, 
                                                                                          xtest, Y_test, 
                                                                                          YValidation_un_val, YValidation_un_test, 
                                                                                          tflite_quant_model, 
                                                                                          network_folder_out_RateDLpy_TFLite, end_folder_Training_Size_dd_max_epochs_load,
                                                                                          mean_array_filepath, variance_array_filepath, test_set_size,
                                                                                          test=test, save_files=save_files_flag)
                #save_files_flag = 1
                test = 4 # Predict with TF-Lite Model with all dataset
                #_, _, Rate_OPT_py_load_tflite, Rate_DL_py_load_tflite = model_predict(xdataset, Y_dataset, 
                #                                                                xval, Y_val, 
                #                                                                xtest, Y_test, 
                #                                                                YValidation_un_val, YValidation_un_test, 
                #                                                                tflite_quant_model, 
                #                                                                network_folder_out_RateDLpy_TFLite, end_folder_Training_Size_dd_max_epochs_load, 
                #                                                                test=test, save_files=save_files_flag)
            
            learning_rate = model_py.optimizer.learning_rate.numpy()
            print(f"Learning rate loaded model: {learning_rate}")

        # %%
        ################################ Train Model ################################
        if train_model_flag == 1 and load_model_flag == 0:
            print("\n### define model architecture")

            log_dir = "/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/Output_Python/Neural_Network/tensorboard_logs_test/"
            tensorboard_command = [
                "tensorboard",
                f"--logdir={log_dir}",
                "--port=6006",
                "--host=localhost"
            ]

            # Funzione per trovare e terminare TensorBoard se è già in esecuzione
            def terminate_tensorboard():
                for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if 'tensorboard' in process.info['name'] or \
                        (process.info['cmdline'] and 'tensorboard' in process.info['cmdline'][0]):
                            print(f"\nTerminazione di TensorBoard con PID: {process.info['pid']}")
                            os.kill(process.info['pid'], signal.SIGTERM)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass

            # Controlla se TensorBoard è già in esecuzione
            try:
                # Avvia TensorBoard in background
                tensorboard_process = subprocess.Popen(tensorboard_command)
                print(f"\nTensorBoard avviato in background.")
            except:
                print(f"\nfErrore nell'avvio di TensorBoard: {e}")
                # Chiudi TensorBoard se è già in esecuzione
                terminate_tensorboard()
                # Avvia TensorBoard in background
                tensorboard_process = subprocess.Popen(tensorboard_command)
                print(f"\nTensorBoard avviato in background.")

            # Define the neural network architecture
            #model_py = Sequential([
            #    Input(shape=(X_train.shape[1],), name='input'),
            #
            #    Dense(units=Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully1_'),
            #    ReLU(name='relu1'),
            #    Dropout(0.5, name='dropout1'),
            #
            #    Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully2_'),
            #    ReLU(name='relu2'),
            #    Dropout(0.5, name='dropout2'),
            #
            #    Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully3_'),
            #    ReLU(name='relu3'),
            #    Dropout(0.5, name='dropout3'),
            #
            #    Dense(units=Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully4_'),
            #])

            model_py = build_mlp(input_features, output_dim, num_layers, hidden_units_list)

            # Compile the model with SGD optimizer and mean squared error loss
            optimizer = SGD(learning_rate=1e-1, momentum=0.9)

            #model_py.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
            #model_py.compile(optimizer=optimizer, loss=mse_keras, metrics=['mse'])
            #model_py.compile(optimizer=optimizer, loss=mse_matlab, metrics=['mse'])
            model_py.compile(optimizer=optimizer, loss=mse_custom, metrics=['mse'])
            model_py.summary()

        if train_model_flag == 1:
            print("\n### train_model")

            # For the output filenames
            end_folder_Training_Size_dd_max_epochs = end_folder_Training_Size_dd + '_' + str(max_epochs_new)
            #TODO suffix
            model_type_save = 'model_py_test' + end_folder_Training_Size_dd_max_epochs + model_name_suffix
            #model_type_tb = 'model_py_test' + end_folder_Training_Size_dd # TODO ACTIVATE

            #filename_Rate_OPT_py = network_folder_out_RateDLpy + 'Rate_OPT_py_test' + end_folder_Training_Size_dd_max_epochs + '.mat'
            #filename_Rate_DL_py = network_folder_out_RateDLpy + 'Rate_DL_py_test' + end_folder_Training_Size_dd_max_epochs + '.mat'

            #print(model_py.loss)

            mini_batch_size = 500
            
            tensorboard_logs = log_dir + model_type_save
            #tensorboard_logs = log_dir + model_type_tb # TODO ACTIVATE
            tensorboard_callback = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1)

            #if Training_Size_dd < mini_batch_size:
            #    validationFrequency = Training_Size_dd
            #else:
            #    validationFrequency = int(np.floor(Training_Size_dd/mini_batch_size))
            validationFrequency = 1

            #def lr_schedule(epoch, lr):
            #    if epoch > 0 and epoch % 5 == 0: # Prima era modulo 3
            #        return lr * 0.5  # Drop learning rate by factor of 0.5 every x epochs
            #    return lr

            #lr_scheduler = LearningRateScheduler(lr_schedule)

            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=factor,         # Riduce di metà
                patience=patience,     # Numero di epoche senza miglioramento ≥ y
                min_delta=min_delta,   # Miglioramento minimo da considerare significativo
                verbose=1
            )
            
            print("\nStart DL training...")
            start_time = time.time()

            history = model_py.fit(
                xtrain, Y_train,
                validation_data=(xval, Y_val),
                #train_dataset, 
                #validation_data=val_dataset
                batch_size=mini_batch_size,
                initial_epoch=max_epochs_load,
                #epochs=max_epochs,
                epochs=max_epochs_new,
                shuffle=True,  # Shuffle data at each epoch
                callbacks=[lr_scheduler, tensorboard_callback],
                validation_freq=validationFrequency,
                verbose=2
            )

            elapsed_time = time.time() - start_time
            print(f"Training completed in {elapsed_time / 60:.2f} minutes.")

            # Save the trained model
            #The saved .keras file contains:
            # - The model's configuration (architecture)
            # - The model's weights
            # - The model's optimizer's state (if any)
            # model.save() is an alias for keras.saving.save_model()
            model_py.save(saved_models_keras + model_type_save + '.keras')  # The file needs to end with the .keras extension
            print(f"\nModel saved in {saved_models_keras}")

            #np.save(os.path.join(output_folder, 'history.npy'), history.history)
            #np.save(os.path.join(output_folder, 'Y_predicted.npy'), Y_predicted)
            #print("History and Y_predicted saved successfully.")

            # %%
            ## DL Model Prediction
            print("\n### predict_trained_model")
            test = 0
            Rate_OPT_py_val, Rate_DL_py_val  = model_predict(xdataset, Y_dataset, 
                                                             xval, Y_val, 
                                                             xtest, Y_test, 
                                                             YValidation_un_val, YValidation_un_test, 
                                                             model_py, 
                                                             network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs, 
                                                             mean_array_filepath, variance_array_filepath, test_set_size, 
                                                             test=test)
            test = 1
            Rate_OPT_py_test, Rate_DL_py_test = model_predict(xdataset, Y_dataset, 
                                                              xval, Y_val, 
                                                              xtest, Y_test, 
                                                              YValidation_un_val, YValidation_un_test, 
                                                              model_py, 
                                                              network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs, 
                                                              mean_array_filepath, variance_array_filepath, test_set_size, 
                                                              test=test)
            test = 2
            Rate_OPT_py_test, Rate_DL_py_test = model_predict(xdataset, Y_dataset, 
                                                              xval, Y_val,
                                                              xtest, Y_test, 
                                                              YValidation_un_val, YValidation_un_test, 
                                                              model_py, 
                                                              network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs, 
                                                              mean_array_filepath, variance_array_filepath, test_set_size, 
                                                              test=test)

        if Training_Size_dd >= 10000 or Training_Size_dd == 2:
            
            filename_Rate_OPT = network_folder_in + 'Rate_OPT' + end_folder_Training_Size_dd + '.mat'

            with h5py.File(filename_Rate_OPT, 'r') as f:
                Rate_OPT = f['Rate_OPT'][:][0][0]

            filename_Rate_DL = network_folder_in + 'Rate_DL' + end_folder_Training_Size_dd + '.mat'

            with h5py.File(filename_Rate_DL, 'r') as f:
                Rate_DL_valOld = f['Rate_DL'][:][0][0]

            # Output finali
            print(f"\nRate_OPT: {Rate_OPT}")
            print(f"Rate_DL_valOld: {Rate_DL_valOld}")

        # Output finali       
        try:
            if load_model_flag == 1:
                print(f"\nRate_OPT_py_load_val: {Rate_OPT_py_load_val}")
                print(f"Rate_DL_py_load_val: {Rate_DL_py_load_val}")

                print(f"\nRate_OPT_py_load_test: {Rate_OPT_py_load_test}")
                print(f"Rate_DL_py_load_test: {Rate_DL_py_load_test}")
                print(f"Rate_DL_py_load_test_tflite: {Rate_DL_py_load_test_tflite}")
                
                print(f"\nRate_OPT_py_load: {Rate_OPT_py_load}")
                print(f"Rate_DL_py_load: {Rate_DL_py_load}")

                print(f"\nRate_OPT_py_load_tflite: {Rate_OPT_py_load_tflite}")
                print(f"Rate_DL_py_load_tflite: {Rate_DL_py_load_tflite}")
            if train_model_flag == 1:
                print(f"\nRate_OPT_py_test: {Rate_OPT_py_test}")
                print(f"\nRate_DL_py_test: {Rate_DL_py_test}")
                print(f"Rate_DL_py_val: {Rate_DL_py_val}")
        except Exception as e:
            print('Some missing info to print, no problem')
        
        print(f"\nRicorda che Rate_DL_valOld non può essere confrontato con Rate_DL_py_load_val e Rate_DL_py_load_test, perchè il primo è calcolato sul validation set old, mentre i secondi sul val e test set nuovi")

        tf.keras.backend.clear_session()

        return model_py, \
               Rate_OPT_py_load_val,    Rate_DL_py_load_val, \
               Rate_OPT_py_load_test,   Rate_DL_py_load_test, \
               Rate_DL_py_load_test_tflite, \
               Indmax_OPT_py_load_test, Indmax_DL_py_load_test, \
               Indmax_DL_py_load_test_tflite, \
               YValidation_un_test
