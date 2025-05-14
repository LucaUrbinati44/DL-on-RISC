# %%
import os, sys
#import logging
import random
import numpy as np

#import absl.logging
#absl.logging.set_verbosity(absl.logging.ERROR)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Sopprime INFO, WARNING ed ERROR

#tf.get_logger().setLevel('ERROR')  # Nasconde i messaggi di logging di TensorFlow
#logging.disable(logging.WARNING)
#logging.getLogger('tensorflow').disabled = True

#sys.stderr = open(os.devnull, 'w')

import tf2onnx
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
from tensorflow.keras.layers import Input, Normalization, Dense, Dropout, ReLU, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ProgbarLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import h5py
import sys
import importlib
import time
from datetime import datetime
import subprocess
import psutil 

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

# %%
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

# %% Define functions

def model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test, save_files=1):

    if test == 2:
        t = ''
        x = xdataset
        y = Y_dataset

        print('x.shape:', x.shape)
        print('y.shape:', y.shape)

        filename_Indmax_OPT_py = network_folder_out_RateDLpy + 'Indmax_OPT_py' + end_folder_Training_Size_dd_max_epochs_load + '.mat'
        filename_Indmax_DL_py = network_folder_out_RateDLpy + 'Indmax_DL_py' + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    elif test == 1:
        t = 'test'
        x = xtest
        y = Y_test
        YValidation_un = YValidation_un_test

        filename_Indmax_OPT_py = network_folder_out_RateDLpy + 'Indmax_OPT_py_test' + end_folder_Training_Size_dd_max_epochs_load + '.mat'
        filename_Indmax_DL_py = network_folder_out_RateDLpy + 'Indmax_DL_py_test' + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    elif test == 0:
        t = 'val'
        x = xval
        y = Y_val
        YValidation_un = YValidation_un_val
        
    filename_MaxR_OPT_py = network_folder_out_RateDLpy + 'MaxR_OPT_py_'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    filename_MaxR_DL_py = network_folder_out_RateDLpy + 'MaxR_DL_py_'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    filename_Rate_OPT_py = network_folder_out_RateDLpy + 'Rate_OPT_py_'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    filename_Rate_DL_py = network_folder_out_RateDLpy + 'Rate_DL_py_'+t + end_folder_Training_Size_dd_max_epochs_load + '.mat'
        
    print(f"\nStart DL prediction {t} set...")

    Indmax_OPT_py = np.argmax(y, axis=1)
    print(f'Indmax_OPT_py.shape: {Indmax_OPT_py.shape}')
    print(f'np.min(Indmax_OPT_py): {np.min(Indmax_OPT_py)}')
    print(f'np.max(Indmax_OPT_py): {np.max(Indmax_OPT_py)}')
    print(Indmax_OPT_py[0:5])

    YPredicted = model_py.predict(x, verbose=1, batch_size=128)
    print(f'x.shape: {x.shape}')  # (3100, 1024)
    print(f'YPredicted.shape: {YPredicted.shape}')

    Indmax_DL_py = np.argmax(YPredicted, axis=1)
    print(f'Indmax_DL_py.shape: {Indmax_DL_py.shape}')
    # Questi devono essere numeri interi
    print(f'np.min(Indmax_DL_py): {np.min(Indmax_DL_py)}')
    print(f'np.max(Indmax_DL_py): {np.max(Indmax_DL_py)}')
    print(Indmax_DL_py[0:5])

    if test == 0 or test == 1:
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


    if test == 1 or test == 2:
        with h5py.File(filename_Indmax_OPT_py, 'w') as f:
            f.create_dataset('Indmax_OPT_py', data=Indmax_OPT_py)
            print(f"\Indmax_OPT_py saved in {filename_Indmax_OPT_py}")

        with h5py.File(filename_Indmax_DL_py, 'w') as f:
            f.create_dataset('Indmax_DL_py', data=Indmax_DL_py)
            print(f"\Indmax_DL_py saved in {filename_Indmax_DL_py}")

    if save_files == 1:
        # Scrittura in formato HDF5 (compatibile MATLAB v7.3)
        with h5py.File(filename_MaxR_OPT_py, 'w') as f:
            f.create_dataset('MaxR_OPT_py', data=MaxR_OPT_py)
            print(f"\MaxR_OPT_py saved in {filename_MaxR_OPT_py}")
        
        with h5py.File(filename_MaxR_DL_py, 'w') as f:
            f.create_dataset('MaxR_DL_py', data=MaxR_DL_py)
            print(f"\MaxR_DL_py saved in {filename_MaxR_DL_py}")

        with h5py.File(filename_Rate_DL_py, 'w') as f:
            f.create_dataset('Rate_DL_py', data=Rate_DL_py)
            print(f"\Rate_DL_py saved in {filename_Rate_DL_py}")

        with h5py.File(filename_Rate_OPT_py, 'w') as f:
            f.create_dataset('Rate_OPT_py', data=Rate_OPT_py)
            print(f"\Rate_OPT_py saved in {filename_Rate_OPT_py}")
        
    return Rate_OPT_py, Rate_DL_py

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
network_folder_out_YPredicted = output_folder + 'Neural_Network/YPredicted/'
network_folder_out_RateDLpy = output_folder + 'Neural_Network/RateDLpy/'
saved_models_keras = network_folder_out + 'saved_models_keras/'
#saved_models_onnx = network_folder_out + 'saved_models_onnx/'
figure_folder = output_folder + 'Figures/'

import os

folders = [
    output_folder,
    network_folder_out,
    network_folder_out_YPredicted,
    network_folder_out_RateDLpy,
    saved_models_keras,
    #saved_models_onnx,
    figure_folder
]

for folder in folders:
    if not os.path.exists(folder):  # Controlla se la cartella esiste
        os.makedirs(folder, exist_ok=True)  # Crea la cartella se non esiste
        print(f"\nCartella creata: {folder}")
    #else:
    #    print(f"La cartella esiste già: {folder}")

# %%
#My_ar = [32, 64]
#Mz_ar = [32, 64]
#My_ar = [32]
#Mz_ar = [32]
#My_ar = [64]
#Mz_ar = [64]
Mx = 1

M_bar=8
Ur_rows = [1000, 1200]
#              0    1      2      3      4      5      6
#Training_Size=[2, 10000, 14000, 18000, 22000, 26000, 30000]
#Training_Size=[2, 10000, 14000]
#Training_Size=[18000, 22000, 26000, 30000]
#Training_Size=[30000]


#load_model_flag = 1
#max_epochs_load = 60

#train_model_flag = 1
max_epochs = 20

import argparse

# python DL_training_4_v2_test_temp.py --My --Mz --max_epochs_load --Training_Size

# Aggiungi il parser degli argomenti
parser = argparse.ArgumentParser(description='Script per il training e il testing del modello DL.')
parser.add_argument('--My', type=int, default=32, help='Numero di antenne RIS lungo y (32, 64).')
parser.add_argument('--Mz', type=int, default=32, help='Numero di antenne RIS lungo z (32, 64).')
parser.add_argument('--load_model_flag', type=int, default=1, help='Flag per caricare un modello pre-addestrato (1) o no (0).')
parser.add_argument('--max_epochs_load', type=int, default=20, help='Numero di epoche del modello da carica (20, 40, 60, 80).')
parser.add_argument('--train_model_flag', type=int, default=0, help='Flag per addestrare il modello (1) o no (0).')
parser.add_argument('--Training_Size', type=int, default=10000, help='Training_Size (10000, 14000, 18000, 22000, 26000, 30000).')

# Leggi gli argomenti dalla riga di comando
args = parser.parse_args()

# Assegna i valori degli argomenti alle variabili
My_ar = [args.My]
Mz_ar = [args.Mz]
load_model_flag = args.load_model_flag
max_epochs_load = args.max_epochs_load
train_model_flag = args.train_model_flag
Training_Size = [args.Training_Size]

# %%

# count, value
for i, ris in enumerate(My_ar):
    #ris = 32

    My = ris
    Mz = ris
    print(f"\nRIS: {My}x{Mz}")

    for j, Training_Size_dd in enumerate(Training_Size):
        #Training_Size_dd = Training_Size[1]

        end_folder = '_seed' + str(seed) + '_grid' + str(Ur_rows[1]) + '_M' + str(My) + str(Mz) + '_Mbar' + str(M_bar)
        end_folder_Training_Size_dd = end_folder + '_' + str(Training_Size_dd)

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
        ## Dataset split originale

        # Flatten the input and output arrays if necessary
        #X = DL_input_reshaped.reshape(DL_input_reshaped.shape[0], -1).astype(np.float32)
        #Y = DL_output_reshaped.reshape(DL_output_reshaped.shape[0], -1).astype(np.float32)

        # Split the dataset into training and validation sets
        #X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_size / (training_size + validation_size), shuffle=False, random_state=seed)

        #RandP_all2 = np.squeeze(np.array(RandP_all.astype(int))) - 1

        #Training_Ind = RandP_all2[0:Training_Size_dd]

        #Validation_Size = 6200
        #Validation_Ind = RandP_all2[-Validation_Size:]

        #print(Training_Ind.shape)
        #print(Validation_Ind.shape)

        #X_train = np.array(DL_input_reshaped[Training_Ind, :, :, :], dtype=force_datatype).squeeze()
        #Y_train = np.array(DL_output_reshaped[Training_Ind, :, :, :], dtype=force_datatype).squeeze()
        #X_val = np.array(DL_input_reshaped[Validation_Ind, :, :, :], dtype=force_datatype).squeeze()
        #Y_val = np.array(DL_output_reshaped[Validation_Ind, :, :, :], dtype=force_datatype).squeeze()

        #print(f"\nY_train.dtype: {Y_train.dtype}")

        #print(X_train.shape)
        #print(Y_train.shape)
        #print(X_val.shape)
        #print(Y_val.shape)



        ## %%
        ### Dataset split nuovo con test set, ipotesi di split 80-20

        #dataset_size = 36200

        #perc = 0.8
        #Dev_Size = round(perc * dataset_size)
        #Test_Size = round((1-perc) * dataset_size)
        #Validation_Size = Test_Size
        #Train_Size = Dev_Size - Validation_Size

        #print(Dev_Size)
        #print(Train_Size)
        #print(Validation_Size)
        #print(Test_Size)

        #perc_train = Train_Size/Dev_Size
        #perc_val = Validation_Size/Dev_Size
        #print(perc_train)
        #print(perc_val)

        #RandP_all2 = np.squeeze(np.array(RandP_all.astype(int))) - 1

        #Training_Ind = RandP_all2[0:Training_Size_dd]
        #Test_Ind = RandP_all2[-Test_Size:]
        #Validation_Ind = RandP_all2[-(Validation_Size+Test_Size):-Test_Size]

        #print(Training_Ind.shape)
        #print(Validation_Ind.shape)
        #print(Test_Ind.shape)


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

        normalized = 1

        #print(X_train[0][0:5])
        #print(mean_array[0:5])
        #print(X_train_normalized[0][0:5])

        ## DL Model Definition

        if normalized == 1:
            xtrain = X_train_normalized
            xval = X_val_normalized
            xtest = X_test_normalized
            xdataset = X_dataset_normalized
        else:
            xtrain = X_train
            xval = X_val
            xtest = X_test

        # %%
        ## DL Model Training and Prediction

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

        # ------------------ DL Model Prediction ------------------ #
        if load_model_flag == 1:

            end_folder_Training_Size_dd_max_epochs_load = end_folder_Training_Size_dd + '_' + str(max_epochs_load)
            model_type_load = 'model_py_test' + end_folder_Training_Size_dd_max_epochs_load

            model_path_keras = saved_models_keras + model_type_load + '.keras'
            model_py = load_model(model_path_keras, custom_objects={'mse_custom': mse_custom})
            model_py.summary()
            print(f"\nModel {model_path_keras} loaded")

            # Esporta in formato tf (più compatibile con ONNX)
            #export_dir = saved_models_onnx + model_type_load
            #model_py.export(export_dir)
        
            # Esporta in formato h5
            #model_py.save(saved_models_onnx + model_type_load + '.h5')

            #os._exit(0)

            #model_path_onnx = saved_models_onnx + model_type_load + '.onnx'
            ## Converte in ONNX
            #spec = (tf.TensorSpec(model_py.inputs[0].shape, tf.float32, name="input"),)
            #model_py.output_names=['output'] # Dummy assignment to make tf2onnx work https://github.com/onnx/tensorflow-onnx/issues/2319#issuecomment-2009332333
            #onnx_model, _ = tf2onnx.convert.from_keras(model_py, input_signature=spec, opset=13, output_path=model_path_onnx)
            #print(f"\nModel {model_path_onnx} saved")
            #os._exit(0)

            save_files_flag = 0
            test = 0
            #Rate_OPT_py_load_val, Rate_DL_py_load_val   = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
            Rate_OPT_py_load_val = 0
            Rate_DL_py_load_val = 0
            test = 1
            #Rate_OPT_py_load_test, Rate_DL_py_load_test = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
            Rate_OPT_py_load_test = 0
            Rate_DL_py_load_test = 0
            test = 2
            Rate_OPT_py_load, Rate_DL_py_load = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
            
            learning_rate = model_py.optimizer.learning_rate.numpy()
            print(f"Learning rate loaded model: {learning_rate}")

        # ------------------ DL Model Training ------------------ #
        if train_model_flag == 1 and load_model_flag == 0:

            # Define the neural network architecture
            model_py = Sequential([
                Input(shape=(X_train.shape[1],), name='input'),

                Dense(units=Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully1_'),
                ReLU(name='relu1'),
                Dropout(0.5, name='dropout1'),

                Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully2_'),
                ReLU(name='relu2'),
                Dropout(0.5, name='dropout2'),

                Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully3_'),
                ReLU(name='relu3'),
                Dropout(0.5, name='dropout3'),

                Dense(units=Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully4_'),
            ])

            # Compile the model with SGD optimizer and mean squared error loss
            optimizer = SGD(learning_rate=1e-1, momentum=0.9)

            #model_py.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
            #model_py.compile(optimizer=optimizer, loss=mse_keras, metrics=['mse'])
            #model_py.compile(optimizer=optimizer, loss=mse_matlab, metrics=['mse'])
            model_py.compile(optimizer=optimizer, loss=mse_custom, metrics=['mse'])
            model_py.summary()

        if train_model_flag == 1:

            # For the output filenames
            end_folder_Training_Size_dd_max_epochs = end_folder_Training_Size_dd + '_' + str(max_epochs_new)
            model_type = 'model_py_test' + end_folder_Training_Size_dd_max_epochs
            #model_type_tb = 'model_py_test' + end_folder_Training_Size_dd # TODO ACTIVATE

            filename_Rate_OPT_py = network_folder_out_RateDLpy + 'Rate_OPT_py_test' + end_folder_Training_Size_dd_max_epochs + '.mat'
            filename_Rate_DL_py = network_folder_out_RateDLpy + 'Rate_DL_py_test' + end_folder_Training_Size_dd_max_epochs + '.mat'

            #print(model_py.loss)

            # ------------------ Training Options ------------------ #
            mini_batch_size = 500
            
            tensorboard_logs = log_dir + model_type
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
            model_py.save(saved_models_keras + model_type + '.keras')  # The file needs to end with the .keras extension
            print(f"\nModel saved in {saved_models_keras}")

            # TODO copiare saved models onnx anche qui

            #np.save(os.path.join(output_folder, 'history.npy'), history.history)
            #np.save(os.path.join(output_folder, 'Y_predicted.npy'), Y_predicted)
            #print("History and Y_predicted saved successfully.")

            # %%
            ## DL Model Prediction
            # # SOSTITUIRE X_val CON X_test!!!
            test = 0
            Rate_OPT_py_val, Rate_DL_py_val  = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs, test=test)
            test = 1
            Rate_OPT_py_test, Rate_DL_py_test = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs, test=test)
            test = 2
            Rate_OPT_py_test, Rate_DL_py_test = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs, test=test)

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
        if load_model_flag == 1:
            print(f"\nRate_OPT_py_load_val: {Rate_OPT_py_load_val}")
            print(f"Rate_DL_py_load_val: {Rate_DL_py_load_val}")

            print(f"\nRate_OPT_py_load_test: {Rate_OPT_py_load_test}")
            print(f"Rate_DL_py_load_test: {Rate_DL_py_load_test}")
            
            print(f"\nRate_OPT_py_load: {Rate_OPT_py_load}")
            print(f"Rate_DL_py_load: {Rate_DL_py_load}")
        if train_model_flag == 1:
            print(f"\nRate_DL_py_test: {Rate_DL_py_test}")
            print(f"Rate_DL_py_val: {Rate_DL_py_val}")
        
        print(f"\nRicorda che Rate_DL_valOld non può essere confrontato con Rate_DL_py_load_val e Rate_DL_py_load_test, perchè il primo è calcolato sul validation set old, mentre i secondi sul val e test set nuovi")

        tf.keras.backend.clear_session()