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
from tensorflow.keras.layers import Input, Dense, Dropout, ReLU, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
# Load the LiteRT model and allocate tensors.
from ai_edge_litert.interpreter import Interpreter

import h5py
import time
from datetime import datetime
import subprocess
import psutil 

import json
import csv

import io
from contextlib import redirect_stdout

import edgeimpulse as ei
ei.API_KEY = "ei_4a58ede09ec501541b8239002c9ee96833f9fac84c339500a2a04eb3b7c9bcfc"
API_KEY = "ei_4a58ede09ec501541b8239002c9ee96833f9fac84c339500a2a04eb3b7c9bcfc"
PROJECT_ID = 712872

HEADERS = {
    "x-api-key": API_KEY
}

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

def model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test, save_files=1):

    if test == 4: # entire dataset prediction (for figC)
        t = '_tflite'
        x = xdataset
        y = Y_dataset
    elif test == 3:
        t = '_test_tflite'
        x = xtest
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
        #Loading and running a LiteRT model involves the following steps:
        #1) Loading the model into memory.
        #2) Building an Interpreter based on an existing model.
        # Load the LiteRT model and allocate tensors.
        interpreter = Interpreter(model_content=model_py)
        interpreter.allocate_tensors()

        #3) Setting input tensor values.
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_dtype = input_details[0]['dtype'] #int8
        output_dtype = input_details[0]['dtype']
        #print('input_dtype:', input_dtype) # [   1 1024]
        #print('output_dtype:', output_dtype)
        input_shape = input_details[0]['shape']
        output_shape = output_details[0]['shape']
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
            sample = np.expand_dims(sample, axis=0)  # shape: (1, 1024)
            input_int8 = np.round(sample / input_scale + input_zero_point).astype(np.int8)
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
            #print('output_scale:', output_scale)
            #print('output_zero_point:', output_zero_point)
            output_deq = (output_int8.astype(np.float32) - output_zero_point) * output_scale
            #print('output_deq.shape:', output_deq.shape)
            #print('np.max(output_deq):', np.max(output_deq))
            #print('np.min(output_deq):', np.min(output_deq))
            output_float.append(output_deq[0])  # output_deq.shape: (1, 1024) -> prendi [0]

        YPredicted = np.stack(output_float, axis=0)  # shape: (3100, 1024)

    print(f'x.shape: {x.shape}')  # (3100, 1024)
    print(f'YPredicted.shape: {YPredicted.shape}')
    #print('np.max(YPredicted):', np.max(YPredicted))
    #print('np.min(YPredicted):', np.min(YPredicted))

    Indmax_DL_py = np.argmax(YPredicted, axis=1)
    print(f'Indmax_DL_py.shape: {Indmax_DL_py.shape}')
    # Questi devono essere numeri interi
    print(f'np.min(Indmax_DL_py): {np.min(Indmax_DL_py)}')
    print(f'np.max(Indmax_DL_py): {np.max(Indmax_DL_py)}')
    print(Indmax_DL_py[0:5])

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

        if 0 == 1: # TODO: temporaneamente non voglio salvarli perchè già salvati in precedenza
            with h5py.File(filename_Indmax_OPT_py, 'w') as f:
                f.create_dataset('Indmax_OPT_py', data=Indmax_OPT_py)
                print(f"\nIndmax_OPT_py saved in {filename_Indmax_OPT_py}")

            with h5py.File(filename_MaxR_OPT_py, 'w') as f:
                f.create_dataset('MaxR_OPT_py', data=MaxR_OPT_py)
                print(f"\nMaxR_OPT_py saved in {filename_MaxR_OPT_py}")

            with h5py.File(filename_Rate_OPT_py, 'w') as f:
                f.create_dataset('Rate_OPT_py', data=Rate_OPT_py)
                print(f"\nRate_OPT_py saved in {filename_Rate_OPT_py}")

        if test == 3 or test == 4:
            with h5py.File(filename_Indmax_DL_py, 'w') as f:
                f.create_dataset('Indmax_DL_py', data=Indmax_DL_py)
                print(f"\nIndmax_DL_py saved in {filename_Indmax_DL_py}")
            
        if test == 3:
            with h5py.File(filename_MaxR_DL_py, 'w') as f:
                f.create_dataset('MaxR_DL_py', data=MaxR_DL_py)
                print(f"\nMaxR_DL_py saved in {filename_MaxR_DL_py}")

            with h5py.File(filename_Rate_DL_py, 'w') as f:
                f.create_dataset('Rate_DL_py', data=Rate_DL_py)
                print(f"\nRate_DL_py saved in {filename_Rate_DL_py}")
        
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
network_folder_out_RateDLpy_TFLite = output_folder + 'Neural_Network/RateDLpy_TFLite/'
saved_models_keras = network_folder_out + 'saved_models_keras/'
saved_models_onnx = network_folder_out + 'saved_models_onnx/'
saved_models_tfsaved = network_folder_out + 'saved_models_tfsaved/'
saved_models_tfsaved2 = network_folder_out + 'saved_models_tfsaved2/'
saved_models_tflite = network_folder_out + 'saved_models_tflite/'
saved_models_edgeimpulse = network_folder_out + 'saved_models_edgeimpulse/'
figure_folder = output_folder + 'Figures/'
test_data_npy_path = output_folder + 'Test_data/'

import os

folders = [
    output_folder,
    network_folder_out,
    network_folder_out_YPredicted,
    network_folder_out_RateDLpy,
    network_folder_out_RateDLpy_TFLite,
    saved_models_keras,
    saved_models_onnx,
    saved_models_tfsaved,
    saved_models_tfsaved2,
    saved_models_tflite,
    saved_models_edgeimpulse,
    figure_folder,
    test_data_npy_path
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
parser.add_argument('--predict_loaded_model_flag', type=int, default=0, help='Flag per fare predizione (1) o no (0).')
parser.add_argument('--export_model_flag', type=int, default=0, help='Flag per esportare il modello (1) o no (0).')
parser.add_argument('--profiling_model_flag', type=int, default=0, help='Flag per fare il profiling del modello (1) o no (0).')
parser.add_argument('--Training_Size', type=int, default=10000, help='Training_Size (10000, 14000, 18000, 22000, 26000, 30000).')

# Leggi gli argomenti dalla riga di comando
args = parser.parse_args()

# Assegna i valori degli argomenti alle variabili
My_ar = [args.My]
Mz_ar = [args.Mz]
load_model_flag = args.load_model_flag
max_epochs_load = args.max_epochs_load
train_model_flag = args.train_model_flag
predict_loaded_model_flag = args.predict_loaded_model_flag
export_model_flag = args.export_model_flag
profiling_model_flag = args.profiling_model_flag
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

        print("Save test data to npy")
        xtest_npy_filename = test_data_npy_path + 'test_set' + end_folder_Training_Size_dd + '.npy'
        np.save(xtest_npy_filename, xtest)

        #os._exit(0)

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

        # %%
        ################################ Load Model ################################
        if load_model_flag == 1:

            end_folder_Training_Size_dd_max_epochs_load = end_folder_Training_Size_dd + '_' + str(max_epochs_load)
            model_type_load = 'model_py_test' + end_folder_Training_Size_dd_max_epochs_load

            model_path_keras = saved_models_keras + model_type_load + '.keras'
            model_py = load_model(model_path_keras, custom_objects={'mse_custom': mse_custom})
            model_py.summary()
            print(f"\nModel {model_path_keras} loaded")

            model_path_tflite = saved_models_tflite + model_type_load + '_quant.tflite'
            model_path_float32_onnx = saved_models_onnx + model_type_load + '.onnx'

            MODEL_PATH = model_path_tflite
        
            if export_model_flag == 1:
                #### V1 ####
                ## Esporta in formato tf
                #export_dir = saved_models_tfsaved + model_type_load
                #model_py.export(export_dir, format="tf_saved_model")

                #### V2 ####
                ## Convert the model for Edge Impulse
                #model_py2 = Sequential([
                #    Input(shape=(X_train.shape[1],), batch_size=1, name='input'),
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

                #model_py2.load_weights(model_path_keras)

                ## Esporta in formato tf
                #export_dir = saved_models_tfsaved2 + model_type_load
                #model_py2.export(export_dir, format="tf_saved_model")

                #### V3 ####
                converter = tf.lite.TFLiteConverter.from_keras_model(model_py)
                tflite_model = converter.convert()

                #### V4 ####
                def representative_dataset():
                    for i in range(xval.shape[0]): # (3100, 1024)
                        data = xval[i,:]
                        yield [data.astype(np.float32)]
                
                # Enforce full integer quantization for all ops including the input and output
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8  # or tf.uint8
                converter.inference_output_type = tf.int8  # or tf.uint8
                tflite_quant_model = converter.convert()

                ## Enforce full integer quantization for all ops excluding input and output and those operators that don't have an integer implementation
                #converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
                #converter.optimizations = [tf.lite.Optimize.DEFAULT]
                #converter.representative_dataset = representative_dataset_gen
                #tflite_quant_model = converter.convert()

                #model_path_tflite = saved_models_tflite + model_type_load + '.tflite'
                #with open(model_path_tflite, 'wb') as f:
                #    f.write(tflite_model)
                
                #export_dir = saved_models_tfsaved + model_type_load
                #export_val_dataset = export_dir + '_val_dataset.npy'
                #np.save(export_val_dataset, xval)
            
                with open(model_path_tflite, 'wb') as f:
                    f.write(tflite_quant_model)

                #### V5 ####
                model_py.export(model_path_float32_onnx, format="onnx")

            else:
                with open(model_path_tflite, 'rb') as f:
                    tflite_quant_model = f.read()
                print('TFLite model loaded')

            # %%
            ################################ Profile Model ################################

            if profiling_model_flag == 1:
                
                def extract_json_blocks(filepath):
                    blocks = []
                    with open(filepath, 'r') as f:
                        buffer = []
                        brace_count = 0
                        for line in f:
                            line_strip = line.strip()
                            # Salta righe vuote o di testo
                            if not line_strip or line_strip.startswith('=') or line_strip.endswith(':'):
                                continue
                            # Se la riga contiene una parentesi graffa, aggiorna il contatore
                            brace_count += line.count('{')
                            brace_count -= line.count('}')
                            if brace_count > 0 or (brace_count == 0 and '{' in line):
                                buffer.append(line)
                            elif buffer:
                                buffer.append(line)
                            # Quando brace_count torna a zero e buffer non è vuoto, abbiamo un blocco JSON completo
                            if brace_count == 0 and buffer:
                                json_str = ''.join(buffer)
                                try:
                                    blocks.append(json.loads(json_str))
                                except Exception as e:
                                    print("Errore parsing JSON:", e)
                                    # Stampa il blocco che ha causato l'errore per debug
                                    print(json_str)
                                buffer = []
                    
                    first_json = blocks[0]
                    second_json = blocks[1]

                    return first_json, second_json
                                        
                #print(ei.model.list_profile_devices())
                # Solitamente si usano cortex-m4f-80mhz, cortex-m7-216mhz
                #print(ei.model.list_deployment_targets())
                # Solitamente si usa "zip" che è il c++ genirico che si può compilare con tutto

                debug = 1
                deploy = 0

                saved_models_edgeimpulse_model_type_load = saved_models_edgeimpulse + model_type_load + '/'
                folder = saved_models_edgeimpulse_model_type_load
                if not os.path.exists(folder):  # Controlla se la cartella esiste
                    os.makedirs(folder, exist_ok=True)  # Crea la cartella se non esiste
                    print(f"\nCartella creata: {folder}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                logfilename = saved_models_edgeimpulse_model_type_load + 'log_profiling_' + timestamp + '.txt'
                #print(f'Start profiling.\nConsole output redirected to {logfilename}')
                #logfile = open(logfilename, "w")
                #sys.stdout = logfile
                #sys.stderr = logfile  # opzionale: salva anche gli errori

                #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                #export_metrics_all = saved_models_edgeimpulse + model_type_load + f'_metrics_{timestamp}.txt'
                
                ## Se il file esiste, eliminalo
                #if os.path.exists(export_metrics_all):
                #    os.remove(export_metrics_all)
                #    print(f"File {export_metrics_all} removed.")
                
                #export_metrics_all = saved_models_edgeimpulse + model_type_load + f'_metrics_.txt'
                #with open(export_metrics_all, 'a') as f:
                #    #f.write("quant_type,device_type,ram,rom,arena,ram_eon,rom_eon,arena_eon,lat,isSupportedOnMcu,hasPerformance\n")
                #    f.write("device_type,quant_type,ram_eon,rom_eon,lat\n")
                
                #for quant_type, model_to_profile in zip(['int8','float32'], [model_path_tflite, model_path_float32_onnx]):
                for quant_type, model_to_profile in zip(['int8'], [model_path_tflite]):

                    metrics_outputpath_merge = saved_models_edgeimpulse_model_type_load + f"metrics_merge_{quant_type}.csv"
                    metrics_outputpath_mean_list = []

                    # https://docs.edgeimpulse.com/docs/edge-ai-hardware/edge-ai-hardware
                    # Non funzionano: 'st-stm32n6', 
                    #                           M0+                    M4                   M7                    M55+acc                A72            cpu+gpu
                    #for device_type in ['raspberry-pi-rp2040', 'cortex-m4f-80mhz', 'cortex-m7-216mhz', 'arduino-nicla-vision-m4', 'raspberry-pi-4', 'jetson-nano']: 
                    # 'st-stm32n6', 'jetson-nano', 'jetson-orin-nano', 'alif-he', 'alif-hp' non vanno
                    for device_type in ['raspberry-pi-rp2040', 'cortex-m4f-80mhz', 'cortex-m7-216mhz', 'arduino-nicla-vision-m4', 'raspberry-pi-4']: 

                        metrics_outputpath_target = saved_models_edgeimpulse_model_type_load + f"metrics_{device_type}_{quant_type}_target.csv"
                        metrics_outputpath_mean = saved_models_edgeimpulse_model_type_load + f"metrics_{device_type}_{quant_type}.csv"
                        metrics_outputpath_mean_list.append(metrics_outputpath_mean)
                        metrics_outputpath_lowEndMcu = saved_models_edgeimpulse_model_type_load + f"metrics_{device_type}_{quant_type}_lowEndMcu.csv"
                        metrics_outputpath_highEndMcu = saved_models_edgeimpulse_model_type_load + f"metrics_{device_type}_{quant_type}_highEndMcu.csv"
                        metrics_outputpath_highEndMcuPlusAccelerator = saved_models_edgeimpulse_model_type_load + f"metrics_{device_type}_{quant_type}_highEndMcuPlusAccelerator.csv"
                        metrics_outputpath_mpu = saved_models_edgeimpulse_model_type_load + f"metrics_{device_type}_{quant_type}_mpu.csv"
                        metrics_outputpath_gpuOrMpuAccelerator = saved_models_edgeimpulse_model_type_load + f"metrics_{device_type}_{quant_type}_gpuOrMpuAccelerator.csv"
                        metrics_outputpath_list = [metrics_outputpath_target, metrics_outputpath_lowEndMcu, metrics_outputpath_highEndMcu,
                                                    metrics_outputpath_highEndMcuPlusAccelerator, metrics_outputpath_mpu, metrics_outputpath_gpuOrMpuAccelerator]
                        metrics_device_list = [device_type, 'lowEndMcu', 'highEndMcu', 'highEndMcuPlusAccelerator', 'mpu', 'gpuOrMpuAccelerator']

                        header_string = "rep,device_type,quant_type,ram_eon[KB],rom_eon[MB],lat[ms]\n"
                        for filpath in metrics_outputpath_list:
                            with open(filpath, 'w') as f:
                                #f.write("quant_type,device_type,ram,rom,arena,ram_eon,rom_eon,arena_eon,lat,isSupportedOnMcu,hasPerformance\n")
                                f.write(header_string)

                        repetitions_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        for rep in repetitions_list:
                        #for rep in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        #for rep in [0]:

                            #### Edge Impulse Profile with Python SDK ####
                            print(f"\n*** Edge Impulse Profiling of {model_to_profile}: {device_type}, {quant_type}, rep: {rep} ***")
                            t = time.time()

                            profiling_outputpath = saved_models_edgeimpulse_model_type_load + f"profiling_{device_type}_{quant_type}_{rep}.json"

                            if debug == 0:
                                # Estimate the RAM, ROM, and inference time for our model on the target hardawre
                                #The response includes estimates of memory usage and latency for the model across a range of
                                #targets, including low-end MCU, high-end MCU, high-end MCU with accelerator, microprocessor unit
                                #(MPU), and a GPU or neural network accelerator. It will also include details of any conditions
                                #that preclude operation on a given type of device.
                                #If you request a specific `device`, the results will also include estimates for that specific
                                #device. A list of devices can be obtained from `edgeimpulse.model.list_profile_devices()`.
                                #You can call `.summary()` on the response to obtain a more readable version of the most relevant
                                #information.
                                try:
                                    profile = ei.model.profile(model=model_to_profile, 
                                                                device=device_type)
                                    print(profile.summary())
                                except Exception as e:
                                    print(f"Could not profile: {e}")

                                # Salva il risultato su file JSON
                                with open(profiling_outputpath, "w") as f:
                                    buf = io.StringIO()
                                    with redirect_stdout(buf):
                                        profile.summary()
                                    f.write(buf.getvalue())

                            [first_json, second_json] = extract_json_blocks(profiling_outputpath)

                            #for i, block in enumerate(json_blocks):
                            #    print(f"\nBlocco JSON {i+1}:")
                            #    for key, value in block.items():
                            #        print(f"{key}: {value}")

                            ram_list = []
                            rom_list = []
                            lat_list = []

                            #try:
                            try:
                                ram_list.append(round(first_json['memory']['eon']['ram']/1024+0.005, 2))
                            except Exception as e:
                                ram_list.append(-1)
                            try:
                                rom_list.append(round(first_json['memory']['eon']['rom']/1024/1024+0.005, 2))
                            except Exception as e:
                                rom_list.append(-1)
                            #if first_json['hasPerformance'] == 'true':
                            try:
                                lat_list.append(first_json['timePerInferenceMs'])
                            except Exception as e:
                                lat_list.append(-1)

                            try:
                                ram_list.append(round(second_json['lowEndMcu']['memory']['eon']['ram']/1024+0.005, 2))
                            except Exception as e:
                                ram_list.append(-1)
                            try:
                                rom_list.append(round(second_json['lowEndMcu']['memory']['eon']['rom']/1024/1024+0.005, 2))
                            except Exception as e:
                                rom_list.append(-1)
                            try:
                                lat_list.append(second_json['lowEndMcu']['timePerInferenceMs'])
                            except Exception as e:
                                lat_list.append(-1)
                            
                            try:
                                ram_list.append(round(second_json['highEndMcu']['memory']['eon']['ram']/1024+0.005, 2))
                            except Exception as e:
                                ram_list.append(-1)
                            try:
                                rom_list.append(round(second_json['highEndMcu']['memory']['eon']['rom']/1024/1024+0.005, 2))
                            except Exception as e:
                                rom_list.append(-1)
                            try:
                                lat_list.append(second_json['highEndMcu']['timePerInferenceMs'])
                            except Exception as e:
                                lat_list.append(-1)

                            try:
                                ram_list.append(round(second_json['highEndMcuPlusAccelerator']['memory']['eon']['ram']/1024+0.005, 2))
                            except Exception as e:
                                ram_list.append(-1)
                            try:
                                rom_list.append(round(second_json['highEndMcuPlusAccelerator']['memory']['eon']['rom']/1024/1024+0.005, 2))
                            except Exception as e:
                                rom_list.append(-1)
                            try:
                                lat_list.append(second_json['highEndMcuPlusAccelerator']['timePerInferenceMs'])
                            except Exception as e:
                                lat_list.append(-1)

                            ram_list.append(-1)
                            try:
                                rom_list.append(round(second_json['mpu']['rom']/1024/1024+0.005, 2))
                            except Exception as e:
                                rom_list.append(-1)
                            try:
                                lat_list.append(second_json['mpu']['timePerInferenceMs'])
                            except Exception as e:
                                lat_list.append(-1)

                            ram_list.append(-1)
                            try:
                                rom_list.append(round(second_json['gpuOrMpuAccelerator']['rom']/1024/1024+0.005, 2))
                            except Exception as e:
                                rom_list.append(-1)
                            try:
                                lat_list.append(second_json['gpuOrMpuAccelerator']['timePerInferenceMs'])
                            except Exception as e:
                                lat_list.append(-1)

                            #except Exception as e:
                            #    print(f"Error getting profile results: {e}")


                            for filpath, device, ram, rom, lat in zip(metrics_outputpath_list, metrics_device_list, ram_list, rom_list, lat_list):
                                with open(filpath, 'a') as f:
                                    #"rep,device_type,quant_type,ram_eon[KB],rom_eon[MB],lat[ms]\n"
                                    f.write(f"{rep},{device},{quant_type},{ram},{rom},{lat}\n")
                                
                                if rep == len(repetitions_list)-1:
                                    with open(filpath, mode='r', newline='') as csvfile:
                                        reader = csv.DictReader(csvfile)
                                        lat_values = []
                                        row_template = None

                                        for row in reader:
                                            print(row['lat[ms]'])
                                            lat = int(row['lat[ms]'])
                                            lat_values.append(lat)
                                            if row_template is None:
                                                row_template = row  # salva una riga (i valori costanti)

                                    lat_mean = sum(lat_values) / len(lat_values) if lat_values else 0.0

                                    fieldnames = ['rep', 'device_type', 'quant_type', 'ram_eon[KB]', 'rom_eon[MB]', 'lat[ms]']
                                    with open(filpath, mode='a', newline='') as csvfile:
                                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                                        #writer.writeheader()
                                        writer.writerow({
                                            'rep': 'mean'+len(repetitions_list)+'run',
                                            'device_type': row_template['device_type'],
                                            'quant_type': row_template['quant_type'],
                                            'ram_eon[KB]': row_template['ram_eon[KB]'],
                                            'rom_eon[MB]': row_template['rom_eon[MB]'],
                                            'lat[ms]': f"{lat_mean}"
                                        })
  
                            elapsed = (time.time() - t)/60
                            print(f"Elapsed time: {elapsed} min\n")


                        # Calculate latency mean
                        ultime_righe = []

                        for filepath in metrics_outputpath_list:
                            with open(filepath, mode='r', newline='') as csvfile:
                                reader = list(csv.DictReader(csvfile))
                                ultima_riga = reader[-1]
                                ultime_righe.append(ultima_riga)

                        with open(metrics_outputpath_mean, mode='w', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(ultime_righe)

                    # Crea file finale con tutti i risultati del profiling per i vari dispositivi

                    #header = None
                    #with open(metrics_outputpath_merge, mode='w', newline='') as out_csv:
                    #    writer = None
                    #
                    #    for filepath in metrics_outputpath_mean_list:
                    #        with open(filepath, mode='r', newline='') as in_csv:
                    #            reader = csv.DictReader(in_csv)
                    #            if header is None:
                    #                header = reader.fieldnames
                    #                writer = csv.DictWriter(out_csv, fieldnames=header)
                    #                writer.writeheader()
                    #            for row in reader:
                    #                writer.writerow(row)

                    # Crea file finale con tutti i risultati del profiling per i vari dispositivi interlacciati
                    header = None
                    all_rows = []

                    # Legge tutte le righe (senza header) da ciascun file
                    for filepath in metrics_outputpath_mean_list:
                        with open(filepath, mode='r', newline='') as in_csv:
                            reader = list(csv.DictReader(in_csv))
                            if header is None:
                                header = reader[0].keys()
                            all_rows.append(reader)

                    # Verifica che tutti abbiano lo stesso numero di righe
                    num_rows = len(all_rows[0])
                    assert all(len(rows) == num_rows for rows in all_rows), "I file devono avere lo stesso numero di righe."

                    # Scrive l'output interlacciato
                    with open(metrics_outputpath_merge, mode='w', newline='') as out_csv:
                        writer = csv.DictWriter(out_csv, fieldnames=header)
                        writer.writeheader()
                        for i in range(num_rows):
                            for file_rows in all_rows:
                                writer.writerow(file_rows[i])



                        #if deploy == 1:
                        #    #### Edge Impulse Deploy ####
                        #    
                        #    # Set model input type
                        #    model_input_type = ei.model.input_type.OtherInput()
                        #
                        #    # Set model information, such as your list of labels
                        #    model_output_type = ei.model.output_type.Classification(labels=labels)
                        #    
                        #    # Create C++ library with trained model
                        #    deploy_bytes = None
                        #    try:
                        #        deploy_bytes = ei.model.deploy(model=model_to_profile,
                        #                                        model_output_type=model_output_type,
                        #                                        model_input_type=model_input_type,
                        #                                        deploy_target='zip',
                        #                                        output_directory=saved_models_edgeimpulse)
                        #    except Exception as e:
                        #        print(f"Could not profile: {e}")
                        #
                        #    # Write the downloaded raw bytes to a file
                        #    if deploy_bytes:
                        #        deploy_filename = saved_models_edgeimpulse + model_type_load + '_' + quant_type + '_' + device_type + '_deploy_bytes.zip'
                        #        with open(deploy_filename, 'wb') as f:
                        #            f.write(deploy_bytes.getvalue())
            
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
                save_files_flag = 0
                test = 0 # Predict with val set
                ##Rate_OPT_py_load_val, Rate_DL_py_load_val   = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
                save_files_flag = 0
                test = 1 # Predict with test set
                #Rate_OPT_py_load_test, Rate_DL_py_load_test = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
                save_files_flag = 0
                test = 2 # Predict with all dataset
                #Rate_OPT_py_load, Rate_DL_py_load = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, model_py, network_folder_out_RateDLpy, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
                save_files_flag = 0
                test = 3 # Predict with TF-Lite Model
                #Rate_OPT_py_load_test_tflite, Rate_DL_py_load_test_tflite = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, tflite_quant_model, network_folder_out_RateDLpy_TFLite, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
                save_files_flag = 1
                test = 4 # Predict with TF-Lite Model with all dataset
                Rate_OPT_py_load_tflite, Rate_DL_py_load_tflite = model_predict(xdataset, Y_dataset, xval, Y_val, xtest, Y_test, YValidation_un_val, YValidation_un_test, tflite_quant_model, network_folder_out_RateDLpy_TFLite, end_folder_Training_Size_dd_max_epochs_load, test=test, save_files=save_files_flag)
            
            learning_rate = model_py.optimizer.learning_rate.numpy()
            print(f"Learning rate loaded model: {learning_rate}")

        # %%
        ################################ Train Model ################################
        if train_model_flag == 1 and load_model_flag == 0:

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
                print(f"\nRate_DL_py_test: {Rate_DL_py_test}")
                print(f"Rate_DL_py_val: {Rate_DL_py_val}")
        except Exception as e:
            print('Some missing info to print, no problem')
        
        print(f"\nRicorda che Rate_DL_valOld non può essere confrontato con Rate_DL_py_load_val e Rate_DL_py_load_test, perchè il primo è calcolato sul validation set old, mentre i secondi sul val e test set nuovi")

        tf.keras.backend.clear_session()

#logfile.close()