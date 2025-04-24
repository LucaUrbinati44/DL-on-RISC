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
log_dir = "/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/Output_Python/Neural_Network/tensorboard_logs/"
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

def model_predict(xv, model_py, YValidation_un2):

    print(f"\nStart DL prediction...")

    #print(xv.shape)  # (6200, 1024)

    YPredicted = model_py.predict(xv, verbose=1, batch_size=128)

    #print(YPredicted.shape)

    Indmax_DL_py = np.argmax(YPredicted, axis=1)
    #print(Indmax_DL_py.shape)
    print(np.min(Indmax_DL_py))
    print(np.max(Indmax_DL_py))

    # Questi devono essere numeri interi

    Indmax_DL = Indmax_DL_py

    #validation_accuracy = 0
    MaxR_DL = np.zeros((Indmax_DL.shape[0],), dtype=np.float32)
    #MaxR_OPT = np.zeros((len(Indmax_OPT),), dtype=np.float32)

    # Ciclo di confronto
    for b in range(Indmax_DL.shape[0]):
        #MaxR_DL[b] = YValidation_un[b, Indmax_DL[b], 0, 0]
        MaxR_DL[b] = YValidation_un2[0, 0, Indmax_DL[b], b]
        #MaxR_OPT[b] = YValidation_un[b, Indmax_OPT[b], 0, 0]

        #if MaxR_DL[b] == MaxR_OPT[b]:
        #    validation_accuracy += 1

    Rate_DL_py = MaxR_DL.mean()
    #Rate_OPT = MaxR_OPT.mean()
    #validation_accuracy = validation_accuracy / Indmax_DL.shape[0]

    #print(f"size(MaxR_DL): {MaxR_DL.shape}")

    return Rate_DL_py

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
saved_models = network_folder_out + 'saved_models/'
figure_folder = output_folder + 'Figures/'

import os

folders = [
    output_folder,
    network_folder_out,
    network_folder_out_YPredicted,
    network_folder_out_RateDLpy,
    saved_models,
    figure_folder
]

for folder in folders:
    if not os.path.exists(folder):  # Controlla se la cartella esiste
        os.makedirs(folder, exist_ok=True)  # Crea la cartella se non esiste
        print(f"\nCartella creata: {folder}")
    #else:
    #    print(f"La cartella esiste già: {folder}")

# %%
My_ar = [32, 64]
Mz_ar = [32, 64]
#My_ar = [32]
#Mz_ar = [32]
#My_ar = [64]
#Mz_ar = [64]
Mx = 1

M_bar=8
Ur_rows = [1000, 1200]
#              0    1      2      3      4      5      6
#Training_Size=[2, 10000, 14000, 18000, 22000, 26000, 30000]
#Training_Size=[10000, 14000, 18000, 22000, 26000, 30000]
Training_Size=[2]


load_model_flag = 1
max_epochs_load = 20

train_model_flag = 1
max_epochs = 20

# %%

# count, value
for i, ris in enumerate(My_ar):

    #My = My_ar[i]
    #Mz = Mz_ar[i]
    My = ris
    Mz = ris
    print(f"\nRIS: {My}x{Mz}")

    for j, Training_Size_dd in enumerate(Training_Size):

        end_folder = '_seed' + str(seed) + '_grid' + str(Ur_rows[1]) + '_M' + str(My) + str(Mz) + '_Mbar' + str(M_bar)
        end_folder_Training_Size_dd = end_folder + '_' + str(Training_Size_dd)

        #Training_Size_dd = Training_Size[j]
        print(f"\nTraining_Size_dd: {Training_Size_dd}")

        Rate_DL_py = 0

        #os._exit(0)

        # %% [markdown]
        # ## Directly import XTrain

        #Import .mat files of datasets splits
        filename_XTrain = DL_dataset_folder + 'XTrain' + end_folder_Training_Size_dd + '.mat'
        filename_YTrain = DL_dataset_folder + 'YTrain' + end_folder_Training_Size_dd + '.mat'
        filename_XValidation = DL_dataset_folder + 'XValidation' + end_folder_Training_Size_dd + '.mat'
        filename_YValidation = DL_dataset_folder + 'YValidation' + end_folder_Training_Size_dd + '.mat'

        #print(filename_XTrain)
        #print(filename_YTrain)
        #print(filename_XValidation)
        #print(filename_YValidation)

        # Load the data using h5py for MATLAB v7.3 files
        with h5py.File(filename_XTrain, 'r') as f:
            X_train = np.array(f['XTrain'][:], dtype=force_datatype)
        with h5py.File(filename_YTrain, 'r') as f:
            Y_train = np.array(f['YTrain'][:], dtype=force_datatype)
        with h5py.File(filename_XValidation, 'r') as f:
            X_val = np.array(f['XValidation'][:], dtype=force_datatype)
        with h5py.File(filename_YValidation, 'r') as f:
            Y_val = np.array(f['YValidation'][:], dtype=force_datatype)

        #print(X_train.shape)
        #print(Y_train.shape)
        #print(X_val.shape)
        #print(Y_val.shape)
        print(f"\nY_train.dtype: {Y_train.dtype}")

        # %% [markdown]
        # ## Load Dataset DL_input_reshaped
        
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

        #print(DL_input_reshaped.shape)
        #print(DL_output_reshaped.shape)
        #print(RandP_all.shape)

        #print(np.min(DL_input_reshaped))
        #print(np.max(DL_input_reshaped))
        #print(np.min(DL_output_reshaped))
        #print(np.max(DL_output_reshaped))

        # %% [markdown]
        # ## Load Rates

        # Costruzione del nome file
        filename_DL_output_un_reshaped = DL_dataset_folder + 'DL_output_un_reshaped' + end_folder + '.mat'

        # Load the data using h5py for MATLAB v7.3 files
        with h5py.File(filename_DL_output_un_reshaped, 'r') as f:
            # Accesso alla variabile (nome del dataset = nome della variabile in MATLAB)
            YValidation_un = np.array(f['DL_output_un_reshaped'], dtype=force_datatype)

        #print(YValidation_un.shape)

        YValidation_un2 = np.transpose(YValidation_un, (3, 2, 1, 0))  # conversione a (b, z, y, x)
        #print(YValidation_un2.shape)

        # %% [markdown]
        # ## Dataset split originale

        # Flatten the input and output arrays if necessary
        #X = DL_input_reshaped.reshape(DL_input_reshaped.shape[0], -1).astype(np.float32)
        #Y = DL_output_reshaped.reshape(DL_output_reshaped.shape[0], -1).astype(np.float32)

        # Split the dataset into training and validation sets
        #X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_size / (training_size + validation_size), shuffle=False, random_state=seed)

        RandP_all2 = np.squeeze(np.array(RandP_all.astype(int))) - 1

        Training_Ind = RandP_all2[0:Training_Size_dd]

        Validation_Size = 6200
        Validation_Ind = RandP_all2[-Validation_Size:]

        #print(Training_Ind.shape)
        #print(Validation_Ind.shape)

        X_train = np.array(DL_input_reshaped[Training_Ind, :, :, :], dtype=force_datatype).squeeze()
        Y_train = np.array(DL_output_reshaped[Training_Ind, :, :, :], dtype=force_datatype).squeeze()
        X_val = np.array(DL_input_reshaped[Validation_Ind, :, :, :], dtype=force_datatype).squeeze()
        Y_val = np.array(DL_output_reshaped[Validation_Ind, :, :, :], dtype=force_datatype).squeeze()

        #print(X_train.shape)
        #print(Y_train.shape)
        #print(X_val.shape)
        #print(Y_val.shape)

        # %% [markdown]
        # ## Recreate the same network in Python

        # ### Load normalization parameters from Matlab trained model

        filename_trainedNet_scaler = network_folder_in + 'trainedNet_scaler' + end_folder_Training_Size_dd + '.mat'

        with h5py.File(filename_trainedNet_scaler, 'r') as f:
            trainedNet_scaler = f['trainedNet_scaler'][:][0][0]

        #print(trainedNet_scaler.shape)
        print(f"\ntrainedNet_scaler: {trainedNet_scaler}") # should be -5.1904644e-06 for Training_Size_dd=30000


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

        # %% [markdown]
        # ### Normalize data

        # %%
        # Normalizzazione manuale se già hai mean_array e variance_array da MATLAB
        X_train_normalized = np.array((X_train - mean_array) / np.sqrt(variance_array), dtype=force_datatype)

        X_val_normalized = np.array((X_val - mean_array) / np.sqrt(variance_array), dtype=force_datatype)

        #print(mean_array)
        #print(variance_array)

        normalized = 1

        # %%
        #print(X_train[0][0:5])
        #print(mean_array[0:5])
        #print(X_train_normalized[0][0:5])

        # %% [markdown]
        # ## DL Model Definition

        if normalized == 1:
            xt = X_train_normalized
            xv = X_val_normalized
        else:
            xt = X_train
            xv = X_val

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

        #print(model_py.loss)

        # %% [markdown]
        # ## DL Model Training

        # ------------------ Training Options ------------------ #
        mini_batch_size = 500

        if load_model_flag == 1:
            max_epochs_new = max_epochs_load + max_epochs
            factor = 0.5
            patience = 3
            min_delta = 0.05
        else:
            max_epochs_new = max_epochs
            factor = 0.5
            patience = 2
            min_delta = 0.1

        # For the output filenames
        end_folder_Training_Size_dd_max_epochs = end_folder_Training_Size_dd + '_' + str(max_epochs_new)
        filename_Rate_DL_py = network_folder_out_RateDLpy + 'Rate_DL_py' + end_folder_Training_Size_dd_max_epochs + '.mat'
        model_type = 'model_py' + end_folder_Training_Size_dd_max_epochs
        
        tensorboard_logs = log_dir + model_type
        tensorboard_callback = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1)

        #if Training_Size_dd < mini_batch_size:
        #    validationFrequency = Training_Size_dd
        #else:
        #    validationFrequency = int(np.floor(Training_Size_dd/mini_batch_size))
        validationFrequency = 1

        # ------------------ Learning Rate Scheduler ------------------ #
        #def lr_schedule(epoch, lr):
        #    if epoch > 0 and epoch % 5 == 0: # Prima era modulo 3
        #        return lr * 0.5  # Drop learning rate by factor of 0.5 every x epochs
        #    return lr

        #lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,         # Riduce di metà
            patience=patience,  # Numero di epoche senza miglioramento ≥ y
            min_delta=min_delta,      # Miglioramento minimo da considerare significativo
            verbose=1
        )
                
        # ------------------ DL Model Training ------------------ #
        #train_dataset = tf.data.Dataset.from_tensor_slices((x, Y_train)).batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)
        #val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

        if load_model_flag == 1:
            end_folder_Training_Size_dd_max_epochs_load = end_folder_Training_Size_dd + '_' + str(max_epochs_load)
            model_type_load = 'model_py' + end_folder_Training_Size_dd_max_epochs_load

            model_py = load_model(saved_models + model_type_load + '.keras', custom_objects={'mse_custom': mse_custom})
            print(f"\nModel {saved_models + model_type_load + '.keras'} loaded")

            Rate_DL_py_load = model_predict(xv, model_py, YValidation_un2)
            print(f"Rate_DL_py: {Rate_DL_py_load}")

            learning_rate = model_py.optimizer.learning_rate.numpy()
            print(f"Learning rate loaded model: {learning_rate}")
        

        if train_model_flag == 1:
            print("\nStart DL training...")
            start_time = time.time()

            history = model_py.fit(
                xt, Y_train,
                validation_data=(xv, Y_val),
                #train_dataset, 
                #validation_data=val_dataset
                batch_size=mini_batch_size,
                epochs=max_epochs,
                shuffle=True,  # Shuffle data at each epoch
                callbacks=[lr_scheduler, tensorboard_callback],
                validation_freq=validationFrequency,
                verbose=2
            )

            elapsed_time = time.time() - start_time
            print(f"Training completed in {elapsed_time / 60:.2f} minutes.")

            # %%
            # Save the trained model
            #The saved .keras file contains:
            # - The model's configuration (architecture)
            # - The model's weights
            # - The model's optimizer's state (if any)
            # model.save() is an alias for keras.saving.save_model()
            model_py.save(saved_models + model_type + '.keras')  # The file needs to end with the .keras extension

            print(f"\nModel saved in {saved_models}")

            #np.save(os.path.join(output_folder, 'history.npy'), history.history)
            #np.save(os.path.join(output_folder, 'Y_predicted.npy'), Y_predicted)
            #print("History and Y_predicted saved successfully.")


        # %% [markdown]
        # ## DL Model Prediction
        # # SOSTITUIRE X_val CON X_test!!!
        Rate_DL_py = model_predict(xv, model_py, YValidation_un2)

        # Scrittura in formato HDF5 (compatibile MATLAB v7.3)
        with h5py.File(filename_Rate_DL_py, 'w') as f:
            f.create_dataset('Rate_DL_py', data=Rate_DL_py)

        filename_Rate_OPT = network_folder_in + 'Rate_OPT' + end_folder_Training_Size_dd + '.mat'

        with h5py.File(filename_Rate_OPT, 'r') as f:
            Rate_OPT = f['Rate_OPT'][:][0][0]

        filename_Rate_DL = network_folder_in + 'Rate_DL' + end_folder_Training_Size_dd + '.mat'

        with h5py.File(filename_Rate_DL, 'r') as f:
            Rate_DL = f['Rate_DL'][:][0][0]

        # Output finali
        print(f"\nRate_OPT: {Rate_OPT}")
        print(f"Rate_DL: {Rate_DL}")
        if load_model_flag == 1:
            print(f"Rate_DL_py_load: {Rate_DL_py_load}")
        print(f"Rate_DL_py: {Rate_DL_py}")

        #Rate_DLt_py[ris, j] = Rate_DL_py #(len(My_ar), len(Training_Size))

