import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)

import csv
import os
import json
import random
import re
#import h5py
#from ai_edge_litert.interpreter import Interpreter
#from tensorflow.lite.python import schema_py_generated as schema_fb

import psutil
import subprocess
import signal
import datetime
import time
import serial.tools.list_ports
import argparse
import shutil


import serial_feeder_and_logger

# Read mcu name from user
parser = argparse.ArgumentParser()
parser.add_argument('--mcu', type=str, default='pico', help='MCU name (pico, esp32, stm32f4, stm32h7)')
args = parser.parse_args()
mcu_name = args.mcu
allowed_mcus = ['pico', 'esp32', 'stm32f4', 'stm32h7']
if mcu_name not in allowed_mcus:
    print(f"Errore: --mcu deve essere uno tra {allowed_mcus}.")
    os._exit(1)

def is_windows():
    return 1 if os.name == 'nt' else 0
ISWINDOWS = is_windows()

if not(ISWINDOWS):
    import DL_training_4_v3_test

print("TensorFlow version:", tf.__version__)

# Imposta il seed globale
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ------------------------------------------------------------------------------------------
# Definizione dei parametri

K_DL = 64 # subcarriers, costante (per ora)
Ur_rows = [1000, 1200]
#My_ar = [32, 64]
#Mz_ar = [32, 64]

# TODO DEVO CAMBIARE QUI
#My_ar = [32]
#Mz_ar = [32]
My_ar = [64]
Mz_ar = [64]

My = My_ar[0]
Mz = Mz_ar[0]

#Training_Size = [2, 10000, 14000, 18000, 22000, 26000, 30000]
#Training_Size = [10000, 30000]
#Training_Size = [10000]
#Training_Size = [30000]
#Training_Size = [2000, 4000, 6000, 8000]
#Training_Size = [2, 2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000]
#Training_Size = [8000, 10000, 14000, 18000, 22000, 26000, 30000]
#Training_Size = [10000, 30000]
Training_Size = [30000]

Training_Size_dd = Training_Size[0]

# ------------------------------------------------------------------------------------------

debug = 2            # 0: production mode, 1: production mode for one case, 2: debug mode

#dummy = 'dummy_'    # '': production mode, 'dummy_': dummy mode
dummy = ''

#test_set_size = 'small' # 'full' in prodcution
test_set_size = 'full'

if test_set_size == 'small':
    small_samples = 5 # even and greater than or equal to 2
    warmup_samples_for_statistics = 0
else:
    small_samples = 5 # non cambiare, usata solo per le print
    warmup_samples_for_statistics = 100

# SPAZIO DI RICERCA 
# 8 celle attive x 64 subcarriers x 2 (real/img) = 1024
if debug == 0:
    active_cells = [1,4,8,12,28]
    output_dims = [My*Mz]         # Numero di neuroni di uscita (32x32 = 1024)
    num_layers_list = [0,1,2,3]   # Numero di layer MLP (ESCLUSO L'ULTIMO!!!)
    R_list = [32, 8, 1]
elif debug == 1: # (9x9+1)x2 x 8min ciascuno = 2.67h
    active_cells = [28] # TODO DEVO CAMBIARE QUI
    output_dims = [My*Mz]
    num_layers_list = [0,1,2,3]
    R_list = [32, 8, 1]
elif debug == 2:
    active_cells = [28]
    output_dims = [My*Mz]
    num_layers_list = [3]
    R_list = [1] # indifferente il valore di R se num_layers_list = [0]. Verrà eseguita una sola iterazione
else: # modello di Taha
    #input_featuress = [1024]
    active_cells = [8]
    output_dims = [1024]
    num_layers_list = [3]

# ------------------------------------------------------------------------------------------

# default Taha
#patience = 3
#min_delta = 0.05
#max_epochs = 20

# CosineGuidedReduceOnPlateau
#init_learning_rate = 0.1
#min_learning_rate = 1e-5
#patience_list = 2
#min_delta_list = 0.001
#max_epochs_list = 60

# ReduceLROnPlateau
init_learning_rate = 0.1
min_learning_rate = 0.0001
factor = 0.75
#patience_list   = [      4,       4,      4,     4 ]
#min_delta_list  = [  0.001,   0.001,  0.001, 0.001 ]
#max_epochs_list = [    200,     200,    200,   200 ] # tanto c'è early stopping
patience = 4
min_delta = 0.001
max_epochs = 200 # tanto c'è early stopping

# ------------------------------------------------------------------------------------------

# STEP 1) Train DL model
#datetime_str = datetime.datetime.now().strftime("experiment-%d-%m-%y-%H-%M")
datetime_str = datetime.datetime.now().strftime("experiment" + '_M' + str(My) + str(Mz) + '_Mbar' + str(active_cells[0]))
initial_epoch = 0

train_model_flag = 1
convert_model_flag = 1
save_files_flag_master = 1
save_files_flag_master_once = 1

load_model_flag = 0
predict_loaded_model_flag = load_model_flag # deve essere uguale a load_model_flag
profiling_flag = 0
compile_and_upload_flag = 0

# STEP 2) Inference on MCUs
##### LOADING (for inference and profiling) #####
#datetime_str = USA ALTRO SCRIPT

# ------------------------------------------------------------------------------------------

if profiling_flag == 1:
    # These boards must run on Windows
    if ISWINDOWS:
        if mcu_name == 'pico':
            mcu_type = {'name': 'pico', 
                        #'port': '/dev/ttyACM0',
                        'serial_number': "E66480454F7E2833", # when using earlephilhower/arduino-pico
                        'serial_number_2': "458064E633287E4F", # when using ArduinoCore-mbed
                        #'baud_rate': 9600}
                        #'baud_rate': 115200}
                        'baud_rate': 115200}
        elif mcu_name == 'esp32':
            mcu_type = {'name': 'esp32-s2-saola-tflm', 
                        #'port': '/dev/ttyUSB0',
                        'serial_number': "2017651D8BD2EB11B8CCC149E93FD3F1",
                        'baud_rate': 921600}
        else:
            # Non riesce a scaricare le librerie per STM32 su Windows. Serve attach per lavorare su Linux
            print("Run this board on Linux")
            os._exit(1)
            
    else: # These boards must run on Linux
        if mcu_name == 'stm32f4':
            mcu_type = {'name': 'nucleo-f446ze', 
                        #'port': '/dev/ttyACM1',
                        'serial_number': "066FFF485570854967101750",
                        'baud_rate': 115200}
        elif mcu_name == 'stm32h7':
            mcu_type = {'name': 'nucleo-h753zi',
                        #'port': '/dev/ttyACM0',
                        'serial_number': "0024002F3234510737333934",
                        'baud_rate': 921600}
        else: 
            print("Run this board on Windows")
            os._exit(1)

# ------------------------------------------------------------------------------------------

# Aprire tensorboard da terminale:
# tensorboard --logdir=/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/Output_Python/Neural_Network/experiment-09-09-25-15-56

# ------------------------------------------------------------------------------------------

base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'
if ISWINDOWS:
    #base_folder = subprocess.check_output(["wslpath", "-w", base_folder]).decode().strip()
    base_folder = subprocess.check_output(["wsl", "wslpath", "-w", base_folder]).decode().strip()
    print(base_folder)

input_folder = os.path.join(base_folder, 'Output Matlab')

DL_dataset_folder = os.path.join(input_folder, 'DL Dataset')
network_folder_in = os.path.join(input_folder, 'Neural Network')


output_folder = os.path.join(base_folder, 'Output_Python')
network_folder_out = os.path.join(output_folder, 'Neural_Network')
figure_folder = os.path.join(output_folder, 'Figures')

datetime_dir = os.path.join(network_folder_out, datetime_str)
network_folder_out_RateDLpy = os.path.join(datetime_dir, 'RateDLpy')
network_folder_out_RateDLpy_TFLite = os.path.join(datetime_dir, 'RateDLpy_TFLite')
network_folder_out_RateDLpy_TFLite_mcu = os.path.join(datetime_dir, 'RateDLpy_TFLite_mcu')
saved_models_keras = os.path.join(datetime_dir, 'saved_models_keras')

mcu_profiling_folder = os.path.join(output_folder, 'Profiling_Search_MCU')
mcu_profiling_folder_model = os.path.join(mcu_profiling_folder, 'model')
mcu_profiling_folder_scaler = os.path.join(mcu_profiling_folder, 'scaler')
mcu_profiling_folder_test_data = os.path.join(mcu_profiling_folder, 'test_data')
mcu_profiling_folder_test_data_normalized = os.path.join(mcu_profiling_folder, 'test_data_normalized')
mcu_profiling_folder_outputdata = os.path.join(mcu_profiling_folder, 'output_data')
mcu_profiling_folder_logfile = os.path.join(mcu_profiling_folder, 'logs') 
if profiling_flag == 1:
    mcu_profiling_logfile = os.path.join(mcu_profiling_folder_logfile, f'log_{mcu_type['name']}_{datetime_str}.txt')
#pio_projects_folder = '/mnt/c/Users/Work/Documents/PlatformIO/Projects/'
pio_projects_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/mcu'
if ISWINDOWS:
    #pio_projects_folder = subprocess.check_output(["wslpath", "-w", pio_projects_folder]).decode().strip()
    pio_projects_folder = subprocess.check_output(["wsl", "wslpath", "-w", pio_projects_folder]).decode().strip()
    print(pio_projects_folder)

#header_folder = 'tensorflow/lite/micro/examples/ml_on_risc/c_models'
#if ISWINDOWS:
#    #header_folder = subprocess.check_output(["wslpath", "-w", header_folder]).decode().strip()
#    header_folder = subprocess.check_output(["wsl", "wslpath", "-w", header_folder]).decode().strip()
#    print(header_folder)
tensorboard_dir =  os.path.join(datetime_dir, "tensorboard_logs_test")
training_history_dir = os.path.join(datetime_dir, "training_history")

folders = [
    output_folder,
    network_folder_out,
    figure_folder,
    datetime_dir,
    network_folder_out_RateDLpy,
    network_folder_out_RateDLpy_TFLite,
    network_folder_out_RateDLpy_TFLite_mcu,
    saved_models_keras,
    mcu_profiling_folder,
    mcu_profiling_folder_test_data,
    mcu_profiling_folder_test_data_normalized,
    mcu_profiling_folder_model,
    mcu_profiling_folder_scaler,
    mcu_profiling_folder_outputdata,
    mcu_profiling_folder_logfile,
    pio_projects_folder,
    #header_folder,
    tensorboard_dir,
    training_history_dir
]

for folder in folders:
    if not os.path.exists(folder):  # Controlla se la cartella esiste
        os.makedirs(folder, exist_ok=True)  # Crea la cartella se non esiste
        print(f"\nCartella creata: {folder}")
    else:
        print(f"La cartella esiste già: {folder}")


# ----- Run TensorBoard ------
def run_tensorboard():

    tensorboard_command = [
        "tensorboard",
        f"--logdir={tensorboard_dir}",
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
        terminate_tensorboard()
        tensorboard_process = subprocess.Popen(tensorboard_command)
        print(f"\nTensorBoard avviato in background.")
    except:
        print(f"\nfErrore nell'avvio di TensorBoard: {e}")
        # Chiudi TensorBoard se è già in esecuzione
        terminate_tensorboard()
        # Avvia TensorBoard in background
        tensorboard_process = subprocess.Popen(tensorboard_command)
        print(f"\nTensorBoard avviato in background.")

# ----- Costruzione del modello MLP parametrico -----
def build_mlp(input_features, output_dim, num_layers, hidden_units_list):
    model = tf.keras.Sequential([tf.keras.Input(shape=(input_features,), name='input')])
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(hidden_units_list[i], activation='relu', name=f'hidden_{i}'))
    model.add(tf.keras.layers.Dense(output_dim, activation=None, name='output'))
    return model

# ----- Conversione in TF-Lite INT8 -----
def convert_to_tflite_int8(model, x_sample, model_path_tflite):
    def representative_data_gen():
        for i in range(min(100, len(x_sample))):
            yield [x_sample[i:i+1].astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8_model = converter.convert()

    with open(model_path_tflite, 'wb', encoding='utf-8') as f:
        f.write(tflite_int8_model)

def get_model_paths(dummy, end_folder_Training_Size_dd, max_epochs, num_layers, R, input_features, hidden_units_list, output_dim):
    print('\n*** get_model_paths')

    #if len(hidden_units_list) == 1:
    #    hul = str(hidden_units_list[0])
    #elif len(hidden_units_list) == 2:
    #    hul = str(hidden_units_list[0]) + "_" + str(hidden_units_list[1])
    #elif len(hidden_units_list) == 3:
    #    hul = str(hidden_units_list[0]) + "_" + str(hidden_units_list[1]) + "_" + str(hidden_units_list[2])
    #
    #model_name_suffix = f"_in{input_features}_out{output_dim}_nl{num_layers}_R{R}_hul{hul}_initlr{init_learning_rate}_minlr{min_learning_rate}_fact{factor}_pat{patience}_mindelta{min_delta}"

    if num_layers == 0:
        hidden = '_'
    elif num_layers == 1:
        hidden = '_' + str(hidden_units_list[0]) + '_'
    elif num_layers == 2:
        hidden = '_' + str(hidden_units_list[0]) + '_' + str(hidden_units_list[1]) + '_'
    elif num_layers == 3:
        hidden = '_' + str(hidden_units_list[0]) + '_' + str(hidden_units_list[1]) + '_' + str(hidden_units_list[2]) + '_'
    
    model_name_suffix = f"_initlr{init_learning_rate}_minlr{min_learning_rate}_fact{factor}_pat{patience}_mindelta{min_delta}_nl{num_layers}_R{R}_in{input_features}{hidden}out{output_dim}"
    
    end_folder_Training_Size_dd_epochs = end_folder_Training_Size_dd + f"_ep{str(max_epochs)}"

    model_type_load = dummy + 'model_py_test' + end_folder_Training_Size_dd_epochs + model_name_suffix
    model_path_tflite = os.path.join(mcu_profiling_folder_model, f"{model_type_load}_int8.tflite")    

    model_path_keras = os.path.join(saved_models_keras, f"{model_type_load}.keras")
    
    return end_folder_Training_Size_dd_epochs, model_name_suffix, model_type_load, model_path_tflite, model_path_keras

def mse_custom(y_true, y_pred):
    # Calcola l'errore quadratico tra vero e predetto
    squared_error = tf.square(y_true - y_pred)  # shape: (batch_size, output_dim)=6200,1024

    # Somma degli errori lungo l'ultima dimensione (output_dim)
    sum_squared_error = tf.reduce_sum(squared_error, axis=-1)  # shape: (batch_size,)=6200

    # Media su tutto il batch
    loss = 0.5 * tf.reduce_mean(sum_squared_error)  # scalar
    return loss
  
        
# ----- Export TF-Lite INT8 model to C for TFLM -----
def export_to_c(model_path_tflite):
    print('\n*** export_to_c')

    model_name = 'mlp'
    model_name_lowercase = model_name.replace('/', '_').replace('.', '_').replace('-', '_')
    model_name_model_data = f"{model_name_lowercase}_model_data"
    header_path = os.path.join(mcu_lib_model_folder, f"{model_name_model_data}.h")
    source_path = os.path.join(mcu_lib_model_folder, f"{model_name_model_data}.cc")

    #generate_header(header_path, model_name)
    guard = f"{model_name.upper()}_H_".replace(".", "_")
    with open(header_path, "w", encoding='utf-8') as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"extern const uint8_t g_{model_name_model_data}[] PROGMEM;\n")
        f.write(f"extern const unsigned int g_{model_name_model_data}_len;\n\n")
        f.write(f"#endif  // {guard}\n")

    #generate_source(source_path, model_path_tflite, model_name)
    #https://stackoverflow.com/questions/73301347/how-to-convert-model-tflite-to-model-cc-and-model-h-on-windows-10
    #def generate_source(model_path_tflite: str, output_prefix: str, var_name: str = "g_model_data"):
    """
    Usa xxd per generare file .cc e .h per TFLM a partire da un modello .tflite quantizzato.

    Args:
        model_path_tflite: path al file .tflite
        output_prefix: prefisso per i file output (es. "magic_wand_model_data")
        var_name: nome della variabile da usare nell'array (default: g_model_data)
    """
    xxd_output_temp = os.path.join(mcu_lib_model_folder, f"xxd_output_temp.h")

    if ISWINDOWS:
        with open(model_path_tflite, "rb") as f:
            data = f.read()
        print(data[0:300])
        print(data[-300:-1])

        # Effettua sostituzioni
        with open(xxd_output_temp, "w", encoding='utf-8') as f:
            f.write(f"#include <config.h>\n")
            f.write(f"#include \"{header_path}\"\n")
            f.write(f"alignas(8) const uint8_t g_{model_name_model_data}[] PROGMEM = {{\n")        
            for i, b in enumerate(data):
                f.write(f"0x{b:02x},")
                if (i+1) % 16 == 0:
                    f.write("\n")
            f.write("\n};\n")
            f.write(f"const unsigned int g_{model_name_model_data}_len = {i+1};")
    else:
        os.system("xxd -i " + model_path_tflite + " > " + xxd_output_temp)

        # Leggi e modifica il contenuto
        with open(xxd_output_temp, "r", encoding='utf-8') as f:
            content = f.read()
        print(content[0:300])
        print(content[-300:-1])

        # Effettua sostituzioni
        model_path_tflite_lowercase = model_path_tflite.replace('/', '_').replace('.', '_').replace('-', '_')
        #print(model_path_tflite_lowercase)
        first_line_old = f"unsigned char {model_path_tflite_lowercase}[] = {{"
        # Aggiungi include dell'header
        # PROGMEM serve per salvare il modello in Flash invece che in RAM
        first_line_new = f"#include <config.h>\n#include \"{f"{model_name_model_data}.h"}\"\nalignas(8) const uint8_t g_{model_name_model_data}[] PROGMEM = {{"
        last_line_old = f"unsigned int {model_path_tflite_lowercase}_len"
        last_line_new = f"const unsigned int g_{model_name_model_data}_len"
        content = content.replace(first_line_old, first_line_new)
        content = content.replace(last_line_old, last_line_new)
        print(content[0:300])
        print(content[-300:-1])

    try:
        os.remove(source_path)
        print(f"File {source_path} eliminato con successo")
    except PermissionError:
        print("Errore permessi: file bloccato o senza permessi")
    except FileNotFoundError:
        print("File già inesistente")
    except Exception as e:
        print(f"Errore sconosciuto: {e}")
    
    with open(source_path, "w", encoding='utf-8') as f:
        #f.write(f'#include "{header_path.name}"\n\n')
        print("Scrivo contenuto modificato")
        if ISWINDOWS:
            cmd = f'copy "{xxd_output_temp}" "{source_path}"'
            os.system(cmd)
        else:
            f.write(content)
        f.flush()
        os.fsync(f.fileno())

    os.remove(xxd_output_temp)
    print(f"File {xxd_output_temp} eliminato con successo")

# ----- Stima delle MAC operations e della dimensione del modello -----
def get_model_info_precise(model, model_path_tflite, save_dir='./'):
    print('\n*** get_model_info_precise')

    ## ROM: peso del modello in float32 e in int8 (1 byte per peso/bias)
    #model_size_bytes_float32 = 0
    #model_size_bytes_int8 = 0
    #for v in model.trainable_weights:
    #    model_size_bytes_float32 += np.prod(v.shape) * 4
    #    model_size_bytes_int8 += np.prod(v.shape)
    #
    #model_size_float32_kb = model_size_bytes_float32 /1024
    #model_size_int8_kb = model_size_bytes_int8 /1024
    #model_size_int8_mb = model_size_bytes_int8 /1024 /1024
    #
    ## MACs totali (product kernel shape)
    #macs = 0
    #for layer in model.layers:
    #    if isinstance(layer, tf.keras.layers.Dense):
    #        macs += np.prod(layer.kernel.shape)
    #
    ## RAM: tensor arena (int8)
    #input_elements = model.input_shape[1]
    #max_intermediate = 0
    #output_elements = 0
    #
    #for layer in model.layers:
    #    if isinstance(layer, tf.keras.layers.Dense):
    #        output_elements = layer.units
    #        if output_elements > max_intermediate:
    #            max_intermediate = output_elements
    #
    #input_tensor_bytes = input_elements  # int8: 1 byte
    #output_tensor_bytes = output_elements
    #max_intermediate_tensor_bytes = max_intermediate
    #overhead_struct_bytes = 4096  # stima conservativa (struct + allocazioni)
    #stack_bytes = 2048            # stack base
    #
    #ram_total_bytes = input_tensor_bytes + output_tensor_bytes + max_intermediate_tensor_bytes + overhead_struct_bytes + stack_bytes
    #ram_total_kb = ram_total_bytes / 1024

    model_file_size_int8_kb = os.path.getsize(model_path_tflite) /1024
    model_file_size_int8_mb = model_file_size_int8_kb /1024

    # Sanity check che dà lo stesso risultato di os.path.getsize
    len_pattern = re.compile(r"const\s+unsigned\s+int\s+g_mlp_model_data_len\s*=\s*(\d+);", re.IGNORECASE)
    model_name = 'mlp'
    source_path = os.path.join(save_dir, f"{model_name}_model_data.cc")
    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = len_pattern.search(line)
            if match:
                bytes_used = int(match.group(1))
                model_h_size_int8_kb = bytes_used /1024

    #return round(model_size_int8_kb, 2), round(model_size_int8_mb, 2), round(ram_total_kb, 2), macs
    return round(model_h_size_int8_kb, 2), round(model_file_size_int8_kb, 2), round(model_file_size_int8_mb, 2), 


def change_config_file_1(file_path, input_features, output_dim, baud_rate):

    print('\n*** change_config_file_1')

    # regex per INPUT_FEATURE_SIZE e OUTPUT_FEATURE_SIZE
    pattern_input = re.compile(r"(#define\s+INPUT_FEATURE_SIZE\s+)\d+")
    pattern_output = re.compile(r"(#define\s+OUTPUT_FEATURE_SIZE\s+)\d+")
    pattern_baudrate = re.compile(r"(#define\s+BAUD_RATE\s+)\d+")
    pattern_chunksizemax = re.compile(r"(#define\s+CHUNK_SIZE_MAX\s+)\d+")

    # leggi file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # sostituisci i numeri
    # r = raw string, cioè le backslash \ non vengono interpretate da Python.
    # f = f-string, puoi mettere variabili dentro {}.
    # \1 è un altro modo per riferirsi al primo gruppo catturato.
    content = pattern_input.sub(rf"\g<1>{input_features}", content)
    content = pattern_output.sub(rf"\g<1>{output_dim}", content)
    content = pattern_baudrate.sub(rf"\g<1>{baud_rate}", content)
    content = pattern_chunksizemax.sub(rf"\g<1>{input_features}", content) # CAMBIARE QUI se serve ridurre il chunk_size_max

    # salva file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def change_lib_file_1(folder_path, input_features, output_dim):

    print('\n*** change_lib_file_1')

    normalize_input_file_path = os.path.join(folder_path, 'normalize_input.cc')
    quantize_input_file_path = os.path.join(folder_path, 'quantize_input.cc')
    dequantize_output_file_path = os.path.join(folder_path ,'dequantize_output.cc')
    extract_codebook_index_fast_file_path = os.path.join(folder_path, 'extract_codebook_index_fast.cc')

    file_path_list = [normalize_input_file_path, quantize_input_file_path, dequantize_output_file_path, extract_codebook_index_fast_file_path]
    
    # regex per unroll
    pattern_unroll = re.compile(r"(#pragma\s+GCC\s+unroll\s+)\d+")

    for i, file_path in enumerate(file_path_list):

        # leggi file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if i < 2:
            variable = input_features
        else:
            variable = output_dim

        # sostituisci i numeri
        content = pattern_unroll.sub(rf"\g<1>{variable}", content)

        # salva file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def change_config_file_2(mcu_include_config, mean_array_filepath, variance_array_filepath):

    print('\n*** change_config_file_2')
 
    with open(mcu_include_config, "r", encoding="utf-8") as f:
        content = f.read()

    # Versione con mean array e variance array
    #pattern_mean = re.compile(r"(const float mean_array\[INPUT_FEATURE_SIZE\] PROGMEM = \{)([\s\S]*?)(\};)")
    #pattern_variance = re.compile(r"(const float variance_array_sqrt_inv\[INPUT_FEATURE_SIZE\] PROGMEM = \{)([\s\S]*?)(\};)")
    pattern_mean = re.compile(r"(const float mean_array\[INPUT_FEATURE_SIZE\] = \{)([\s\S]*?)(\};)")
    pattern_variance = re.compile(r"(const float variance_array_sqrt_inv\[INPUT_FEATURE_SIZE\] = \{)([\s\S]*?)(\};)")
     
    mean_array = np.load(mean_array_filepath)      
    array_str = ", ".join(f"{v}f" for v in mean_array)
    content = re.sub(pattern_mean, rf"\g<1>\n    {array_str}\n\3", content)
    
    variance_array = np.load(variance_array_filepath)
    array_str = ", ".join(f"{1/np.sqrt(v)}f" for v in variance_array)
    content = re.sub(pattern_variance, rf"\g<1>\n    {array_str}\n\3", content)

    # salva file
    with open(mcu_include_config, "w", encoding="utf-8") as f:
        f.write(content)

def change_config_file_3(mcu_include_config, mean_array_filepath):

    print('\n*** change_config_file_3')
 
    with open(mcu_include_config, "r", encoding="utf-8") as f:
        content = f.read()

    # Versione con solo mean scalar
    #pattern_mean = re.compile(r"(const float mean_array PROGMEM = )([0-9eE\+\-\.]+)(;)")
    pattern_mean = re.compile(r"(const float mean_array = )([0-9eE\+\-\.]+)(;)")
     
    mean_array = np.load(mean_array_filepath)
    content = re.sub(pattern_mean, rf"\g<1>{mean_array}\3", content)

    # salva file
    with open(mcu_include_config, "w", encoding="utf-8") as f:
        f.write(content)


# ----- Recupera RAM e Flash da file di log generato da PlatformIO -----
def parse_compilation_logfile(file_path):
    """
    Estrae i byte usati per RAM e Flash da un file di log di PlatformIO.
    Se viene rilevato un overflow di RAM o Flash, assegna -1.
    Restituisce un dizionario: {'RAM': <int>, 'Flash': <int>}
    """
    
    print('\n*** parse_compilation_logfile')

    RAM_KB = 0
    Flash_MB = 0

    CLK_FREQ_MHZ = 0
    RAM_HW_KB = 0
    Flash_HW_MB = 0

    Error_does_not_fit = 0

    # Pattern normale quando la compilazione ha successo
    # | significa "o"
    # : — corrisponde al carattere due punti letterale.
    # .*? — corrisponde a qualsiasi carattere (tranne newline), in modo non greedy (prende il minimo necessario)
    # used\s+ — la parola "used" seguita da almeno uno spazio
    # (\d+) — gruppo catturante: una sequenza di cifre (numeri interi), che rappresenta i byte usati
    # \s+bytes — uno o più spazi seguiti dalla parola "bytes"
    #no_overflow = re.compile(r'^\s*(RAM|Flash):.*?used\s+(\d+)\s+bytes', re.IGNORECASE)
    ram_pattern = re.compile(r'^\s*RAM:.*?used\s+(\d+)\s+bytes', re.IGNORECASE)
    flash_pattern = re.compile(r'^\s*Flash:.*?used\s+(\d+)\s+bytes', re.IGNORECASE)
    hardware_pattern_MB = re.compile(r"HARDWARE:\s+\S+\s+(\d+)MHz,\s+(\d+)KB RAM,\s+(\d+)MB Flash", re.IGNORECASE)
    hardware_pattern_KB = re.compile(r"HARDWARE:\s+\S+\s+(\d+)MHz,\s+(\d+)KB RAM,\s+(\d+)KB Flash", re.IGNORECASE)

    disconnected_pattern = re.compile(r"No accessible", re.IGNORECASE)

    #error_pattern = re.compile(r"\berror\b", re.IGNORECASE)
    #error_pattern = re.compile(r"\bError+\s\b")
    error_pattern = re.compile(r"\[FAILED\]")

    # Pattern di errore da overflow
    # \d+ per indicare un numero di byte (così non catturi testo non previsto).
    # re.IGNORECASE per sicurezza, anche se di solito PlatformIO le scrive in minuscolo.
    # ESP32
    overflow_flash_esp32 = re.compile(r"region `drom0_0_seg' overflowed by (\d+) bytes", re.IGNORECASE)
    overflow_ram   = re.compile(r"region `dram0_0_seg' overflowed by (\d+) bytes", re.IGNORECASE)
    # STM32
    overflow_flash_stm32 = re.compile(r"region `FLASH' overflowed by (\d+) bytes", re.IGNORECASE)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            #print("repr(line):", repr(line))

            match = ram_pattern.search(line)
            if match:
                bytes_used = int(match.group(1))
                RAM_KB = bytes_used  /1024
            
            match = flash_pattern.search(line)
            if match:
                bytes_used = int(match.group(1))
                Flash_MB = bytes_used /1024 /1024
            
            match = hardware_pattern_MB.search(line)
            if match:
                #print("match hardware")
                CLK_FREQ_MHZ = int(match.group(1))
                RAM_HW_KB = int(match.group(2))
                Flash_HW_MB = int(match.group(3))

            match = hardware_pattern_KB.search(line)
            if match:
                #print("match hardware")
                CLK_FREQ_MHZ = int(match.group(1))
                RAM_HW_KB = int(match.group(2))
                Flash_HW_MB = int(match.group(3))/1024

            # Cerca overflow RAM
            match = overflow_ram.search(line)
            if match:
                RAM_KB = -int(match.group(1))
                Error_does_not_fit = 1

            # Cerca overflow Flash
            match = overflow_flash_esp32.search(line)
            if match:
                Flash_MB = -int(match.group(1))
                Error_does_not_fit = 2
            match = overflow_flash_stm32.search(line)
            if match:
                Flash_MB = -int(match.group(1))
                Error_does_not_fit = 2

            match = disconnected_pattern.search(line)
            if match:
                Error_does_not_fit = 4

            if Error_does_not_fit == 0: # per non sovrascrivere gli altri errori
                match = error_pattern.search(line)
                if match:
                    print("match Error")
                    Error_does_not_fit = 3

    return RAM_KB, Flash_MB, CLK_FREQ_MHZ, RAM_HW_KB, Flash_HW_MB, Error_does_not_fit


#def find_com_port_by_vid_pid(DEVICE_VID, DEVICE_PID):
#    for port in serial.tools.list_ports.comports():
#        if f"VID:PID={DEVICE_VID}:{DEVICE_PID}" in port.hwid:
#            com_port = port.device
#            print(f"Dispositivo trovato su {com_port}")
#            return com_port
#    print(f"Dispositivo VID:PID={DEVICE_VID}:{DEVICE_PID} non trovato!")
#    return -1

def find_com_port_by_serial_number(SERIAL_NUMBER):
    while True:
        for port in serial.tools.list_ports.comports():
            if port.serial_number == SERIAL_NUMBER or (mcu_name == 'pico' and port.serial_number == mcu_type['serial_number_2']):
                com_port = port.device
                print(f"Dispositivo trovato su {com_port}")
                return com_port
        print(f"Dispositivo SERIAL_NUMBER={SERIAL_NUMBER} non trovato!")
        time.sleep(5)
                    
# ----- Salva risultati del profiling su file -----
def save_results_v3(dummy, end_folder_Training_Size_dd_epochs, K_DL,
                    Error_does_not_fit, Error_model_in_ram,
                    input_features, num_layers, R, hidden_units_list, output_dim, 
                    patience, min_delta, max_epochs, initial_epoch,
                    #model_h_size_int8_kb, model_file_size_int8_kb, model_file_size_int8_mb,
                    RAM_KB, Flash_MB, CLK_FREQ_MHZ, RAM_HW_KB, Flash_HW_MB, env_name,
                    mean_norm, perc50_norm, perc95_norm, std_norm,
                    mean_quant, perc50_quant, perc95_quant, std_quant,
                    mean_normquant, perc50_normquant, perc95_normquant, std_normquant,
                    mean_invoke, perc50_invoke, perc95_invoke, std_invoke,
                    mean_dequant, perc50_dequant, perc95_dequant, std_dequant,
                    mean_extract, perc50_extract, perc95_extract, std_extract,
                    mean_extract_fast, perc50_extract_fast, perc95_extract_fast, std_extract_fast,
                    mean_tot_latency, perc50_tot_latency, perc95_tot_latency, std_tot_latency,
                    mean_tot_latency_fast, perc50_tot_latency_fast, perc95_tot_latency_fast, std_tot_latency_fast,
                    Rate_OPT_py_load_val, Rate_DL_py_load_val,
                    Rate_OPT_py_load_test, Rate_DL_py_load_test,
                    Rate_DL_py_load_test_tflite, Rate_DL_py_load_test_tflite_mcu,
                    Indmax_OPT_py_load_test, Indmax_DL_py_load_test, 
                    Indmax_DL_py_load_test_tflite, Indmax_DL_py_load_test_tflite_mcu):
    
    print('\n*** save_results_v3')

    fieldnames = [
        'timestamp',
        'end_folder_Training_Size_dd_epochs', 'K_DL', 
        'Error_does_not_fit', 'Error_model_in_ram', 
        'input_features', 'num_layers', 'R', 'hidden_units_list', 'output_dim',
        'patience', 'min_delta', 'max_epochs', 'initial_epoch',
        #'model_h_size_int8_kb', 'model_file_size_int8_kb', 'model_file_size_int8_mb',
        'RAM_KB', 'Flash_MB', 'CLK_FREQ_MHZ', 'RAM_HW_KB', 'Flash_HW_MB', 'env_name',

        # normalize_input
        'mean_norm', 'perc50_norm', 'perc95_norm', 'std_norm',
        # quantize_input
        'mean_quant', 'perc50_quant', 'perc95_quant', 'std_quant',
        # normalize_and_quantize_input
        'mean_normquant', 'perc50_normquant', 'perc95_normquant', 'std_normquant',
        # interpreter_invoke
        'mean_invoke', 'perc50_invoke', 'perc95_invoke', 'std_invoke',
        # dequantize_output
        'mean_dequant', 'perc50_dequant', 'perc95_dequant', 'std_dequant',
        # extract_codebook_index
        'mean_extract', 'perc50_extract', 'perc95_extract', 'std_extract',
        # extract_codebook_index_fast
        'mean_extract_fast', 'perc50_extract_fast', 'perc95_extract_fast', 'std_extract_fast',
        # total latency
        'mean_tot_latency', 'perc50_tot_latency', 'perc95_tot_latency', 'std_tot_latency',
        'mean_tot_latency_fast', 'perc50_tot_latency_fast', 'perc95_tot_latency_fast', 'std_tot_latency_fast',

        # subset di risultati
        'Rate_OPT_py_load_val', 'Rate_DL_py_load_val',
        'Rate_OPT_py_load_test', 'Rate_DL_py_load_test',
        'Rate_DL_py_load_test_tflite', 'Rate_DL_py_load_test_tflite_mcu',
        
        'Indmax_OPT_py_load_test[0:5]', 'Indmax_DL_py_load_test[0:5]', 
        'Indmax_DL_py_load_test_tflite[0:5]', 'Indmax_DL_py_load_test_tflite_mcu[0:5]'
    ]

    output_csv = os.path.join(mcu_profiling_folder, f"profiling{dummy}{end_folder_Training_Size_dd_epochs}_{mcu_type['name']}.csv")

    if dummy == '':
        filename_Indmax_OPT_py_load_test = os.path.join(mcu_profiling_folder_outputdata, f"Indmax_OPT_py_load_test{end_folder_Training_Size_dd_epochs}.txt")
        filename_Indmax_DL_py_load_test = os.path.join(mcu_profiling_folder_outputdata, f"Indmax_DL_py_load_test{end_folder_Training_Size_dd_epochs}.txt")
        filename_Indmax_DL_py_load_test_tflite = os.path.join(mcu_profiling_folder_outputdata, f"Indmax_DL_py_load_test_tflite{end_folder_Training_Size_dd_epochs}.txt")
        filename_Indmax_DL_py_load_test_tflite_mcu = os.path.join(mcu_profiling_folder_outputdata, f"Indmax_DL_py_load_test_tflite_mcu{end_folder_Training_Size_dd_epochs}.txt")

        codebook_lists = [Indmax_OPT_py_load_test, Indmax_DL_py_load_test, Indmax_DL_py_load_test_tflite, Indmax_DL_py_load_test_tflite_mcu]
        filenames = [filename_Indmax_OPT_py_load_test, filename_Indmax_DL_py_load_test, filename_Indmax_DL_py_load_test_tflite, filename_Indmax_DL_py_load_test_tflite_mcu]

        for filename, codebook_list in zip(filenames, codebook_lists):
            with open(filename, 'w', encoding='utf-8') as f:
                for codebook in codebook_list:
                    f.write(str(codebook) + "\n")

    write_header = not os.path.exists(output_csv)

    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        try:
            print(Rate_DL_py_load_test_tflite)
            rate_dl_py_load_test_tflite_str = f"{Rate_DL_py_load_test_tflite.item():.6f}"
        except Exception:
            rate_dl_py_load_test_tflite_str = f"{Rate_DL_py_load_test_tflite:.6f}"

        try:
            Indmax_DL_py_load_test_tflite_str = Indmax_DL_py_load_test_tflite.tolist()[0:5], 
        except Exception:
            Indmax_DL_py_load_test_tflite_str = Indmax_DL_py_load_test_tflite[0:5], 

        row = {
            'timestamp': datetime.datetime.now().strftime("%d-%m-%y--%H:%M"),
            'end_folder_Training_Size_dd_epochs': end_folder_Training_Size_dd_epochs, 
            'K_DL': K_DL,
            'Error_does_not_fit': Error_does_not_fit,
            'Error_model_in_ram': Error_model_in_ram,
            'input_features':     input_features,
            'num_layers':         num_layers,
            'R':                  R,
            'hidden_units_list':  ' '.join(str(x) for x in hidden_units_list).ljust(10),
            'output_dim':         output_dim,
            'patience':           patience,
            'min_delta':          min_delta,
            'max_epochs':         max_epochs,
            'initial_epoch':      initial_epoch,
            #'model_h_size_int8_kb':    f"{model_h_size_int8_kb:.3f}",
            #'model_file_size_int8_kb': f"{model_file_size_int8_kb:.3f}",
            #'model_file_size_int8_mb': f"{model_file_size_int8_mb:.3f}",
            'RAM_KB':       f"{RAM_KB:.3f}",
            'Flash_MB':     f"{Flash_MB:.3f}",
            'CLK_FREQ_MHZ': f"{CLK_FREQ_MHZ:.3f}",
            'RAM_HW_KB':    f"{RAM_HW_KB:.3f}",
            'Flash_HW_MB':  f"{Flash_HW_MB:.3f}",
            'env_name':     env_name.ljust(40),

            # normalize_input
            'mean_norm': f"{mean_norm:.3f}",
            'perc50_norm': f"{perc50_norm:.3f}",
            'perc95_norm': f"{perc95_norm:.3f}",
            'std_norm': f"{std_norm:.3f}",

            # quantize_input
            'mean_quant': f"{mean_quant:.3f}",
            'perc50_quant': f"{perc50_quant:.3f}",
            'perc95_quant': f"{perc95_quant:.3f}",
            'std_quant': f"{std_quant:.3f}",

            # normalize_and_quantize_input
            'mean_normquant': f"{mean_normquant:.3f}",
            'perc50_normquant': f"{perc50_normquant:.3f}",
            'perc95_normquant': f"{perc95_normquant:.3f}",
            'std_normquant': f"{std_normquant:.3f}",

            # interpreter_invoke
            'mean_invoke': f"{mean_invoke:.3f}",
            'perc50_invoke': f"{perc50_invoke:.3f}",
            'perc95_invoke': f"{perc95_invoke:.3f}",
            'std_invoke': f"{std_invoke:.3f}",

            # dequantize_output
            'mean_dequant': f"{mean_dequant:.3f}",
            'perc50_dequant': f"{perc50_dequant:.3f}",
            'perc95_dequant': f"{perc95_dequant:.3f}",
            'std_dequant': f"{std_dequant:.3f}",

            # extract_codebook_index
            'mean_extract': f"{mean_extract:.3f}",
            'perc50_extract': f"{perc50_extract:.3f}",
            'perc95_extract': f"{perc95_extract:.3f}",
            'std_extract': f"{std_extract:.3f}",

            # extract_codebook_index_fast
            'mean_extract_fast': f"{mean_extract_fast:.3f}",
            'perc50_extract_fast': f"{perc50_extract_fast:.3f}",
            'perc95_extract_fast': f"{perc95_extract_fast:.3f}",
            'std_extract_fast': f"{std_extract_fast:.3f}",

            # total latency
            'mean_tot_latency': f"{mean_tot_latency:.3f}",
            'perc50_tot_latency': f"{perc50_tot_latency:.3f}",
            'perc95_tot_latency': f"{perc95_tot_latency:.3f}",
            'std_tot_latency': f"{std_tot_latency:.3f}",

            'mean_tot_latency_fast': f"{mean_tot_latency_fast:.3f}",
            'perc50_tot_latency_fast': f"{perc50_tot_latency_fast:.3f}",
            'perc95_tot_latency_fast': f"{perc95_tot_latency_fast:.3f}",
            'std_tot_latency_fast': f"{std_tot_latency_fast:.3f}",

            'Rate_OPT_py_load_val': f"{Rate_OPT_py_load_val.item():.3f}",
            'Rate_DL_py_load_val': f"{Rate_DL_py_load_val.item():.3f}", # serve per capire se c'è stato overfitting del modello sul test set
            'Rate_OPT_py_load_test': f"{Rate_OPT_py_load_test.item():.6f}",
            'Rate_DL_py_load_test': f"{Rate_DL_py_load_test.item():.6f}",
            'Rate_DL_py_load_test_tflite': rate_dl_py_load_test_tflite_str,
            'Rate_DL_py_load_test_tflite_mcu': f"{Rate_DL_py_load_test_tflite_mcu:.6f}",
            
            # subset
            'Indmax_OPT_py_load_test[0:5]': Indmax_OPT_py_load_test.tolist()[0:5], 
            'Indmax_DL_py_load_test[0:5]': Indmax_DL_py_load_test.tolist()[0:5], 
            'Indmax_DL_py_load_test_tflite[0:5]': Indmax_DL_py_load_test_tflite_str,
            'Indmax_DL_py_load_test_tflite_mcu[0:5]': Indmax_DL_py_load_test_tflite_mcu[0:5]
        }

        writer.writerow(row)
        print(json.dumps(row, indent=4))

# QUESTI METODI NON VANNO
#def usbipd(action, busid=None):
#    cmd = [
#        #"powershell.exe",
#        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
#        "-ExecutionPolicy", "Bypass",
#        "-File", "C:\\Users\\Work\\Desktop\\deepMIMO\\RIS\\usbipd_wrapper.ps1",
#        #"-File", "/mnt/c/Users/Work/Desktop/deepMIMO/RIS/usbipd_wrapper.ps1",
#        action
#    ]
#    if busid:
#        cmd.append(busid)
#
#    try:
#        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
#        print(out)
#        return out
#    except subprocess.CalledProcessError as e:
#        print(f"[ERROR] usbipd {action} failed (code {e.returncode}): {e.output}")
#        return None
#def usbipd(action, busid=None):
#    if action == "list":
#        cmd = ["/mnt/c/Windows/System32/cmd.exe", "/C", "usbipd list"]
#    elif action == "detach" and busid:
#        cmd = ["/mnt/c/Windows/System32/cmd.exe", "/C", f"usbipd detach --busid {busid}"]
#    elif action == "attach" and busid:
#        cmd = ["/mnt/c/Windows/System32/cmd.exe", "/C", f"usbipd attach --wsl --busid {busid}"]
#    else:
#        raise ValueError("Invalid action or missing busid")
#
#    try:
#        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
#        print(out)
#        return out
#    except subprocess.CalledProcessError as e:
#        print(f"[ERROR] usbipd {action} failed (code {e.returncode}): {e.output}")
#        return None
    

def windows_to_wsl(path_windows: str) -> str:
    """
    Converte un path Windows in un path WSL/Unix-like.
    
    Gestisce:
    - backslash → slash
    - spazi nel path
    - caratteri speciali
    """
    # Normalizza backslash in slash
    path_clean = path_windows.replace("\\", "/")

    try:
        # Usa wslpath per la conversione
        path_wsl = subprocess.check_output(
            ["wsl", "wslpath", "-u", path_clean],
            stderr=subprocess.PIPE
        ).decode().strip()
        return path_wsl
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Errore nella conversione del path '{path_windows}' a WSL.\n"
            f"stdout: {e.output}\nstderr: {e.stderr.decode()}"
        )


# ----- Run di una singola configurazione -----
def run_experiment(dummy, data_csv, x_sample, 
                   input_features, output_dim, num_layers, R, hidden_units_list, 
                   patience, min_delta, max_epochs, initial_epoch,
                   mean_array_filepath, variance_array_filepath, warmup_samples_for_statistics, xtest_npy_filename,
                   end_folder, end_folder_Training_Size_dd):
    #print('\n*** run_experiment')

    # %%
    # Crea o recupera nome del modello tflite
    end_folder_Training_Size_dd_epochs, model_name_suffix, model_type_load, model_path_tflite, model_path_keras = get_model_paths(dummy, 
                                                                                                                                  end_folder_Training_Size_dd, max_epochs, num_layers, 
                                                                                                                                  R, input_features, hidden_units_list, output_dim)

    if train_model_flag == 1 and os.path.exists(model_path_keras): # Controlla se il file esiste
        print(f"\nModello già allenato: {model_path_keras}")
        return
    
    tensorboard_logs = os.path.join(tensorboard_dir, f"tensorboard_logs_{model_type_load}")
    training_history_json = os.path.join(training_history_dir, f"training_history_{model_type_load}.json")
    
    # %%
    # Allena o carica modello pre-allenato
    if dummy == 'dummy_':
        model_py = build_mlp(input_features, output_dim, num_layers, hidden_units_list)
        model_py.compile(optimizer='adam', loss='mse')
        
        # Converti modello keras in tflite int8
        convert_to_tflite_int8(model_py, x_sample, model_path_tflite)

        Rate_OPT_py_load_val = 0
        Rate_DL_py_load_val = 0
        Rate_OPT_py_load_test = 0
        Rate_DL_py_load_test = 0
        Rate_DL_py_load_test_tflite = 0
        Indmax_OPT_py_load_test = [0, 0, 0, 0, 0]
        Indmax_DL_py_load_test = [0, 0, 0, 0, 0]
        Indmax_DL_py_load_test_tflite = [0, 0, 0, 0, 0]
        YValidation_un_test = 0

    else:

        if train_model_flag == 1 and load_model_flag == 0 and ISWINDOWS == 0:
            DL_training_4_v3_test.main(
                                        My, Mz, load_model_flag, max_epochs, initial_epoch,
                                        train_model_flag, predict_loaded_model_flag,
                                        convert_model_flag, Training_Size_dd,
                                        input_features, output_dim, num_layers, hidden_units_list,
                                        init_learning_rate, min_learning_rate, factor, patience, min_delta,
                                        mean_array_filepath, variance_array_filepath, test_set_size, small_samples,
                                        end_folder, end_folder_Training_Size_dd, 
                                        end_folder_Training_Size_dd_epochs, model_name_suffix, model_path_tflite, model_path_keras,
                                        DL_dataset_folder, network_folder_in,
                                        network_folder_out_RateDLpy, network_folder_out_RateDLpy_TFLite, 
                                        mcu_profiling_folder_test_data, mcu_profiling_folder_test_data_normalized, mcu_profiling_folder_scaler, 
                                        tensorboard_logs, training_history_json,
                                        save_files_flag_master, save_files_flag_master_once)     
        elif train_model_flag == 0 and load_model_flag == 1:
            if ISWINDOWS == 0:
                #model_py, \
                Rate_OPT_py_load_val,  Rate_DL_py_load_val, \
                Rate_OPT_py_load_test, Rate_DL_py_load_test, \
                Rate_DL_py_load_test_tflite, \
                Indmax_OPT_py_load_test, Indmax_DL_py_load_test, \
                Indmax_DL_py_load_test_tflite, \
                YValidation_un_test = DL_training_4_v3_test.main(
                                                My, Mz, load_model_flag, max_epochs, initial_epoch,
                                                train_model_flag, predict_loaded_model_flag,
                                                convert_model_flag, Training_Size_dd,
                                                input_features, output_dim, num_layers, hidden_units_list,
                                                init_learning_rate, min_learning_rate, factor, patience, min_delta,
                                                mean_array_filepath, variance_array_filepath, test_set_size, small_samples,
                                                end_folder, end_folder_Training_Size_dd, 
                                                end_folder_Training_Size_dd_epochs, model_name_suffix, model_path_tflite, model_path_keras,
                                                DL_dataset_folder, network_folder_in,
                                                network_folder_out_RateDLpy, network_folder_out_RateDLpy_TFLite, 
                                                mcu_profiling_folder_test_data, mcu_profiling_folder_test_data_normalized, mcu_profiling_folder_scaler, 
                                                tensorboard_logs, training_history_json,
                                                save_files_flag_master, save_files_flag_master_once)
            else: # ISWINDOWS == 1
                print("***** ENTRO NEL NUOVO SCRIPT")

                def call_dl_training_4_v3_test_wsl(params_dict, output_json_path):
                    # Salva i parametri in un file JSON
                    params_path = f"params_dl_training_4_v3_test{end_folder_Training_Size_dd_epochs}.json"
                    with open(params_path, "w", encoding="utf-8") as f:
                        json.dump(params_dict, f)

                    wsl_script_path = "/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/DL_training_4_v3_test.py"
                    process = subprocess.Popen([
                        "wsl", "bash", "-lc", 
                        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate deepmimo && /home/work_wsl/miniconda3/envs/deepmimo/bin/python -u /mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/DL_training_4_v3_test.py"
                        f" --params {params_path}"], # --output {output_json_path}"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        errors='replace',
                        bufsize=1)

                    #for line in process.stdout:
                    #    print(line, end="")       # scrive a terminale
                    #    #logfile.write(line)       # scrive sul file
                    #    #logfile.flush()
                    #
                    #process.wait()

                    # stampa in tempo reale
                    for line in iter(process.stdout.readline, ''):
                        print(line, end='')
                    # lettura in binario, decodifica robusta
                    #for line in iter(process.stdout.readline, b''):
                    #    try:
                    #        print(line.decode('utf-8'), end='')
                    #    except UnicodeDecodeError:
                    #        # se non si può decodificare, ignora i byte non decodificabili
                    #        print(line.decode('utf-8', errors='replace'), end='')

                    process.stdout.close()
                    return_code = process.wait()

                    print("***** PROCESSO WSL RITORNATO")
                    
                    # Carica i risultati SOLO dopo che il processo è terminato
                    with open(output_json_path, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    return results

                params_dict = {
                    "My": My,
                    "Mz": Mz,
                    "load_model_flag": load_model_flag,
                    "max_epochs": max_epochs,
                    "initial_epoch": initial_epoch,
                    "train_model_flag": train_model_flag,
                    "predict_loaded_model_flag": predict_loaded_model_flag,
                    "convert_model_flag": convert_model_flag,
                    "Training_Size_dd": Training_Size_dd,
                    "input_features": input_features,
                    "output_dim": output_dim,
                    "num_layers": num_layers,
                    "hidden_units_list": hidden_units_list,
                    "init_learning_rate": init_learning_rate,
                    "min_learning_rate": min_learning_rate,
                    "factor": factor,
                    "patience": patience,
                    "min_delta": min_delta,
                    "mean_array_filepath": windows_to_wsl(mean_array_filepath),
                    "variance_array_filepath": windows_to_wsl(variance_array_filepath),
                    "test_set_size": test_set_size,
                    "small_samples": small_samples,
                    "end_folder": windows_to_wsl(end_folder),
                    "end_folder_Training_Size_dd": windows_to_wsl(end_folder_Training_Size_dd),
                    "end_folder_Training_Size_dd_epochs": windows_to_wsl(end_folder_Training_Size_dd_epochs),
                    "model_name_suffix": model_name_suffix,
                    "model_path_tflite": windows_to_wsl(model_path_tflite),
                    "model_path_keras": windows_to_wsl(model_path_keras),
                    "DL_dataset_folder": windows_to_wsl(DL_dataset_folder),
                    "network_folder_in": windows_to_wsl(network_folder_in),
                    "network_folder_out_RateDLpy": windows_to_wsl(network_folder_out_RateDLpy),
                    "network_folder_out_RateDLpy_TFLite": windows_to_wsl(network_folder_out_RateDLpy_TFLite),
                    "mcu_profiling_folder_test_data": windows_to_wsl(mcu_profiling_folder_test_data),
                    "mcu_profiling_folder_test_data_normalized": windows_to_wsl(mcu_profiling_folder_test_data_normalized),
                    "mcu_profiling_folder_scaler": windows_to_wsl(mcu_profiling_folder_scaler),
                    "tensorboard_logs": windows_to_wsl(tensorboard_logs),
                    "training_history_json": windows_to_wsl(training_history_json),
                    "save_files_flag_master": save_files_flag_master,
                    "save_files_flag_master_once": save_files_flag_master_once
                }
                output_json_path = os.path.join(output_folder, f"output_dl_training_4_v3_test{end_folder_Training_Size_dd_epochs}.json")
                results = call_dl_training_4_v3_test_wsl(params_dict, output_json_path)

                # Poi estrai i risultati come prima:
                #model_py = None  # Non puoi serializzare direttamente il modello, usa solo i risultati numerici
                Rate_OPT_py_load_val =          np.array(results["Rate_OPT_py_load_val"])
                Rate_DL_py_load_val =           np.array(results["Rate_DL_py_load_val"])
                Rate_OPT_py_load_test =         np.array(results["Rate_OPT_py_load_test"])
                Rate_DL_py_load_test =          np.array(results["Rate_DL_py_load_test"])
                Rate_DL_py_load_test_tflite =   np.array(results["Rate_DL_py_load_test_tflite"])
                Indmax_OPT_py_load_test =       np.array(results["Indmax_OPT_py_load_test"])
                Indmax_DL_py_load_test =        np.array(results["Indmax_DL_py_load_test"])
                Indmax_DL_py_load_test_tflite = np.array(results["Indmax_DL_py_load_test_tflite"])
                YValidation_un_test =           np.array(results["YValidation_un_test"])

                print(Rate_OPT_py_load_test)
                print(Rate_DL_py_load_test)
                print(Rate_DL_py_load_test_tflite)

        else:
            Rate_OPT_py_load_val = 0
            Rate_DL_py_load_val = 0
            Rate_OPT_py_load_test = 0
            Rate_DL_py_load_test = 0
            Rate_DL_py_load_test_tflite = 0
            Indmax_OPT_py_load_test = [0, 0, 0, 0, 0]
            Indmax_DL_py_load_test = [0, 0, 0, 0, 0]
            Indmax_DL_py_load_test_tflite = [0, 0, 0, 0, 0]
            YValidation_un_test = DL_training_4_v3_test.main(
                                            My, Mz, load_model_flag, max_epochs, initial_epoch,
                                            train_model_flag, predict_loaded_model_flag,
                                            convert_model_flag, Training_Size_dd,
                                            input_features, output_dim, num_layers, hidden_units_list,
                                            init_learning_rate, min_learning_rate, factor, patience, min_delta,
                                            mean_array_filepath, variance_array_filepath, test_set_size, small_samples,
                                            end_folder, end_folder_Training_Size_dd, 
                                            end_folder_Training_Size_dd_epochs, model_name_suffix, model_path_tflite, model_path_keras,
                                            DL_dataset_folder, network_folder_in,
                                            network_folder_out_RateDLpy, network_folder_out_RateDLpy_TFLite, 
                                            mcu_profiling_folder_test_data, mcu_profiling_folder_test_data_normalized, mcu_profiling_folder_scaler, 
                                            tensorboard_logs, training_history_json,
                                            save_files_flag_master, save_files_flag_master_once)

        #model_py.summary()
            
    # %%
    if profiling_flag == 1 and train_model_flag == 0:
        # Esporta il modello in formato header file per TFLM
        export_to_c(model_path_tflite)
        
        #if load_model_flag == 1:
        #    model_h_size_int8_kb, model_file_size_int8_kb, model_file_size_int8_mb = get_model_info_precise(model_py, model_path_tflite, save_dir=mcu_lib_model_folder)
        #else:
        #    model_h_size_int8_kb = 0
        #    model_file_size_int8_kb = 0
        #    model_file_size_int8_mb = 0
        
        #change_config_file_2(mcu_include_config, mean_array_filepath, variance_array_filepath)
        change_config_file_3(mcu_include_config, mean_array_filepath)

        # Lancia compilazione e upload firmware in maniera bloccante
        compilation_logfile = os.path.join(mcu_folder, 'compilation_'+model_type_load+'.txt')

        i = 1 # TODO: must be 1
        #while i <= 4:
        if mcu_type['name'] == 'nucleo-f446ze':
            i_max = 1
        else:
            i_max = 2
        while i <= i_max: # TODO: must be 2
            # prima provi con modello in ram
            # poi provi con modello in flash
            if i == 1:
                #if ISWINDOWS:
                #    env_name = mcu_type['name']+'-o3-windows'
                #else:
                #    env_name = mcu_type['name']+'-o3'
                env_name = mcu_type['name']+'-o3'
            elif i == 2:
                #if ISWINDOWS:
                #    env_name = mcu_type['name']+'-o3-modelinram-windows'
                #else:
                #    #env_name = mcu_type['name']+'-o3-modelinram'
                #    env_name = mcu_type['name']+'-o3-modelinram'
                env_name = mcu_type['name']+'-o3-modelinram'
            #elif i == 3:
            #    env_name = mcu_type['name']+'-o3-modelinram'
            #elif i == 4:
            #    env_name = mcu_type['name']+'-o3-unrollnorm-modelinram'

            if compile_and_upload_flag == 1:
                print(f"\n*** Lancio pio run {i})... Segui gli avanzamenti qui: {compilation_logfile}")

                com_port = find_com_port_by_serial_number(mcu_type['serial_number'])
                if com_port == -1:
                    return

                print(f"\n*** Rimuovi cartella .pio")    
                firmware_src_path = os.path.join(mcu_folder, ".pio", "build", env_name, "src")
                if os.path.exists(firmware_src_path):
                    shutil.rmtree(firmware_src_path, ignore_errors=True)

                print(f"\n*** Rimuovi file in .pio")   
                for filename in ["firmware.bin", "firmware.elf", "firmware.map", "src/main.cpp.o", "src/main.cpp.d"]:
                    firmware_file_path = os.path.join(mcu_folder, ".pio", "build", env_name, filename)
                    if os.path.exists(firmware_file_path):
                        os.remove(firmware_file_path)

                with open(compilation_logfile, "w", encoding="utf-8") as logfile:

                    process = subprocess.Popen(
                        #["pio", "run", "--environment", mcu_type['name'], "--project-dir", mcu_folder],
                        #["pio", "run", "--environment", mcu_type['name']+'', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", mcu_type['name']+'-o0', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", mcu_type['name']+'-o3', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", mcu_type['name']+'-o3-unrollallauto', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", mcu_type['name']+'-o3-unrollallforced', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--evironment", mcu_type['name']+'-o3-unrollallforced-selected', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", mcu_type['name']+'-o3-unroll-selected', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", mcu_type['name']+'-o3-unrollnorm-espnn', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", mcu_type['name']+'-o3-unrollnorm', "--project-dir", mcu_folder, "--target", "upload"],
                        #["pio", "run", "--environment", env_name, "--project-dir", mcu_folder, "--target", "upload", '-v'],
                        ["pio", 
                            "run", "--environment", env_name,  "--project-dir",  mcu_folder, "--target", "upload", 
                                   "--upload-port", com_port,  "--monitor-port", com_port],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )

                    for line in process.stdout:
                        print(line, end="")       # scrive a terminale
                        logfile.write(line)       # scrive sul file
                        logfile.flush()  # <-- aggiungi questa riga
                    
                    ret = process.wait() # <-- è bloccante: quando ritorna significa che pio è ritornato, cioè la programmazione è stata terminata
                    print(f"[INFO] Process exited with code {ret}") 

            print("In attesa prima del parser")
            time.sleep(5)

            # Lancia il parser per recuperare RAM e ROM da compilation output
            # La Flash include non solo il modello perciò sarà maggiore di model_size_int8_mb
            RAM_KB, Flash_MB, CLK_FREQ_MHZ, RAM_HW_KB, Flash_HW_MB, Error_does_not_fit = parse_compilation_logfile(compilation_logfile)

            if Error_does_not_fit == 0:

                #time.sleep(5)

                com_port = find_com_port_by_serial_number(mcu_type['serial_number'])
                if com_port == -1:
                    return

                start_time = time.time()
                # Lancia serial feeded and logger per calcolare la latenza
                mean_norm, perc50_norm, perc95_norm, std_norm, \
                mean_quant, perc50_quant, perc95_quant, std_quant, \
                mean_normquant, perc50_normquant, perc95_normquant, std_normquant, \
                mean_invoke, perc50_invoke, perc95_invoke, std_invoke, \
                mean_dequant, perc50_dequant, perc95_dequant, std_dequant, \
                mean_extract, perc50_extract, perc95_extract, std_extract, \
                mean_extract_fast, perc50_extract_fast, perc95_extract_fast, std_extract_fast, \
                mean_tot_latency, perc50_tot_latency, perc95_tot_latency, std_tot_latency, \
                mean_tot_latency_fast, perc50_tot_latency_fast, perc95_tot_latency_fast, std_tot_latency_fast, \
                Indmax_DL_py_load_test_tflite_mcu, Rate_DL_py_load_test_tflite_mcu, Error_model_in_ram = serial_feeder_and_logger.main(dummy,
                                                                                                                           com_port, mcu_type, input_features,
                                                                                                                           data_csv, test_set_size, small_samples,
                                                                                                                           warmup_samples_for_statistics, 
                                                                                                                           YValidation_un_test, 
                                                                                                                           xtest_npy_filename, 
                                                                                                                           end_folder_Training_Size_dd_epochs, model_name_suffix,
                                                                                                                           network_folder_out_RateDLpy_TFLite_mcu,
                                                                                                                           mcu_profiling_logfile)
                
                elapsed_time = time.time() - start_time
                print(f"serial_feeder_and_logger completed in {elapsed_time / 60:.2f} minutes.")                                                                                                    
        
                if Error_model_in_ram != 0 and i == 1:
                    print("*** ATTENZIONE: Error_model_in_ram == 1 and i == 1")
                elif Error_model_in_ram != 0 and i == 2:
                    print("*** ATTENZIONE: Error_model_in_ram == 1 and i == 2")
            
            if Error_does_not_fit != 0 or Error_model_in_ram != 0:
                mean_norm = 0
                perc50_norm = 0
                perc95_norm = 0
                std_norm = 0                                            
                mean_quant = 0
                perc50_quant = 0
                perc95_quant = 0
                std_quant = 0
                mean_normquant = 0
                perc50_normquant = 0
                perc95_normquant = 0
                std_normquant = 0
                mean_invoke = 0
                perc50_invoke = 0
                perc95_invoke = 0
                std_invoke = 0                                     
                mean_dequant = 0
                perc50_dequant = 0
                perc95_dequant = 0
                std_dequant = 0                                  
                mean_extract=0
                perc50_extract = 0
                perc95_extract = 0
                std_extract = 0                                  
                mean_extract_fast = 0
                perc50_extract_fast = 0
                perc95_extract_fast = 0
                std_extract_fast = 0             
                mean_tot_latency = 0
                perc50_tot_latency = 0
                perc95_tot_latency = 0
                std_tot_latency = 0
                mean_tot_latency_fast = 0
                perc50_tot_latency_fast = 0
                perc95_tot_latency_fast = 0
                std_tot_latency_fast = 0
                Indmax_DL_py_load_test_tflite_mcu = [0, 0, 0, 0, 0]
                Rate_DL_py_load_test_tflite_mcu = 0

                if Error_does_not_fit != 0:
                    Error_model_in_ram = 0


            # %%
            # Appendi risultati nel file output_csv
            save_results_v3(dummy, end_folder_Training_Size_dd_epochs, K_DL,
                            Error_does_not_fit, Error_model_in_ram,
                            input_features, num_layers, R, hidden_units_list, output_dim,
                            patience, min_delta, max_epochs, initial_epoch,
                            #model_h_size_int8_kb, model_file_size_int8_kb, model_file_size_int8_mb,
                            RAM_KB, Flash_MB, CLK_FREQ_MHZ, RAM_HW_KB, Flash_HW_MB, env_name,
                            mean_norm, perc50_norm, perc95_norm, std_norm,
                            mean_quant, perc50_quant, perc95_quant, std_quant,
                            mean_normquant, perc50_normquant, perc95_normquant, std_normquant,
                            mean_invoke, perc50_invoke, perc95_invoke, std_invoke,
                            mean_dequant, perc50_dequant, perc95_dequant, std_dequant,
                            mean_extract, perc50_extract, perc95_extract, std_extract,
                            mean_extract_fast, perc50_extract_fast, perc95_extract_fast, std_extract_fast,
                            mean_tot_latency, perc50_tot_latency, perc95_tot_latency, std_tot_latency,
                            mean_tot_latency_fast, perc50_tot_latency_fast, perc95_tot_latency_fast, std_tot_latency_fast,
                            Rate_OPT_py_load_val, Rate_DL_py_load_val,
                            Rate_OPT_py_load_test, Rate_DL_py_load_test,
                            Rate_DL_py_load_test_tflite, Rate_DL_py_load_test_tflite_mcu,
                            Indmax_OPT_py_load_test, Indmax_DL_py_load_test, 
                            Indmax_DL_py_load_test_tflite, Indmax_DL_py_load_test_tflite_mcu)

            i += 1

# %%

# --- ESEMPIO USO --- #
if __name__ == "__main__":

    start_time = time.time()

    if profiling_flag == 1:
        mcu_folder = os.path.join(pio_projects_folder, mcu_type['name'])
        mcu_include_config = os.path.join(mcu_folder, 'include', 'config.h') 
        #mcu_include_config = os.path.join(mcu_folder, 'lib/lib-luca/config.cc') 
        mcu_lib_libluca_folder = os.path.join(mcu_folder, 'lib', 'lib-luca') 
        mcu_lib_model_folder = os.path.join(mcu_folder, 'lib', 'model')

        folders = [
        mcu_folder,
        mcu_lib_libluca_folder,
        mcu_lib_model_folder
        ]

        for folder in folders:
            if not os.path.exists(folder):  # Controlla se la cartella esiste
                os.makedirs(folder, exist_ok=True)  # Crea la cartella se non esiste
                print(f"\nCartella creata: {folder}")
            else:
                print(f"La cartella esiste già: {folder}")

    #if not os.path.exists(mcu_lib_model_folder):  # Controlla se la cartella esiste
    #    os.makedirs(mcu_lib_model_folder, exist_ok=True)  # Crea la cartella se non esiste
    #    os.system('pio project init --project-dir ' + mcu_folder + ' --board ' + mcu_type['name'] + ' --ide arduino' + \
    #              ' --project-option="upload_protocol = esptool" ' \
    #              '--project-option="upload_port = /dev/ttyUSB0" ' \
    #              '--project-option="monitor_port = /dev/ttyUSB0" ' \
    #              '--project-option="monitor_speed = 115200" ' \
    #              '--project-option="build_flags = ' \
    #              '-Ilib/tflite-micro ' \
    #              '-Ilib/tflite-micro/tensorflow ' \
    #              '-Ilib/tflite-micro/tensorflow/lite/c ' \
    #              '-Ilib/tflite-micro/tensorflow/lite/micro ' \
    #              '-Ilib/tflite-micro/tensorflow/lite/schema" ')
    #    print(f"\nProgetto pio creato: {mcu_folder}")

    if train_model_flag == 1 and load_model_flag == 0:
        run_tensorboard()

    # %%
    for M_bar in active_cells:

        end_folder = '_seed' + str(seed) + '_grid' + str(Ur_rows[1]) + '_M' + str(My) + str(Mz) + '_Mbar' + str(M_bar)
        end_folder_Training_Size_dd = end_folder + '_' + str(Training_Size_dd)
                
        input_features = M_bar * K_DL * 2 #    8 celle attive x 64 subcarriers x 2 (real/img) = 1024

        mean_array_filepath = os.path.join(mcu_profiling_folder_scaler, f"{dummy}mean{end_folder_Training_Size_dd}.npy")
        variance_array_filepath = os.path.join(mcu_profiling_folder_scaler, f"{dummy}variance{end_folder_Training_Size_dd}.npy")

        print("\n---------------------------------------------------------\n")

        if dummy == 'dummy_':

            print(f"\n*** Dummy mode")

            # Sample fake data for quantization
            x_sample = np.random.rand(10, input_features).astype(np.float32)
            # Sample mean and variance for normalization
            #mean_array = np.random.rand(1, input_features).astype(np.float32)
            mean_array = np.random.rand(1, 1).astype(np.float32)
            #variance_array = np.random.rand(1, input_features).astype(np.float32)

            data_csv = os.path.join(mcu_profiling_folder_test_data, f"{dummy}data.npy")

            np.save(data_csv, x_sample)
            np.save(mean_array_filepath, mean_array)
            #np.save(variance_array_filepath, variance_array)    

            xtest_npy_filename = ''
        else:
            print(f"\n*** Normal mode")

            x_sample = 0
            data_csv = 0
            xtest_npy_filename = os.path.join(mcu_profiling_folder_test_data, f"test_set{end_folder_Training_Size_dd}.npy")

        for output_dim in output_dims:

            if profiling_flag == 1:
                change_config_file_1(mcu_include_config, input_features, output_dim, mcu_type['baud_rate'])
                change_lib_file_1(mcu_lib_libluca_folder, input_features, output_dim)
            
            once_zero_layers = 0 # serve per evitare che si iteri su R quando num_layers = 0

            for R in R_list:
                
                for num_layers in num_layers_list:

                    if num_layers == 0 and once_zero_layers == 1:
                        continue
                    else:
                        once_zero_layers = 1

                    hidden_units_list = [int(input_features/R), int(4*input_features/R), int(4*input_features/R)]     # Numero di neuroni per layer

                    #patience = 3 * 4/(1+num_layers)
                    #patience = patience_list[num_layers]
                    #min_delta = min_delta_list[num_layers]
                    #max_epochs = max_epochs_list[num_layers]

                    # %%
                    print(f"\n--> Profiling: act_cell={M_bar}, in={input_features}, out={output_dim}, layers={num_layers}, R={R}, hidden_units={hidden_units_list}")
                    print(f"num_layers={num_layers}: init_lr={init_learning_rate}, min_lr={min_learning_rate}, factor={factor}, patience={patience}, min_delta={min_delta}")
                    run_experiment(dummy, data_csv, x_sample,
                                   input_features, output_dim, num_layers, R, hidden_units_list, 
                                   patience, min_delta, max_epochs, initial_epoch,
                                   mean_array_filepath, variance_array_filepath, warmup_samples_for_statistics, xtest_npy_filename,
                                   end_folder, end_folder_Training_Size_dd)
                    
    elapsed_time = time.time() - start_time
    if profiling_flag == 1:
        print(f"Script debug={debug}, test_set_size={test_set_size}, mcu_type={mcu_type['name']} completed in {elapsed_time / 60:.2f} minutes ({elapsed_time / 3600:.2f} hours)")
    else:
        print(f"Script debug={debug}, test_set_size={test_set_size} completed in {elapsed_time / 60:.2f} minutes ({elapsed_time / 3600:.2f} hours)")