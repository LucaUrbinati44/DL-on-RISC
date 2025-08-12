import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)

import csv
import os
import time
import io
import json
from contextlib import redirect_stdout
import random
from statistics import mean, stdev
import re
import h5py
from tensorflow.keras.saving import load_model
from ai_edge_litert.interpreter import Interpreter

print("TensorFlow version:", tf.__version__)

# Imposta il seed globale
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Assegna i valori degli argomenti alle variabili
M_bar=8
Ur_rows = [1000, 1200]
My_ar = [32]
Mz_ar = [32]
My = My_ar[0]
Mz = Mz_ar[0]
max_epochs_load = 60
Training_Size = [10000]
Training_Size_dd = Training_Size[0]

end_folder = '_seed' + str(seed) + '_grid' + str(Ur_rows[1]) + '_M' + str(My) + str(Mz) + '_Mbar' + str(M_bar)
end_folder_Training_Size_dd = end_folder + '_' + str(Training_Size_dd)

base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'

input_folder = base_folder + 'Output Matlab/'

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
figure_folder = output_folder + 'Figures/'
profiling_estimation_folder = output_folder + 'Profiling_Search_Estimation/'
profiling_mcu_folder = output_folder + 'Profiling_Search_MCU/'
pio_projects_folder = '/mnt/c/Users/Work/Documents/PlatformIO/Projects/'
header_folder = 'tensorflow/lite/micro/examples/ml_on_risc/c_models'
test_data_npy_path = output_folder + 'Test_data/'

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
    figure_folder,
    profiling_estimation_folder,
    profiling_mcu_folder,
    pio_projects_folder,
    test_data_npy_path
]

for folder in folders:
    if not os.path.exists(folder):  # Controlla se la cartella esiste
        os.makedirs(folder, exist_ok=True)  # Crea la cartella se non esiste
        print(f"\nCartella creata: {folder}")
    #else:
    #    print(f"La cartella esiste già: {folder}")

# ----- Costruzione del modello MLP parametrico -----
def build_mlp(input_dim, output_dim, num_layers, hidden_units_list):
    model = tf.keras.Sequential([tf.keras.Input(shape=(input_dim,), name='input')])
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(hidden_units_list[i], activation='relu', name=f'hidden_{i}'))
    model.add(tf.keras.layers.Dense(output_dim, activation=None, name='output'))
    return model

# ----- Conversione in TF-Lite INT8 -----
def convert_to_tflite_int8(model, x_sample):
    def representative_data_gen():
        for i in range(min(100, len(x_sample))):
            yield [x_sample[i:i+1].astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()

def get_model_path_tflite():
    print('*** get_model_path_tflite')
    end_folder_Training_Size_dd_max_epochs_load = end_folder_Training_Size_dd + '_' + str(max_epochs_load)
    model_type_load = 'model_py_test' + end_folder_Training_Size_dd_max_epochs_load
    model_path_tflite = saved_models_tflite + model_type_load + '_quant.tflite'
    return end_folder_Training_Size_dd_max_epochs_load, model_type_load, model_path_tflite

def mse_custom(y_true, y_pred):
    # Calcola l'errore quadratico tra vero e predetto
    squared_error = tf.square(y_true - y_pred)  # shape: (batch_size, output_dim)=6200,1024

    # Somma degli errori lungo l'ultima dimensione (output_dim)
    sum_squared_error = tf.reduce_sum(squared_error, axis=-1)  # shape: (batch_size,)=6200

    # Media su tutto il batch
    loss = 0.5 * tf.reduce_mean(sum_squared_error)  # scalar
    return loss

# ----- Export test data for inference -----
def export_test_data(model_path_tflite, end_folder_Training_Size_dd_max_epochs_load, size='small'):
    print('*** export_test_data')
    xtest_npy_filename = test_data_npy_path + 'test_set' + end_folder_Training_Size_dd + '.npy'
    xtest = np.load(xtest_npy_filename)
    print(xtest.shape)
    
    if size == 'small':
        xtest_size = 10
    else:
        xtest_size = xtest.shape[0]

    print(xtest[:xtest_size,:].shape)

    xtest_renode_filename = test_data_renode_path + 'test_set_' + size + '.data'
    with open(xtest_renode_filename, 'w') as f:
        # map(str, sample): trasforma ogni elemento del vettore sample in una stringa.
        # ' '.join(...): concatena tutte queste stringhe, separandole con uno spazio.
        f.write(' '.join([f'ch{i+1}' for i in range(xtest.shape[1])]) + '\n')
        for sample in xtest[:xtest_size,:]:
            line = ','.join(map(str, sample))
            f.write(line + '\n')
    #with open(xtest_renode_filename, 'w') as f:
    #    f.write('data\n')  # intestazione richiesta per il campo BinaryData
    #    for sample in xtest[:xtest_size, :]:
    #        # Conversione delle xtest.shape[1] features float32 in una sola stringa esadecimale continua per il formato Renode Binary Data 
    #        sample_bytes = sample.astype(np.float32).tobytes()
    #        hex_str = sample_bytes.hex()
    #        f.write(f'{hex_str}\n')

    xtest_header_filename = test_data_renode_path + 'test_set_' + size + '_' + str(xtest_size) + '.h'
    num_samples = xtest_size
    sample_size = xtest.shape[1]
    with open(xtest_header_filename, 'w') as f:
        f.write("#ifndef SAMPLE_DATA_H\n")
        f.write("#define SAMPLE_DATA_H\n\n")
        f.write(f"#define NUM_SAMPLES {num_samples}\n")
        #f.write(f"#define SAMPLE_SIZE {sample_size}\n\n")
        #f.write("const float samples[NUM_SAMPLES][SAMPLE_SIZE] = {\n")
        f.write("const float samples[NUM_SAMPLES][INPUT_FEATURE_SIZE] = {\n")
        
        for i, sample in enumerate(xtest[:xtest_size]):
            # converto ogni valore in stringa formattata a 6 decimali
            sample_str = ", ".join(f"{x:.6f}f" for x in sample)
            if i == num_samples - 1:
                f.write(f"    {{ {sample_str} }}\n")
            else:
                f.write(f"    {{ {sample_str} }},\n")
        f.write("};\n\n")
        f.write("#endif // SAMPLE_DATA_H\n")

    # Esempio di utilizzo:
    # xtest = np.random.rand(5, 1024)  # esempio di 5 sample da 1024 feature
    # generate_header_from_array(xtest, "sample_data.h")
    

    # load tflite model
    with open(model_path_tflite, 'rb') as f:
        tflite_quant_model = f.read()
    print('TFLite model loaded')

    from tensorflow.lite.python import schema_py_generated as schema_fb

    tflite_model = schema_fb.Model.GetRootAsModel(tflite_quant_model, 0)

    # Gets metadata from the model file.
    for i in range(tflite_model.MetadataLength()):
        meta = tflite_model.Metadata(i)
        if meta.Name().decode("utf-8") == "min_runtime_version":
            buffer_index = meta.Buffer()
            metadata = tflite_model.Buffers(buffer_index)
            min_runtime_version_bytes = metadata.DataAsNumpy().tobytes()
            print(min_runtime_version_bytes)
        
    #return

    # inference using tflite model with xtest
    #Loading and running a LiteRT model involves the following steps:
    #1) Loading the model into memory.
    #2) Building an Interpreter based on an existing model.
    # Load the LiteRT model and allocate tensors.
    interpreter = Interpreter(model_content=tflite_quant_model)
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
    print(f'input_scale: {input_scale:.6f}')
    print(f'input_zero_point: {input_zero_point:.6f}')
    output_scale, output_zero_point = output_details[0]['quantization']
    print(f'output_scale: {output_scale:.6f}')
    print(f'output_zero_point: {output_zero_point:.6f}')
        
    input_float = xtest[:xtest_size,:]
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
        print('\nfloat_input:', ' '.join([f"{x:.6f}" for x in sample]))
        sample = np.expand_dims(sample, axis=0)  # shape: (1, 1024)
        input_int8 = np.round(sample / input_scale + input_zero_point).astype(np.int8)
        print('\nquantized_input:', ' '.join([f"{x}" for x in input_int8.flatten()]))
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
        output_deq = (output_int8.astype(np.float32) - output_zero_point) * output_scale
        print('\nquantized_output:', ' '.join([f"{x}" for x in output_int8.flatten()]))
        print('\ndequantized_output:', ' '.join([f"{x:.6f}" for x in output_deq[0]]))
        #print('output_deq.shape:', output_deq.shape)
        #print('np.max(output_deq):', np.max(output_deq))
        #print('np.min(output_deq):', np.min(output_deq))
        output_float.append(output_deq[0])  # output_deq.shape: (1, 1024) -> prendi [0]
        
        print(f"\ncodebook_index {i+1}: {np.argmax(output_deq[0])}\n")

    YPredicted = np.stack(output_float, axis=0)  # shape: (3100, 1024)
    #print(f'YPredicted.shape: {YPredicted.shape}')

    Indmax_DL_py = np.argmax(YPredicted, axis=1)
    #print(f'Indmax_DL_py.shape: {Indmax_DL_py.shape}')
    # Questi devono essere numeri interi
    print(f'np.min(Indmax_DL_py): {np.min(Indmax_DL_py)}')
    print(f'np.max(Indmax_DL_py): {np.max(Indmax_DL_py)}')
    print(Indmax_DL_py[0:5])

    # save int8 inference results as golden output on a .data file for TFLM comparison
    Indmax_DL_py_renode_filename_int8 = test_data_renode_path + 'output_golden_codebook_int8_' + size + '.data'
    with open(Indmax_DL_py_renode_filename_int8, 'w') as f:
        # trasforma ogni elemento in una stringa
        for sample in Indmax_DL_py[:xtest_size]:
            f.write(str(sample) + '\n')

    # save float32 inference results as golden output on a .data file for TFLM comparisong
    filename_Indmax_DL_py = network_folder_out_RateDLpy + 'Indmax_DL_py' + '_test' + end_folder_Training_Size_dd_max_epochs_load + '.mat'
    with h5py.File(filename_Indmax_DL_py, 'r') as f:
        Indmax_DL_py = np.array(f['Indmax_DL_py'][:], dtype=np.float32)
        print(f"\nIndmax_DL_py loaded")
    Indmax_DL_py_renode_filename_float32 = test_data_renode_path + 'output_golden_codebook_float32_' + size + '.data'
    with open(Indmax_DL_py_renode_filename_float32, 'w') as f:
        # trasforma ogni elemento in una stringa
        for sample in Indmax_DL_py[:xtest_size]:
            f.write(str(int(sample)) + '\n')


# ----- Export TF-Lite INT8 model to C for TFLM -----
def export_to_c(model_type_load, model_path_tflite, save_dir="./"):
    print('*** export_to_c')

    if len(hidden_units_list) == 1:
        hul = str(hidden_units_list[0])
    elif len(hidden_units_list) == 2:
        hul = str(hidden_units_list[0]) + "_" + str(hidden_units_list[1])
    elif len(hidden_units_list) == 3:
        hul = str(hidden_units_list[0]) + "_" + str(hidden_units_list[1]) + "_" + str(hidden_units_list[2])

    model_name = model_type_load + f"_in{input_dim}_out{output_dim}_nl{num_layers}_hul{hul}"
    header_path = os.path.join(save_dir, f"{model_name}_model_data.h")
    source_path = os.path.join(save_dir, f"{model_name}_model_data.cc")

    #generate_header(header_path, model_name)
    guard = f"{model_name.upper()}_H_".replace(".", "_")
    with open(header_path, "w") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"extern const unsigned char g_{model_name}_model_data[];\n")
        f.write(f"extern const int g_{model_name}_model_data_len;\n\n")
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
    xxd_output_temp = os.path.join(save_dir, f"xxd_output_temp.h")

    # Usa xxd -i per creare il sorgente base
    #with open(xxd_output_temp, "w") as cc:
        #subprocess.run(["xxd", "-i", model_path_tflite], stdout=cc, check=True)
    os.system("xxd -i " + model_path_tflite + " > " + xxd_output_temp)

    # Leggi e modifica il contenuto
    with open(xxd_output_temp, "r") as f:
        content = f.read()
    print(content[0:300])
    print(content[-300:-1])
    os.system('rm ' + xxd_output_temp)

    # Sostituisci i nomi delle variabili
    model_path_tflite_lowercase = model_path_tflite.replace('/', '_').replace('.', '_').replace('-', '_')
    model_name_lowercase = model_name.replace('/', '_').replace('.', '_').replace('-', '_')
    #var_name = "g_model_data" # default
    var_name = model_name_lowercase # default
    #print(model_path_tflite_lowercase)
    first_line_old = f"unsigned char {model_path_tflite_lowercase}[] = {{"
    # Aggiungi include dell'header
    first_line_new = f"#include \"{mcu_include_folder}/{model_name}_model_data.h\"\nalignas(8) const unsigned char g_{var_name}_model_data[] = {{"
    var_name = 'model_tflite' # default
    last_line_old = f"unsigned int {model_path_tflite_lowercase}_len"
    last_line_new = f"unsigned int {var_name}_len"
    content = content.replace(first_line_old, first_line_new)
    content = content.replace(last_line_old, last_line_new)
    print(content[0:300])
    print(content[-300:-1])

    print("Scrittura su:", source_path)
    try:
        os.remove(source_path)
        print("File eliminato con successo")
    except PermissionError:
        print("Errore permessi: file bloccato o senza permessi")
    except FileNotFoundError:
        print("File già inesistente")
    except Exception as e:
        print(f"Errore sconosciuto: {e}")
    with open(source_path, "w") as f:
        #f.write(f'#include "{header_path.name}"\n\n')
        print("Scrivo contenuto modificato")
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    # TODO: cambiare riga di include nel main

    return model_name

# ----- Stima delle MAC operations e della dimensione del modello -----
def get_model_info(model):
    print('*** get_model_info')
    model_size_bytes_float32 = 0
    model_size_bytes_int8 = 0
    for v in model.trainable_weights:
        #try:
        #    dtype = tf.as_dtype(v.dtype)
        #    size_in_bytes = dtype.size
        #except:
        # fallback a float32 (4 bytes)
        #size_in_bytes = 32 / 8 
        
        #print(v.shape)
        #print(np.prod(v.shape))
        
        model_size_bytes_float32 += np.prod(v.shape) * 32 / 8 # bias incluso perchè questo for loop itera su pesi e bias
        model_size_bytes_int8 += np.prod(v.shape)

    #print(model_size_bytes_float32)
    model_size_float32_kb = model_size_bytes_float32 / (1024)
    model_size_float32_mb = model_size_bytes_float32 / (1024 ** 2)
    model_size_int8_kb = model_size_bytes_int8 / (1024)
    model_size_int8_mb = model_size_bytes_int8 / (1024 ** 2)

    # Stima delle MACs (input_dim * output_dim per ogni Dense)
    macs = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            macs += np.prod(layer.kernel.shape)

    return round(model_size_float32_kb, 1), round(model_size_float32_mb, 1), round(model_size_int8_kb, 1), round(model_size_int8_mb, 1), int(macs)

def get_model_info_precise(model):
    #print('*** Profiling RAM/ROM (modello solo Dense, int8 target)')

    # ROM: peso del modello in float32 e in int8 (1 byte per peso/bias)
    model_size_bytes_float32 = 0
    model_size_bytes_int8 = 0
    for v in model.trainable_weights:
        model_size_bytes_float32 += np.prod(v.shape) * 4
        model_size_bytes_int8 += np.prod(v.shape)

    model_size_float32_kb = model_size_bytes_float32 / 1024
    model_size_int8_kb = model_size_bytes_int8 / 1024
    model_size_int8_mb = model_size_bytes_int8 / 1024 / 1024

    # MACs totali (product kernel shape)
    macs = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            macs += np.prod(layer.kernel.shape)

    # RAM: tensor arena (int8)
    input_elements = model.input_shape[1]
    max_intermediate = 0
    output_elements = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            output_elements = layer.units
            if output_elements > max_intermediate:
                max_intermediate = output_elements

    input_tensor_bytes = input_elements  # int8: 1 byte
    output_tensor_bytes = output_elements
    max_intermediate_tensor_bytes = max_intermediate
    overhead_struct_bytes = 4096  # stima conservativa (struct + allocazioni)
    stack_bytes = 2048            # stack base

    ram_total_bytes = input_tensor_bytes + output_tensor_bytes + max_intermediate_tensor_bytes + overhead_struct_bytes + stack_bytes
    ram_total_kb = ram_total_bytes / 1024

    return round(model_size_int8_kb, 2), round(model_size_int8_mb, 2), round(ram_total_kb, 2), macs

# ----- Recupera RAM e ROM da file di log generato da PlatformIO -----
def parse_compilation_logfile():
    """
    Estrae i byte usati per RAM e Flash da un file di log di PlatformIO.
    Restituisce un dizionario: {'RAM': <int>, 'Flash': <int>}
    """
    usage = {}
    pattern = re.compile(r'(RAM|Flash):.*?used\s+(\d+)\s+bytes', re.IGNORECASE)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                mem_type = match.group(1).upper()
                bytes_used = int(match.group(2))
                usage[mem_type] = bytes_used

    return usage

# ----- Salva risultati del profiling su file -----
def save_results(summary, macs, model_size_float32_kb, model_size_float32_mb, model_size_int8_kb, model_size_int8_mb, output_csv):
    print('*** save_results')
    fieldnames = [
        'input_dim', 'num_layers', 'hidden_units_list', 'output_dim',
        'device_type', 'model_size_float32_kb', 'model_size_float32_mb', 'model_size_int8_kb', 'model_size_int8_mb', 'mac_ops',
        'ram_mean_kb', 'rom_mean_kb', 'lat_mean_ms', 'lat_std_ms', 'ops_s'
    ]
    write_header = not os.path.exists(output_csv)

    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for device_type, metrics in summary.items():
            
            ops_s = int((macs*2)/(metrics['lat_mean_ms']/1000)) # operations per seconds

            row = {
                'input_dim': input_dim,
                'num_layers': num_layers,
                'hidden_units_list': hidden_units_list,
                'output_dim': output_dim,
                'device_type': device_type,
                'model_size_float32_kb': model_size_float32_kb,
                'model_size_float32_mb': model_size_float32_mb,
                'model_size_int8_kb': model_size_int8_kb,
                'model_size_int8_mb': model_size_int8_mb,
                'mac_ops': macs,
                **metrics,
                'ops_s': ops_s
            }
            writer.writerow(row)


# ----- Salva risultati del profiling su file -----
def save_results_v2(active_cell, input_dim, output_dim, num_layers, hidden_units_list,
                    model_size_int8_kb, model_size_int8_mb, ram_total_kb, macs, output_csv):

    #print('*** save_results')
    
    fieldnames = [
        'active_cell', 'input_dim', 'num_layers', 'hidden_units_list', 'output_dim',
        'model_size_int8_kb', 'model_size_int8_mb', 'ram_total_kb', 'mac_ops'
    ]
    
    write_header = not os.path.exists(output_csv)

    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        row = {
            'active_cell': active_cell,
            'input_dim': input_dim,
            'num_layers': num_layers,
            'hidden_units_list': hidden_units_list,
            'output_dim': output_dim,
            'model_size_int8_kb': model_size_int8_kb,
            'model_size_int8_mb': model_size_int8_mb,
            'ram_total_kb': ram_total_kb,
            'mac_ops': macs
        }

        writer.writerow(row)

# ----- Run di una singola configurazione -----
def run_experiment(active_cell, input_dim, output_dim, num_layers, hidden_units_list, x_sample, output_csv):
    #print('*** run_experiment')
    model = build_mlp(input_dim, output_dim, num_layers, hidden_units_list)
    model.compile(optimizer='adam', loss='mse')
    #model.summary()

    model_size_int8_kb, model_size_int8_mb, ram_total_kb, macs = get_model_info_precise(model)

    save_results_v2(active_cell, input_dim, output_dim, num_layers, hidden_units_list,
                    model_size_int8_kb, model_size_int8_mb, ram_total_kb, macs, 
                    output_csv)
    
    # TODO: invece di questa riga, fare quantizazione del modello
    end_folder_Training_Size_dd_max_epochs_load, model_type_load, model_path_tflite = get_model_path_tflite()

    # Attivare al bisogno
    #export_test_data(model_path_tflite, end_folder_Training_Size_dd_max_epochs_load, size='small')
    #export_test_data(size='full')
    
    # TODO: COMMENTATO TEMPORANEAMENTE
    #model_name = export_to_c(model_type_load, model_path_tflite, save_dir=mcu_include_folder)

    #os.system('pio run --environment ' + mcu_type + ' -t upload > ' + os.path.join(mcu_folder, 'compilation.txt'))
    compilation_logfile = os.path.join(mcu_folder, 'compilation.txt')
    os.system('pio run --environment ' + mcu_type + ' --project-dir ' + mcu_folder + ' > ' + compilation_logfile)
    print('ARRIVATO QUI')

    ram, rom = parse_compilation_logfile(compilation_logfile)

    ##tflite_model = convert_to_tflite_int8(model, x_sample)
    #model_size_float32_kb, model_size_float32_mb, model_size_int8_kb, model_size_int8_mb, macs = get_model_info(model)
    ##summary = profile_model_ei(tflite_model, reload=RELOAD, save_dir=profiling_ei_folder)
    #inference_time = profile_model_renode(model_name, reload=RELOAD, save_dir=profiling_ei_folder)
    ##save_results(summary, macs, model_size_float32_kb, model_size_float32_mb, model_size_int8_kb, model_size_int8_mb, output_csv)

# %%

# --- ESEMPIO USO --- #
if __name__ == "__main__":

    debug = 1 # 0 = loop; 1 = modello di Taha

    subcarriers = 64 # costante

    # DEFINISCI TU LO SPAZIO DI RICERCA QUI:
    # Teniamo inalterati:
    # - subcarriers
    # - proporzioni tra i neuroni nei vari layer
    if debug == 0:
        #input_dims = [1, 1024]            # Numero di feature in ingresso
        # 8 celle attive x 64 subcarriers x 2 (real/img) = 1024
        active_cells = [1,2,4,8,12,16]
        output_dims = [1024, 512, 256, 128, 64]     # Numero di neuroni di uscita
        num_layers_list = [0,1,2,3]       # Numero di layer MLP (ESCLUSO L'ULTIMO!!!)
    else:    
        # modello di Taha
        #input_dims = [1024]
        active_cells = [8]
        output_dims = [1024]
        num_layers_list = [3]
        
    mcu_type = 'esp32-s2-saola-1'
    mcu_folder = pio_projects_folder + mcu_type
    mcu_include_folder = os.path.join(mcu_folder, 'include') 
    output_csv = profiling_mcu_folder + 'profiling_grid_results.csv'

    if not os.path.exists(mcu_folder):  # Controlla se la cartella esiste
        os.makedirs(mcu_folder, exist_ok=True)  # Crea la cartella se non esiste
        os.system('pio project init --project-dir ' + mcu_folder + ' --board ' + mcu_type + ' --ide arduino' + \
                  ' --project-option="upload_protocol = esptool" ' \
                  '--project-option="upload_port = /dev/ttyUSB0" ' \
                  '--project-option="monitor_port = /dev/ttyUSB0" ' \
                  '--project-option="monitor_speed = 115200" ' \
                  '--project-option="build_flags = ' \
                  '-Ilib/tflite-micro ' \
                  '-Ilib/tflite-micro/tensorflow ' \
                  '-Ilib/tflite-micro/tensorflow/lite/c ' \
                  '-Ilib/tflite-micro/tensorflow/lite/micro ' \
                  '-Ilib/tflite-micro/tensorflow/lite/schema" ')
        print(f"\nProgetto pio creato: {mcu_folder}")

    for active_cell in active_cells:
        
        input_dim = active_cell * subcarriers * 2 #    8 celle attive x 64 subcarriers x 2 (real/img) = 1024
        
        # Sample fake data per quantizzazione
        x_sample = np.random.rand(10, input_dim).astype(np.float32)
            
        data_csv = profiling_mcu_folder + 'data.csv'
        with open(data_csv, 'w') as f:
            # trasforma ogni elemento in una stringa
            for sample in x_sample[:input_dim]:
                f.write(str(sample) + '\n')
        print(f"\nDati creati: {data_csv}")

        for output_dim in output_dims:
            
            for num_layers in num_layers_list:
                
                hidden_units_list = [input_dim, 4*input_dim, 4*input_dim]     # Numero di neuroni per layer
            
                print(f"\nProfiling: act_cell={active_cell}, in={input_dim}, out={output_dim}, layers={num_layers}, units={hidden_units_list}")
                run_experiment(active_cell,input_dim, output_dim, num_layers, hidden_units_list, x_sample, output_csv)

    #model_py = Sequential([
    #    Input(shape=(X_train.shape[1],), name='input'),

    #    Dense(units=Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully1_'),
    #    ReLU(name='relu1'),
    #    Dropout(0.5, name='dropout1'),

    #    Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully2_'),
    #    ReLU(name='relu2'),
    #    Dropout(0.5, name='dropout2'),

    #    Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully3_'),
    #    ReLU(name='relu3'),
    #    Dropout(0.5, name='dropout3'),

    #    Dense(units=Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully4_'),
    #])

    