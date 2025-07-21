import tensorflow as tf
import numpy as np
import edgeimpulse as ei
import csv
import os
import time
import io
import json
from contextlib import redirect_stdout
import random
from statistics import mean, stdev
import re

ei.API_KEY = "ei_4a58ede09ec501541b8239002c9ee96833f9fac84c339500a2a04eb3b7c9bcfc"
PROJECT_ID = 712872

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
saved_models_edgeimpulse = network_folder_out + 'saved_models_edgeimpulse/'
figure_folder = output_folder + 'Figures/'
profiling_ei_folder = output_folder + 'Profiling_Search/'
profiling_renode_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_boards/litex-vexriscv-tensorflow-lite-demo/tensorflow/tensorflow/lite/micro/examples/ml_on_risc/c_models'
header_folder = 'tensorflow/lite/micro/examples/ml_on_risc/c_models'
test_data_npy_path = output_folder + 'Test_data/'
test_data_renode_path = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_boards/litex-vexriscv-tensorflow-lite-demo/renode/'

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
    profiling_ei_folder,
    profiling_renode_folder,
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
    return model_type_load, model_path_tflite

# ----- Export test data for inference -----
def export_test_data(size='small'):
    print('*** export_test_data')
    xtest_npy_filename = test_data_npy_path + 'test_set' + end_folder_Training_Size_dd + '.npy'
    xtest = np.load(xtest_npy_filename)
    print(xtest.shape)
    
    if size == 'small':
        xtest_size = 100
    else:
        xtest_size = xtest.shape[0]

    print(xtest[:(xtest_size-1),:].shape)

    xtest_renode_filename = test_data_renode_path + 'test_set_' + size + '.data'
    with open(xtest_renode_filename, 'w') as f:
        # map(str, sample): trasforma ogni elemento del vettore sample in una stringa.
        # ' '.join(...): concatena tutte queste stringhe, separandole con uno spazio.
        for sample in xtest[:xtest_size,:]:
            line = ' '.join(map(str, sample))
            f.write(line + '\n')

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

    # Sostituisci i nomi delle variabili
    model_path_tflite_lowercase = model_path_tflite.replace('/', '_').replace('.', '_').replace('-', '_')
    model_name_lowercase = model_name.replace('/', '_').replace('.', '_').replace('-', '_')
    #var_name = "g_model_data" # default
    var_name = model_name_lowercase # default
    #print(model_path_tflite_lowercase)
    first_line_old = f"unsigned char {model_path_tflite_lowercase}[] = {{"
    # Aggiungi include dell'header
    first_line_new = f"#include \"{header_folder}/{model_name}_model_data.h\"\nalignas(8) const unsigned char g_{var_name}_model_data[] = {{"
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

    return model_name

# ----- Profiling su Edge Impulse -----
def extract_json_blocks(filepath):
    print('*** extract_json_blocks')
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
                    # Stampa il blocco che ha causato l'errore per reload
                    print(json_str)
                buffer = []
    
    first_json = blocks[0]
    second_json = blocks[1]

    return first_json, second_json

def profile_model_ei(tflite_model, reload=0, save_dir="./"):
    print('*** profile_model_ei')
    device_type = DEVICE
    targets = [device_type, 'lowEndMcu', 'highEndMcu', 'highEndMcuPlusAccelerator', 'mpu', 'gpuOrMpuAccelerator']
    results = {t: {'ram': [], 'rom': [], 'lat': []} for t in targets}

    for rep in range(REPETITIONS):
        print(f"   *** Profiling run {rep+1}/{REPETITIONS} on {device_type}")

        profiling_outputpath = os.path.join(save_dir, f"profile_ei_{device_type}_in{input_dim}_out{output_dim}_nl{num_layers}_hul{hidden_units_list}_rep{rep}.json")

        if reload == 0 and not os.path.exists(profiling_outputpath):
            profile = ei.model.profile(model=tflite_model, device=device_type)
            buf = io.StringIO()
            with redirect_stdout(buf):
                profile.summary()
            text = buf.getvalue()

            with open(profiling_outputpath, "w") as f:
                f.write(text)
        else:
            print(f"   Profiling già esistente, ricarico da file {profiling_outputpath}")
            with open(profiling_outputpath, "r") as f:
                text = f.read()

        [first_json, second_json] = extract_json_blocks(profiling_outputpath)

        try:
            results[device_type]['ram'].append(first_json['memory']['eon']['ram'] / 1024)
        except:
            results[device_type]['ram'].append(-1)
        try:
            results[device_type]['rom'].append(first_json['memory']['eon']['rom'] / 1024)
        except:
            results[device_type]['rom'].append(-1)
        try:
            results[device_type]['lat'].append(first_json['timePerInferenceMs'])
        except:
            results[device_type]['lat'].append(-1)

        for target in targets[1:]:
            try:
                ram = second_json[target]['memory']['eon']['ram'] / 1024
            except:
                ram = -1
            try:
                rom = second_json[target]['memory']['eon']['rom'] / 1024
            except:
                rom = -1
            try:
                lat = second_json[target]['timePerInferenceMs']
            except:
                lat = -1

            results[target]['ram'].append(ram)
            results[target]['rom'].append(rom)
            results[target]['lat'].append(lat)

        #except Exception as e:
        #    print("Errore nel profiling o nella lettura:", e)
        #    for t in targets:
        #        results[t]['ram'].append(-1)
        #        results[t]['rom'].append(-1)
        #        results[t]['lat'].append(-1)

    summary = {}
    for t in targets:
        lat_list = [v for v in results[t]['lat'] if v >= 0]
        ram_list = [v for v in results[t]['ram'] if v >= 0]
        rom_list = [v for v in results[t]['rom'] if v >= 0]

        summary[t] = {
            'ram_mean_kb': round(np.mean(ram_list), 2) if ram_list else -1,
            'rom_mean_kb': round(np.mean(rom_list), 2) if rom_list else -1,
            'lat_mean_ms': round(np.mean(lat_list), 2) if lat_list else -1,
            'lat_std_ms': round(np.std(lat_list), 2) if len(lat_list) > 1 else 0.0
        }

    return summary

# ---------

def profile_model_renode(model_name, reload=0, save_dir="./"):
    print('*** profile_model_renode')
    device_type = 'zephyr_riscv'
    targets = [device_type]
    results = {t: {'ram': [], 'rom': [], 'lat': []} for t in targets}

    for rep in range(REPETITIONS):
        print(f"   *** Profiling run {rep+1}/{REPETITIONS} on {device_type}")

        profiling_outputpath = os.path.join(save_dir, "profile_renode", f"{device_type}", model_name, f"rep{rep}.json")

        if reload == 0 and not os.path.exists(profiling_outputpath):
            print("run renode") # TODOs
        else:
            os._exit(0)
            print(f"   Profiling già esistente, ricarico da file {profiling_outputpath}")
            with open(profiling_outputpath, "r") as f:
                text = f.read()

        #[first_json, second_json] = extract_json_blocks(profiling_outputpath)
#
        #try:
        #    results[device_type]['ram'].append(first_json['memory']['eon']['ram'] / 1024)
        #except:
        #    results[device_type]['ram'].append(-1)
        #try:
        #    results[device_type]['rom'].append(first_json['memory']['eon']['rom'] / 1024)
        #except:
        #    results[device_type]['rom'].append(-1)
        #try:
        #    results[device_type]['lat'].append(first_json['timePerInferenceMs'])
        #except:
        #    results[device_type]['lat'].append(-1)
#
        #for target in targets[1:]:
        #    try:
        #        ram = second_json[target]['memory']['eon']['ram'] / 1024
        #    except:
        #        ram = -1
        #    try:
        #        rom = second_json[target]['memory']['eon']['rom'] / 1024
        #    except:
        #        rom = -1
        #    try:
        #        lat = second_json[target]['timePerInferenceMs']
        #    except:
        #        lat = -1
#
        #    results[target]['ram'].append(ram)
        #    results[target]['rom'].append(rom)
        #    results[target]['lat'].append(lat)

    #summary = {}
    #for t in targets:
    #    lat_list = [v for v in results[t]['lat'] if v >= 0]
    #    ram_list = [v for v in results[t]['ram'] if v >= 0]
    #    rom_list = [v for v in results[t]['rom'] if v >= 0]
#
    #    summary[t] = {
    #        'ram_mean_kb': round(np.mean(ram_list), 2) if ram_list else -1,
    #        'rom_mean_kb': round(np.mean(rom_list), 2) if rom_list else -1,
    #        'lat_mean_ms': round(np.mean(lat_list), 2) if lat_list else -1,
    #        'lat_std_ms': round(np.std(lat_list), 2) if len(lat_list) > 1 else 0.0
    #    }

    inference_time = -1
    return inference_time

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

# ----- Run di una singola configurazione -----
def run_experiment(input_dim, output_dim, num_layers, hidden_units_list, x_sample, output_csv):
    print('*** run_experiment')
    #model = build_mlp(input_dim, output_dim, num_layers, hidden_units_list)
    #model.compile(optimizer='adam', loss='mse')
    #model.summary()
    model_type_load, model_path_tflite = get_model_path_tflite()
    model_name = export_to_c(model_type_load, model_path_tflite, save_dir=profiling_renode_folder)
    #model_size_float32_kb, model_size_float32_mb, model_size_int8_kb, model_size_int8_mb, macs = get_model_info(model)
    #tflite_model = convert_to_tflite_int8(model, x_sample)
    #summary = profile_model_ei(tflite_model, reload=RELOAD, save_dir=profiling_ei_folder)
    inference_time = profile_model_renode(model_name, reload=RELOAD, save_dir=profiling_ei_folder)
    #save_results(summary, macs, model_size_float32_kb, model_size_float32_mb, model_size_int8_kb, model_size_int8_mb, output_csv)

# %%

# --- ESEMPIO USO --- #
if __name__ == "__main__":

    debug = 1

    # DEFINISCI TU LO SPAZIO DI RICERCA QUI:
    if debug == 0:
        input_dims = [1, 1024]            # Numero di feature in ingresso
        output_dims = [1024, 256, 64]     # Numero di neuroni di uscita
        num_layers_list = [0,1,2,3]       # Numero di layer MLP (ESCLUSO L'ULTIMO!!!)
    else:    
        # modello di Taha
        input_dims = [1024]
        output_dims = [1024]
        num_layers_list = [3]
        
    DEVICE = 'raspberry-pi-rp2040'
    REPETITIONS = 1 # TODO: portare a 10
    RELOAD = 1  # Cambia in 1 per saltare Edge Impulse e usare i file salvati

    output_csv = profiling_ei_folder + 'profiling_grid_results.csv'

    # Attivare al bisogno
    #export_test_data(size='small')
    #export_test_data(size='full')

    for input_dim in input_dims:
        
        # Sample fake data per quantizzazione
        x_sample = np.random.rand(100, input_dim).astype(np.float32)

        for output_dim in output_dims:
            
            for num_layers in num_layers_list:
                
                hidden_units_list = [input_dim, 4*input_dim, 4*input_dim]     # Numero di neuroni per layer
            
                print(f"Profiling: in={input_dim}, out={output_dim}, layers={num_layers}, units={hidden_units_list}")
                run_experiment(input_dim, output_dim, num_layers, hidden_units_list, x_sample, output_csv)

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