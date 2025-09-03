import serial
#import time
#import csv
from datetime import datetime
import re
import numpy as np

dummy = 'dummy_'

# Parametri
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
BOARD = 'esp32'

base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'
output_folder = base_folder + 'Output_Python/'
mcu_profiling_folder = output_folder + 'Profiling_Search_MCU/'
mcu_profiling_folder_input = mcu_profiling_folder + 'test_data/'
delimiter = ' '

# File di log con timestamp
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"/mnt/c/Users/Work/Desktop/deepMIMO/RIS/logs/log_{BOARD}_{timestamp_str}.txt"

next_command = "NEXT"        # Comando seriale che invia dati in seriale

def compute_stats(data_list):
    if not data_list:
        return None
    arr = np.array(data_list)
    # Deviazione standard (popolazione intera, ddof=0)
    # Se vuoi quella campionaria (N-1 al denominatore): ddof=1
    return np.mean(arr), np.percentile(arr, 50), np.percentile(arr, 95), np.std(arr, ddof=0)

def get_rate_from_codebook(codebook_index_list, YValidation_un_test):
    MaxR_DL_py = np.zeros(len(codebook_index_list), dtype=np.float32)

    # Ciclo di confronto
    for b in range(len(codebook_index_list)):
        MaxR_DL_py[b]  = YValidation_un_test[codebook_index_list[b]-1,  b] # -1 to come back to 0-based indexing 

    Rate_DL_py = MaxR_DL_py.mean()
    return Rate_DL_py


def main(warmup_samples, YValidation_un_test, xtest_npy_filename):
    # Liste per accumulare i tempi
    normalize_input_list = []
    quantize_input_list = []
    interpreter_invoke_list = []
    dequantize_output_list = []
    extract_codebook_index_list = []
    extract_codebook_index_time_list = []
    extract_codebook_index_fast_list = []
    extract_codebook_index_fast_time_list = []
    tot_latency_list = []
    tot_latency_fast_list = []
    Error = 0

    if dummy == 'dummy_':
        data_csv = mcu_profiling_folder_input + dummy + 'data.npy'
        x_sample = np.load(data_csv)
    else:
        x_sample = np.load(xtest_npy_filename)

    #with open(data_csv, newline='') as f, \
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, \
        open(LOG_FILE, 'w') as log:

        #datafile = csv.reader(f, delimiter=delimiter)
        print("Avviato logger + feeder su", SERIAL_PORT)

        #for datarow in datafile:
        for idx, sample in enumerate(x_sample):

            while True:
                
                # Leggere dalla seriale
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                print(f"MCU: {line}")

                # Scrivere su log
                log.write(f"{line}\n")

                # Parsing dei tempi
                match = re.match(r"normalize_input \[us\]: (\d+)", line)
                if match:
                    if idx > warmup_samples:
                        normalize_input_list.append(int(match.group(1)))
                
                match = re.match(r"quantize_input \[us\]: (\d+)", line)
                if match:
                    if idx > warmup_samples:
                        quantize_input_list.append(int(match.group(1)))

                match = re.match(r"interpreter_invoke \[us\]: (\d+)", line)
                if match:
                    interpreter_invoke_list.append(int(match.group(1)))

                match = re.match(r"dequantize_output \[us\]: (\d+)", line)
                if match:
                    if idx > warmup_samples:
                        dequantize_output_list.append(int(match.group(1)))

                match = re.match(r"extract_codebook_index \[\#\] \[us\]: (\d+) (\d+)", line)
                if match:
                    if idx > warmup_samples:
                        extract_codebook_index_list.append(int(match.group(1)))
                        extract_codebook_index_time_list.append(int(match.group(2)))
                
                match = re.match(r"extract_codebook_index_fast \[\#\] \[us\]: (\d+) (\d+)", line)
                if match:
                    if idx > warmup_samples:
                        extract_codebook_index_fast_list.append(int(match.group(1)))
                        extract_codebook_index_fast_time_list.append(int(match.group(2)))

                        # Quando si arriva all'ultima print da rilevare
                        tot_latency      = normalize_input_list[-1] + quantize_input_list[-1] + interpreter_invoke_list[-1] + dequantize_output_list[-1] + extract_codebook_index_time_list[-1]
                        tot_latency_fast = normalize_input_list[-1] + quantize_input_list[-1] + interpreter_invoke_list[-1] + extract_codebook_index_fast_time_list[-1]
                        tot_latency_list.append(tot_latency)
                        tot_latency_fast_list.append(tot_latency_fast)
                
                # Pattern che causa loop infinito
                LOOP_PATTERN = re.compile(r"\bRebooting\b")
                if LOOP_PATTERN.search(line):
                    Error = 1
                    break

                # Attendere (while) segnale di NEXT dall'MCU (cioè quando richiede i dati)
                if line == next_command:
                    #sample_str = delimiter.join(datarow) + '\n'
                    sample_str = delimiter.join(str(x) for x in sample) + '\n'

                    # Inviare la riga di dati all'MCU
                    ser.write(sample_str.encode('utf-8'))
                    print(f"Inviato: {sample_str.strip()}")
                    break

            if Error == 1:
                break

    if Error == 0:
        #avg_normalize_input_us = sum(normalize_input_list) / len(normalize_input_list) if normalize_input_list else 0
        #avg_quantize_input_us = sum(quantize_input_list) / len(quantize_input_list) if quantize_input_list else 0
        #avg_interpreter_invoke_us = sum(interpreter_invoke_list) / len(interpreter_invoke_list) if interpreter_invoke_list else 0
        #avg_dequantize_output_us = sum(dequantize_output_list) / len(dequantize_output_list) if dequantize_output_list else 0
        #avg_extract_codebook_index_us = sum(extract_codebook_index_time_list) / len(extract_codebook_index_time_list) if extract_codebook_index_time_list else 0
        #avg_extract_codebook_index_fast_us = sum(extract_codebook_index_fast_time_list) / len(extract_codebook_index_fast_time_list) if extract_codebook_index_fast_time_list else 0
        #tot_latency_us = avg_normalize_input_us +  avg_quantize_input_us +  avg_interpreter_invoke_us + avg_dequantize_output_us +  avg_extract_codebook_index_us
        #tot_latency_fast_us = avg_normalize_input_us +  avg_quantize_input_us +  avg_interpreter_invoke_us + avg_extract_codebook_index_fast_us
        mean_norm, perc50_norm, perc95_norm, std_norm                                                 = compute_stats(normalize_input_list)
        mean_quant, perc50_quant, perc95_quant, std_quant                                             = compute_stats(quantize_input_list)
        mean_invoke, perc50_invoke, perc95_invoke, std_invoke                                         = compute_stats(interpreter_invoke_list)
        mean_dequant, perc50_dequant, perc95_dequant, std_dequant                                     = compute_stats(dequantize_output_list)
        mean_extract, perc50_extract, perc95_extract, std_extract                                     = compute_stats(extract_codebook_index_time_list)
        mean_extract_fast, perc50_extract_fast, perc95_extract_fast, std_extract_fast                 = compute_stats(extract_codebook_index_fast_time_list)
        mean_tot_latency, perc50_tot_latency, perc95_tot_latency, std_tot_latency                     = compute_stats(tot_latency_list)
        mean_tot_latency_fast, perc50_tot_latency_fast, perc95_tot_latency_fast, std_tot_latency_fast = compute_stats(tot_latency_fast_list)

        Rate_DL_py_load_test_tflite_mcu = get_rate_from_codebook(extract_codebook_index_list, YValidation_un_test)
    else:
        mean_norm = 0
        perc50_norm = 0
        perc95_norm = 0
        std_norm = 0                                            
        mean_quant = 0
        perc50_quant = 0
        perc95_quant = 0
        std_quant = 0                                      
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
        Rate_DL_py_load_test_tflite_mcu = 0

    # Ritorno delle medie e lista extract_codebook_index
    #return avg_normalize_input_us, avg_quantize_input_us, avg_interpreter_invoke_us, avg_dequantize_output_us, avg_extract_codebook_index_us, avg_extract_codebook_index_fast_us, tot_latency_us, tot_latency_fast_us, extract_codebook_index_list, extract_codebook_index_fast_list, Error
    return mean_norm, perc50_norm, perc95_norm, std_norm, \
           mean_quant, perc50_quant, perc95_quant, std_quant, \
           mean_invoke, perc50_invoke, perc95_invoke, std_invoke, \
           mean_dequant, perc50_dequant, perc95_dequant, std_dequant, \
           mean_extract, perc50_extract, perc95_extract, std_extract, \
           mean_extract_fast, perc50_extract_fast, perc95_extract_fast, std_extract_fast, \
           mean_tot_latency, perc50_tot_latency, perc95_tot_latency, std_tot_latency, \
           mean_tot_latency_fast, perc50_tot_latency_fast, perc95_tot_latency_fast, std_tot_latency_fast, \
           extract_codebook_index_list, extract_codebook_index_fast_list, Rate_DL_py_load_test_tflite_mcu, Error
    