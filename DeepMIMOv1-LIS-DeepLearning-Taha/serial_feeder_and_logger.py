import serial
import time
import csv
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
data_csv = mcu_profiling_folder_input + dummy + 'data.npy'
delimiter = ' '

# File di log con timestamp
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"/mnt/c/Users/Work/Desktop/deepMIMO/RIS/logs/log_{BOARD}_{timestamp_str}.txt"

next_command = "NEXT"        # Comando seriale che invia dati in seriale


def main():
    # Liste per accumulare i tempi
    normalize_input_list = []
    quantize_input_list = []
    interpreter_invoke_list = []
    dequantize_output_list = []
    extract_codebook_index_list = []
    extract_codebook_index_time_list = []
    extract_codebook_index_fast_list = []
    extract_codebook_index_fast_time_list = []

    #with open(data_csv, newline='') as f, \
    x_sample = np.load(data_csv)
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, \
        open(LOG_FILE, 'w') as log:

        #datafile = csv.reader(f, delimiter=delimiter)
        print("Avviato logger + feeder su", SERIAL_PORT)

        #for datarow in datafile:
        for sample in x_sample:

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
                    normalize_input_list.append(int(match.group(1)))
                
                match = re.match(r"quantize_input \[us\]: (\d+)", line)
                if match:
                    quantize_input_list.append(int(match.group(1)))

                match = re.match(r"interpreter_invoke \[us\]: (\d+)", line)
                if match:
                    interpreter_invoke_list.append(int(match.group(1)))

                match = re.match(r"dequantize_output \[us\]: (\d+)", line)
                if match:
                    dequantize_output_list.append(int(match.group(1)))

                match = re.match(r"extract_codebook_index \[\#\] \[us\]: (\d+) (\d+)", line)
                if match:
                    extract_codebook_index_list.append(int(match.group(1)))
                    extract_codebook_index_time_list.append(int(match.group(2)))
                
                match = re.match(r"extract_codebook_index_fast \[\#\] \[us\]: (\d+) (\d+)", line)
                if match:
                    extract_codebook_index_fast_list.append(int(match.group(1)))
                    extract_codebook_index_fast_time_list.append(int(match.group(2)))

                # Attendere (while) segnale di NEXT dall'MCU (cioè quando richiede i dati)
                if line == next_command:
                    #sample_str = delimiter.join(datarow) + '\n'
                    sample_str = delimiter.join(str(x) for x in sample) + '\n'

                    # Inviare la riga di dati all'MCU
                    ser.write(sample_str.encode('utf-8'))
                    print(f"Inviato: {sample_str.strip()}")
                    break

    # Calcolo medie
    avg_normalize_input_us = sum(normalize_input_list) / len(normalize_input_list) if normalize_input_list else 0
    avg_quantize_input_us = sum(quantize_input_list) / len(quantize_input_list) if quantize_input_list else 0
    avg_interpreter_invoke_us = sum(interpreter_invoke_list) / len(interpreter_invoke_list) if interpreter_invoke_list else 0
    avg_dequantize_output_us = sum(dequantize_output_list) / len(dequantize_output_list) if dequantize_output_list else 0
    avg_extract_codebook_index_us = sum(extract_codebook_index_time_list) / len(extract_codebook_index_time_list) if extract_codebook_index_time_list else 0
    avg_extract_codebook_index_fast_us = sum(extract_codebook_index_fast_time_list) / len(extract_codebook_index_fast_time_list) if extract_codebook_index_fast_time_list else 0
    tot_latency_us = avg_normalize_input_us +  avg_quantize_input_us +  avg_interpreter_invoke_us +  avg_dequantize_output_us +  avg_extract_codebook_index_us
    #tot_latency_fast_us = avg_normalize_input_us +  avg_quantize_input_us +  avg_interpreter_invoke_us +  avg_extract_codebook_index_fast_us

    # Ritorno delle medie e lista extract_codebook_index
    return avg_normalize_input_us, avg_quantize_input_us, avg_interpreter_invoke_us, avg_dequantize_output_us, avg_extract_codebook_index_us, avg_extract_codebook_index_fast_us, tot_latency_us, extract_codebook_index_list, extract_codebook_index_fast_list