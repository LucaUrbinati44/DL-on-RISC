import os
import serial
import struct
import time
import re
import numpy as np
import h5py

delimiter = ' '

def compute_stats(data_list):
    if not data_list:
        return None
    arr = np.array(data_list)
    # Deviazione standard (popolazione intera, ddof=0)
    # Se vuoi quella campionaria (N-1 al denominatore): ddof=1
    return np.mean(arr).item(), np.percentile(arr, 50).item(), np.percentile(arr, 95).item(), np.std(arr, ddof=0).item()

def get_rate_from_codebook(codebook_index_list, YValidation_un_test):
    
    #print(len(codebook_index_list))
    #print(codebook_index_list)
    MaxR_DL_py = np.zeros(len(codebook_index_list), dtype=np.float32)

    # Ciclo di confronto
    for b in range(len(codebook_index_list)):
        MaxR_DL_py[b]  = YValidation_un_test[codebook_index_list[b],  b]

    Rate_DL_py = MaxR_DL_py.mean().item()
    return Rate_DL_py


def main(dummy,
         mcu_serial_port, baud_rate, chunk_size_max,
         data_csv, test_set_size,
         small_samples, warmup_samples_for_statistics, 
         YValidation_un_test, 
         xtest_npy_filename, 
         end_folder_Training_Size_dd_epochs, model_name_suffix,
         network_folder_out_RateDLpy_TFLite_mcu,
         mcu_profiling_logfile):
    # Liste per accumulare i tempi
    normalize_input_list = []
    quantize_input_list = []
    normalize_and_quantize_input_list = []
    interpreter_invoke_list = []
    dequantize_output_list = []
    extract_codebook_index_time_list = []
    Indmax_DL_py_load_test_tflite_mcu = []
    extract_codebook_index_fast_time_list = []
    tot_latency_list = []
    tot_latency_fast_list = []
    Error_model_in_ram = 0

    if dummy == 'dummy_':
        xtest = np.load(data_csv)
    else:
        xtest = np.load(xtest_npy_filename)

    if test_set_size == 'small':
        xtest_size = small_samples
    else:
        xtest_size = xtest.shape[0]
    x = xtest[:xtest_size,:]
    print("xtest.shape:", xtest.shape)
    print("x.shape:", x.shape)

    with serial.Serial(mcu_serial_port, baud_rate, timeout=10) as ser, \
        open(mcu_profiling_logfile, 'w', encoding='utf-8') as log:
        print("Avviato logger + feeder su", mcu_serial_port)

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            print(f"MCU: {line}")
            log.write(f"{line}\n") # Scrivere su log
            if line == "NEXT":
                time.sleep(10) # Tempo da lasciare per far preparare l'MCU a leggere dalla seriale
                break
            
            if line == "STOP":
                Error_model_in_ram = 1
                break # Esci dal while    

        for idx, sample in enumerate(x):

            if Error_model_in_ram == 1:
                break # Esci dal for

            print("PSY: ------------------------")
            print(f"PSY: Sample {idx+1}/{xtest_size}")
            
            data_floats = sample.tolist()  # array di float
            total_features = len(data_floats)

            # INVIO PAYLOAD IN CHUNKS (per via del limite buffer UART MCU)
            features_sent = 0
            while features_sent < total_features:
                # INVIA CHUNK
                chunk_size = min(chunk_size_max, total_features - features_sent)
                data = struct.pack('<{}f'.format(chunk_size), *data_floats[features_sent:features_sent+chunk_size]) # little-endian
                ser.write(data)
                ser.flush()
                features_sent += chunk_size
                print(f"PYS: Inviate {features_sent}/{total_features} features ({features_sent/total_features*100}%)")

            while True: # Leggi output MCU

                # Leggere dalla seriale
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                print(f"MCU: {line}")
                log.write(f"{line}\n") # Scrivere su log

                # Parsing dei tempi
                match = re.match(r"normalize_input \[us\]: (\d+)", line)
                if match:
                    #print("PYS: MATCH 1")
                    if idx > (warmup_samples_for_statistics - 1):
                        normalize_input_list.append(int(match.group(1)))
                
                match = re.match(r"quantize_input \[us\]: (\d+)", line)
                if match:
                    #print("PYS: MATCH 2")
                    if idx > (warmup_samples_for_statistics - 1):
                        quantize_input_list.append(int(match.group(1)))
                
                match = re.match(r"normalize_and_quantize_input \[us\]: (\d+)", line)
                if match:
                    #print("PYS: MATCH 2b")
                    if idx > (warmup_samples_for_statistics - 1):
                        normalize_and_quantize_input_list.append(int(match.group(1)))

                match = re.match(r"interpreter_invoke \[us\]: (\d+)", line)
                if match:
                    #print("PYS: MATCH 3")
                    interpreter_invoke_list.append(int(match.group(1)))

                match = re.match(r"dequantize_output \[us\]: (\d+)", line)
                if match:
                    #print("PYS: MATCH 4")
                    if idx > (warmup_samples_for_statistics - 1):
                        dequantize_output_list.append(int(match.group(1)))

                match = re.match(r"extract_codebook_index \[\#\] \[us\]: (\d+) (\d+)", line)
                if match:
                    #print("PYS: MATCH 5")
                    if idx > (warmup_samples_for_statistics - 1):
                        #extract_codebook_index_list.append(int(match.group(1)))
                        extract_codebook_index_time_list.append(int(match.group(2)))
                
                match = re.match(r"extract_codebook_index_fast \[\#\] \[us\]: (\d+) (\d+)", line)
                if match:
                    #print("PYS: MATCH 6")
                    Indmax_DL_py_load_test_tflite_mcu.append(int(match.group(1))) # IMPORTANTE: non deve subire il warmup altrimenti falsa i risultati del rate!
                    
                    if idx > (warmup_samples_for_statistics - 1):
                        extract_codebook_index_fast_time_list.append(int(match.group(2)))

                        # Quando si arriva all'ultima print da rilevare
                        tot_latency      = normalize_input_list[-1] + quantize_input_list[-1] + interpreter_invoke_list[-1] + dequantize_output_list[-1] + extract_codebook_index_time_list[-1]
                        tot_latency_fast = normalize_and_quantize_input_list[-1] + interpreter_invoke_list[-1] + extract_codebook_index_fast_time_list[-1]
                        tot_latency_list.append(tot_latency)
                        tot_latency_fast_list.append(tot_latency_fast)

                    break # Esci dal while perchè hai raccolto tutti i tempi, passa al sample successivo
                    
                # Pattern che causa loop infinito
                LOOP_PATTERN = re.compile(r"\bRebooting\b")
                if LOOP_PATTERN.search(line):
                    Error_model_in_ram = 1
                    break # Esci dal while

            if Error_model_in_ram == 1:
                break # Esci dal for

    if Error_model_in_ram == 0:
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
        mean_normquant, perc50_normquant, perc95_normquant, std_normquant                             = compute_stats(normalize_and_quantize_input_list)
        mean_invoke, perc50_invoke, perc95_invoke, std_invoke                                         = compute_stats(interpreter_invoke_list)
        mean_dequant, perc50_dequant, perc95_dequant, std_dequant                                     = compute_stats(dequantize_output_list)
        mean_extract, perc50_extract, perc95_extract, std_extract                                     = compute_stats(extract_codebook_index_time_list)
        mean_extract_fast, perc50_extract_fast, perc95_extract_fast, std_extract_fast                 = compute_stats(extract_codebook_index_fast_time_list)
        mean_tot_latency, perc50_tot_latency, perc95_tot_latency, std_tot_latency                     = compute_stats(tot_latency_list)
        mean_tot_latency_fast, perc50_tot_latency_fast, perc95_tot_latency_fast, std_tot_latency_fast = compute_stats(tot_latency_fast_list)

        if dummy == '':
            filename_Indmax_DL_py = os.path.join(network_folder_out_RateDLpy_TFLite_mcu, f"Indmax_DL_py_test{end_folder_Training_Size_dd_epochs}{model_name_suffix}.mat")
            with h5py.File(filename_Indmax_DL_py, 'w') as f:
                f.create_dataset('Indmax_DL_py_load_test_tflite_mcu', data=Indmax_DL_py_load_test_tflite_mcu)
                print(f"\nIndmax_DL_py_load_test_tflite_mcu saved in {filename_Indmax_DL_py}")
            
            Rate_DL_py_load_test_tflite_mcu = get_rate_from_codebook(Indmax_DL_py_load_test_tflite_mcu, YValidation_un_test)

            filename_Rate_DL_py = os.path.join(network_folder_out_RateDLpy_TFLite_mcu, f"Rate_DL_py_test{end_folder_Training_Size_dd_epochs}{model_name_suffix}.mat")
            with h5py.File(filename_Rate_DL_py, 'w') as f:
                f.create_dataset('Rate_DL_py_load_test_tflite_mcu', data=Rate_DL_py_load_test_tflite_mcu)
                print(f"\nRate_DL_py_load_test_tflite_mcu saved in {filename_Rate_DL_py}")
        else:
            Rate_DL_py_load_test_tflite_mcu = -1
                
    else:
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
        Rate_DL_py_load_test_tflite_mcu = 0

    # Ritorno delle medie e lista extract_codebook_index
    #return avg_normalize_input_us, avg_quantize_input_us, avg_interpreter_invoke_us, avg_dequantize_output_us, avg_extract_codebook_index_us, avg_extract_codebook_index_fast_us, tot_latency_us, tot_latency_fast_us, extract_codebook_index_list, Indmax_DL_py_load_test_tflite_mcu, Error
    return mean_norm, perc50_norm, perc95_norm, std_norm, \
           mean_quant, perc50_quant, perc95_quant, std_quant, \
           mean_normquant, perc50_normquant, perc95_normquant, std_normquant, \
           mean_invoke, perc50_invoke, perc95_invoke, std_invoke, \
           mean_dequant, perc50_dequant, perc95_dequant, std_dequant, \
           mean_extract, perc50_extract, perc95_extract, std_extract, \
           mean_extract_fast, perc50_extract_fast, perc95_extract_fast, std_extract_fast, \
           mean_tot_latency, perc50_tot_latency, perc95_tot_latency, std_tot_latency, \
           mean_tot_latency_fast, perc50_tot_latency_fast, perc95_tot_latency_fast, std_tot_latency_fast, \
           Indmax_DL_py_load_test_tflite_mcu, Rate_DL_py_load_test_tflite_mcu, Error_model_in_ram
    