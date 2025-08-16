import serial
import time
import csv
from datetime import datetime

# Parametri
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
BOARD = 'esp32'

base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'
output_folder = base_folder + 'Output_Python/'
mcu_profiling_folder = output_folder + 'Profiling_Search_MCU/'
data_csv = mcu_profiling_folder + 'data.csv'
delimiter = ' '

# File di log con timestamp
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"logs/log_{BOARD}_{timestamp_str}.txt"

next_command = "NEXT"        # Comando seriale che invia dati in seriale

def main():

    # Apertura porta seriale
    with open(data_csv, newline='') as f, \
        serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, \
        open(LOG_FILE, 'w') as log:

        datafile = csv.reader(f, delimiter=delimiter)
        print("Avviato logger + feeder su", SERIAL_PORT)

        # Per ogni riga del file di dati
        for datarow in datafile:

            while True:

                # Leggere dalla seriale
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                print(f"MCU: {line}")

                # Scrivere su log
                log.write(f"{line}\n")

                # Attendere (while) segnale di NEXT dall'MCU (cioè quando richiede i dati)
                if line == next_command:
                    sample_str = delimiter.join(datarow) + '\n'

                    # Inviare la riga di dati all'MCU
                    ser.write(sample_str.encode('utf-8'))
                    print(f"Inviato: {sample_str.strip()}")
                    break

if __name__ == "__main__":
    main()