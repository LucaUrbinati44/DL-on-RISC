import serial
import time
import csv
from datetime import datetime

# Parametri
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
BOARD = 'esp32'
csv_file = 'data.csv'
delimiter = ' '

# File di log con timestamp
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"logs/log_{BOARD}_{timestamp_str}.txt"

stop_command = "STOP"        # Comando seriale che arresta il logger
next_command = "NEXT"        # Comando seriale che invia dati in seriale

def main():

    # Apertura porta seriale
    with open(csv_file, newline='') as f, \
        serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, \
        open(LOG_FILE, 'w') as log:

        reader = csv.reader(f, delimiter=delimiter)
        print("Avviato logger + feeder su", serial_port)

        # Per ogni riga del file di dati
        for row in reader:

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
                    sample_str = delimiter.join(row) + '\n'

                    # Inviare la riga di dati all'MCU
                    ser.write(sample_str.encode('utf-8'))
                    print(f"Inviato: {sample_str.strip()}")
                    break

if __name__ == "__main__":
    main()