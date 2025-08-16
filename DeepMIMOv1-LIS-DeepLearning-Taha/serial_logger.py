import serial
import time

from datetime import datetime

# Configura la porta seriale
SERIAL_PORT = '/dev/ttyUSB0'       # Cambia se necessario (es. '/dev/ttyUSB0' su Linux)
BAUD_RATE = 115200         # Cambia in base al tuo micro
BOARD = 'esp32'

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = f"logs/log_{BOARD}_{timestamp_str}.txt"

stop_command = "STOP"        # Comando seriale che arresta il logger

def main():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, open(OUTPUT_FILE, 'a') as f:
            print(f"Logging da {SERIAL_PORT} a {OUTPUT_FILE}...")
            while True:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_line = f"[{timestamp}] {line}"
                    print(log_line)
                    f.write(log_line + '\n')
                    f.flush()  # Salva immediatamente

                    # Verifica del comando STOP
                    if line.strip() == stop_command:
                        print("STOP command received. Exiting logger.")
                        break

    except KeyboardInterrupt:
        print("\nTerminato manualmente.")
    except serial.SerialException as e:
        print(f"Errore seriale: {e}")

if __name__ == "__main__":
    main()
