import serial.tools.list_ports
for port in serial.tools.list_ports.comports():
    print(port.serial_number)