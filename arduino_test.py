import serial
import time

ser = serial.Serial('/dev/tty.usbmodem101', 9600)  # serial port com3
time.sleep(1)

angle = 10

message = str(10)
byte_msg = message.encode('utf-8')

ser.write(byte_msg)

response = ser.readline()
print(response.decode().strip()) 

ser.close()