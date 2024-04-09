import threading
import socket
import sys
from pynput import keyboard
from offroad.utils.joystick import key_response_fn
import struct
import time
# TCP server configuration
# TCP_IP = '172.20.10.2'
TCP_IP = 'rcar.wifi.local.cmu.edu'
TCP_PORT = 15214


command_state = {'throttle': 0.0, 'steering': 0.0}

key_response = key_response_fn(mode='wsad')

def on_press(key):
    global command_state
    try:
        # print(key.char)
        key_response(key, command_state)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()


car_address = (TCP_IP, TCP_PORT)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Function to send input via TCP
def tcp_sender():
    while True:
        time.sleep(0.03)
        global command_state
        # Send the command state (both throttle and steering) via TCP
        packed_data = struct.pack('ff', float(command_state['throttle']), float(command_state['steering']))
        client_socket.sendto(packed_data, car_address)
        # print(len(packed_data))
        # print(str.encode(str(command_state['throttle']) + ',' + str(command_state['steering'])))
        # print('sent')

# Create a TCP client socket
# try:
#     # Connect to the TCP server
#     client_socket.connect()
# except ConnectionRefusedError:
#     print("Failed to connect to the TCP server.")
#     sys.exit(1)

# Create and start the TCP sending thread
tcp_thread = threading.Thread(target=tcp_sender)
tcp_thread.start()
