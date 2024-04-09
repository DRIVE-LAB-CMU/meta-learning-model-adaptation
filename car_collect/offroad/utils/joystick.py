import socket
import struct
import numpy as np

def key_response_fn(mode='wsad'):
    if mode == 'wsad':
        return key_response_wsad


def key_response_wsad(key, command_state):
    if key.char == 'w':
        if command_state['throttle'] < 1.:
            command_state['throttle'] += 0.05
            # command_state['throttle'] = np.clip(command_state['throttle'], 0, 0.2)
        print(f'{key.char} pressed, forward speed: ', command_state['throttle'])

    elif key.char == 's':
        if command_state['throttle'] > -0.2:
            command_state['throttle']-= 0.05
            # command_state['throttle'] = np.clip(command_state['throttle'], 0, 0.2)
        print(f'{key.char} pressed, forward speed: ', command_state['throttle'])
    elif key.char == 'd':
        if command_state['steering'] > -1.0:
            command_state['steering'] -= 0.05
            command_state['steering'] = np.clip(command_state['steering'], -1., 1.)
            
        print(f'{key.char} pressed, side speed: ', command_state['steering'])
    elif key.char == 'a':
        print(f'{key.char} pressed')
        if command_state['steering'] < 1.0:
            command_state['steering'] += 0.05
            command_state['steering'] = np.clip(command_state['steering'], -1., 1.)
    elif key.char == 'k':
        if command_state['steering'] > -1.0:
            command_state['steering'] -= 0.5
            command_state['steering'] = np.clip(command_state['steering'], -1., 1.)
            
        print(f'{key.char} pressed, side speed: ', command_state['steering'])
    elif key.char == 'j':
        print(f'{key.char} pressed')
        if command_state['steering'] < 1.0:
            command_state['steering'] += 0.5
            command_state['steering'] = np.clip(command_state['steering'], -1., 1.)
            
        print(f'{key.char} pressed, side speed: ', command_state['steering'])
    elif key.char == 'e':
        print(f'{key.char} pressed, side speed: ', command_state['throttle'], command_state['steering'])
        command_state['steering'] = 0.0
        command_state['throttle'] = 0.0

def keyboard_server(args):
    teleop_dict = args[0]
    BUFFER_SIZE = 8
    TCP_IP = ""  # Replace with the IP address of your laptop
    TCP_PORT = 15214  # Replace with the desired port number
    # Used for laptop keyboard control
    KeyboardConnect = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    KeyboardConnect.bind((TCP_IP, TCP_PORT))
    # KeyboardConnect.listen(1)

    # print("Waiting for keyboard connection...")
    # conn, addr = KeyboardConnect.accept()
    # print("Keyboard connected!")

    while True:
        data, addr = KeyboardConnect.recvfrom(BUFFER_SIZE)
        if not data:
            break
        unpacked_data = struct.unpack('ff', data)
        teleop_dict['throttle'] = unpacked_data[0]
        teleop_dict['steering'] = unpacked_data[1]