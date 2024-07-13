import socket
import struct
import time
import numpy as np
server_ip = "0.0.0.0"
server_port = 12346
pose_x = 0.
pose_y = 0.
pose_yaw = 0.
t_prev = time.time()
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((server_ip, server_port))

print(f"UDP server listening on {server_ip}:{server_port}")

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
t = time.time()
while True :
    data, addr = server_socket.recvfrom(48)  # 3 doubles * 8 bytes each = 24 bytes
    unpacked_data = struct.unpack('dddddd', data)
    # print(f"Received pose from {addr}: {unpacked_data}", time.time()-t)
    t = time.time()
    time.sleep(0.005)
    