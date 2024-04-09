import rospy
import socket
from sensor_msgs.msg import NavSatFix
from fusion_engine_client.parsers import FusionEngineDecoder
from fusion_engine_client.messages import PoseMessage
TCP_IP = ""
TCP_PORT = 15213

msg_decoder = FusionEngineDecoder(
            max_payload_len_bytes=4096, warn_on_unrecognized=False, return_bytes=True)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.connect((TCP_IP, TCP_PORT))

BUFFER_SIZE = 1024

while True:
    # print("receiving ...")
    data = s.recv(BUFFER_SIZE)
    # data = data.decode("ISO-8859-1")
    if not data:
        break
    try:
        data = msg_decoder.on_data(data)[0][1]
        if type(data) is PoseMessage:
            print(f"loc: {data.lla_deg}\nvel: {data.velocity_body_mps}")

    except:
        print(type(data))