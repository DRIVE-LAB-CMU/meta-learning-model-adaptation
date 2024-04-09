import pyvicon_datastream as pv

VICON_TRACKER_IP = "192.168.3.137"
OBJECT_NAME = "My_Object"

vicon_client = pv.PyViconDatastream()
ret = vicon_client.connect(VICON_TRACKER_IP)

if ret != pv.Result.Success:
    print(f"Connection to {VICON_TRACKER_IP} failed")
else:
    print(f"Connection to {VICON_TRACKER_IP} successful")