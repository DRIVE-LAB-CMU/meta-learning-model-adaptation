import rospy
from sensor_msgs.msg import Joy

joy_lock = False

def joystick_callback(data):
    global joy_lock
    if joy_lock == False:
        joy_lock = True
        throttle = data.axes[1]
        steering = data.axes[2]
        toggle = data.buttons[6]
        print(f"throttle: {throttle}, steering: {steering}, toggle: {toggle}")

        pub = rospy.Publisher('/vesc/joy',Joy)
        cmd = Joy()
        now = rospy.Time.now()
        cmd.header.stamp = now
        cmd.buttons = [0,0,0,0,toggle,0,0,0,0,0,0]
        cmd.axes = [0.,throttle,steering,0.,0.,0.]
        pub.publish(cmd)
        joy_lock = False
        
def joystick_listener():
    global joy_lock 
    if joy_lock == False:
        joy_lock = True
        rospy.init_node("offroad", anonymous=True)
        rospy.Subscriber("/vesc/joy", Joy, joystick_callback)
        rospy.spin()
        joy_lock = False

if __name__ == '__main__':
    joystick_listener()
