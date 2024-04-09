import math

def diff_angle(x,y) :
    d = abs(x-y)
    if abs(d) > 3*math.pi/2. :
        d = abs(d-2*math.pi)
    return d
