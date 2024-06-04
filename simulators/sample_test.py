import numpy as np
import math
import tf_transformations

x = np.array([6.2819160446040545,-1.9285206465298277,6.244338308990819])
q = tf_transformations.quaternion_from_euler(x[0], x[1], x[2])
print(q)
R = tf_transformations.quaternion_matrix(q)[:3, :3]
t = np.array([0.0, 0.0, 1.0])
Rt = np.dot(R, t)
print(Rt)
print(math.atan2(Rt[0], Rt[2]))