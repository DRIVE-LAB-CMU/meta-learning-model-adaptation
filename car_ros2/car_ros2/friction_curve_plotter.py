import numpy as np
import matplotlib.pyplot as plt

alpha = [-0.5,-0.43,-0.3,-0.11,0.04,0.26,0.5]
forces = np.array([-1.03,-1.02,-0.8,-0.33,0.15,0.7,1.006]) + 0.5

plt.plot(alpha,forces,label="Fry/m")

alpha = [-0.5,-0.38,-0.29,-0.16,0.06,0.2,0.37,0.5]
forces = np.array([-0.1,0.2,0.1,0.02,-0.03,-0.11,-0.24,-0.24])*-4. + 0.8

plt.plot(alpha,forces,label="Ffy/m")
plt.plot([-0.5,0.5],[0,0],'--',label="X axis")
plt.plot([0,0],[-0.8,1.8],'--',label="Y axis")
plt.xlabel("alpha (in rad)")
plt.ylabel("forces/mass (in $m/s^2$)")
plt.legend()
plt.savefig("friction_curve_bank.png")
plt.show()