import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('singleMemristor.cir.prn', skiprows=1, comments='E')

voltage = data[:, 2]
current = data[:, 3]

plt.plot(voltage, current)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Memristor I-V Hysteresis')
plt.grid(True)
plt.show()