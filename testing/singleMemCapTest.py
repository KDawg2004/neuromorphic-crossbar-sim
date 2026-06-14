import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt(
    'singleMemCap.cir.prn',
    skip_header=1,
    invalid_raise=False
)

data = data[~np.isnan(data).any(axis=1)]

voltage = data[:,2]
q = data[:,4]

plt.figure(figsize=(7,5))
plt.plot(q, voltage)
plt.xlabel('Charge (C)')
plt.ylabel('Voltage (V)')
plt.title('Biolek Volt-Coulomb Hysteresis')
plt.grid(True)
plt.show()