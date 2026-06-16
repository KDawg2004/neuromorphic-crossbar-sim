import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt(
    'singleMemCap.cir.prn',
    skip_header=1,
    invalid_raise=False
)

data = data[~np.isnan(data).any(axis=1)]

time = data[:,1]
x = data[:,3]

plt.figure(figsize=(8,5))
plt.plot(time, x)
plt.xlabel('Time (s)')
plt.ylabel('State x')
plt.title('Memcapacitor State Variable')
plt.grid(True)
plt.show()