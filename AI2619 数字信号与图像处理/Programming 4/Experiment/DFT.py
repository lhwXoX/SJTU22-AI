import numpy as np
import time
import matplotlib.pyplot as plt
import random

def DFT(x):
    N = len(x)
    X = np.zeros(N,dtype='complex_')
    for k in range(N):
        for n in range(N):
            X[k] = X[k] + x[n]*np.exp(-1j*2*np.pi*n*k/N)

if __name__ == '__main__':
    DFT_times = []
    DFT_size = []
    for i in range(4):
        x = []
        for k in range(2**(4*(i+1))):
            x.append(random.randint(0, 10))
        time_start = time.time()
        DFT(x)
        time_end = time.time()
        time_res = time_end - time_start
        DFT_times.append(time_res)
        DFT_size.append(i+1)
    plt.plot(DFT_size, DFT_times, marker ='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('DFT Computation Time (s)')
    plt.show()