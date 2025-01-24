import numpy as np
import time
import matplotlib.pyplot as plt
import random
from scipy import fft

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd, X_even + terms[int(N/2):] * X_odd])

def DFT(x):
    N = len(x)
    X = np.zeros(N,dtype='complex_')
    for k in range(N):
        for n in range(N):
            X[k] = X[k] + x[n]*np.exp(-1j*2*np.pi*n*k/N)

if __name__ == '__main__':
    DFT_times_def = []
    DFT_size_def = []

    DFT_times = []
    DFT_size = []

    DFT_times_np = []
    DFT_size_np = []

    DFT_times_sci = []
    DFT_size_sci = []

    for i in range(3):
        x = []
        for k in range(2**(4*(i+1))):
            x.append(random.randint(0, 10))
        time_start = time.time()
        DFT(x)
        time_end = time.time()
        time_res = time_end - time_start
        DFT_times_def.append(time_res)
        DFT_size_def.append(i+1)

    print('finish DFT')

    for i in range(6):
        x = []
        for k in range(2**(4*(i+1))):
            x.append(random.randint(0, 10))
        time_start = time.time()
        #np.fft.fft(x)
        #Res = fft.fft(x)
        Res = FFT(x)
        time_end = time.time()
        time_res = time_end - time_start
        DFT_times.append(time_res)
        DFT_size.append(i+1)
    
    print('finish FFT')

    for i in range(6):
        x = []
        for k in range(2**(4*(i+1))):
            x.append(random.randint(0, 10))
        time_start = time.time()
        np.fft.fft(x)
        #Res = fft.fft(x)
        #Res = FFT(x)
        time_end = time.time()
        time_res = time_end - time_start
        DFT_times_np.append(time_res)
        DFT_size_np.append(i+1)

    print('finish numpy')

    for i in range(6):
        x = []
        for k in range(2**(4*(i+1))):
            x.append(random.randint(0, 10))
        time_start = time.time()
        #np.fft.fft(x)
        Res = fft.fft(x)
        #Res = FFT(x)
        time_end = time.time()
        time_res = time_end - time_start
        DFT_times_sci.append(time_res)
        DFT_size_sci.append(i+1)
    
    plt.plot(DFT_size_def, DFT_times_def, marker ='o')
    plt.plot(DFT_size, DFT_times, marker ='o')
    plt.plot(DFT_size_np, DFT_times_np, marker ='o')
    plt.plot(DFT_size_sci, DFT_times_sci, marker ='o')
    plt.legend(['via definition','without function','via numpy','via scipy'],loc = 'upper left')
    plt.xlabel('Sequence Length')
    plt.ylabel('FFT Computation Time (s)')
    plt.title('FFT')
    plt.show()