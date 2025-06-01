import numpy as np
import matplotlib.pyplot as plt

window_size = 256
overlap = 128

def hanning_window(size):
    hanning=np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (size - 1)) for n in range(size)])
    return hanning
hanning= hanning_window(window_size)
