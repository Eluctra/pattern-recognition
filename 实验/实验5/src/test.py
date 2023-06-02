import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.ones(3)
    y = x
    y[0] = 2
    print(x)