import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
import imageio
from skimage.color import rgb2gray

def lucas_kanade_step(I1, I2, window_size):
    """
    Dummy implementation to start with.
    """
    print("Inside the function!")
    return 0, 0