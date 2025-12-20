import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
import imageio
from skimage.color import rgb2gray


# TODO: import pyramid functions from ex3

def lucas_kanade_step(I1, I2, border_cut):
    """
    Computes the Rigid Optical Flow (u, v, theta) between two images.
    Assumes the motion is a combination of translation and rotation.

    Args:
        I1: First image.
        I2: Second image.
        border_cut: Pixels to discard from borders (formerly window_size).

    Returns:
        (u, v, theta): Translation and Rotation parameters.
    """
    # 1. Compute Derivatives (same as before)
    # Remember to divide by 2 for correct scale!
    kernel_x = np.array([[1, 0, -1]]) / 2.0
    kernel_y = kernel_x.T

    Ix = signal.convolve2d(I1, kernel_x, mode='same', boundary='symm')
    Iy = signal.convolve2d(I1, kernel_y, mode='same', boundary='symm')
    It = I2 - I1

    # 2. Create Coordinate Grid (x, y) relative to image center
    # Rotation happens around the center, so (0,0) must be in the middle.
    h, w = I1.shape
    y, x = np.mgrid[0:h, 0:w]

    # Shift origin to center
    x = x - w // 2
    y = y - h // 2

    # 3. Compute the "Angular Derivative" I_theta
    # From the equation: u*Ix + v*Iy + theta*(x*Iy - y*Ix) = -It
    # So the coefficient for theta is (x*Iy - y*Ix)
    # New: I_theta = y * Ix - x * Iy  (Matches image coordinate rotation)
    I_theta = y * Ix - x * Iy

    # 4. Handle Borders (Slice everything)
    b = border_cut
    Ix = Ix[b:-b, b:-b]
    Iy = Iy[b:-b, b:-b]
    I_theta = I_theta[b:-b, b:-b]
    It = It[b:-b, b:-b]

    # 5. Build the 3x3 Matrix equation: A * [u, v, theta] = B

    Ix2 = np.sum(Ix ** 2)
    Iy2 = np.sum(Iy ** 2)
    IxIy = np.sum(Ix * Iy)

    IxIth = np.sum(Ix * I_theta)
    IyIth = np.sum(Iy * I_theta)
    Ith2 = np.sum(I_theta ** 2)

    A = np.array([
        [Ix2, IxIy, IxIth],
        [IxIy, Iy2, IyIth],
        [IxIth, IyIth, Ith2]
    ])

    # B vector
    IxIt = np.sum(Ix * It)
    IyIt = np.sum(Iy * It)
    IthIt = np.sum(I_theta * It)

    B = np.array([[-IxIt], [-IyIt], [-IthIt]])

    # 6. Solve
    try:
        res = np.linalg.solve(A, B).flatten()
        return res[0], res[1], res[2]  # u, v, theta
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0
