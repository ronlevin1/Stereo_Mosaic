import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
import imageio
from skimage.color import rgb2gray
#TODO: import pyramid functions from ex3

def lucas_kanade_step(I1, I2, window_size):
    """
    Computes the optical flow (u, v) between two images using the Lucas-Kanade method.

    Args:
        I1: First image (numpy array).
        I2: Second image (numpy array).
        window_size: The number of pixels to discard from the borders.

    Returns:
        (u, v): The translation vector.
    """

    # 1. Compute gradients Ix, Iy
    # Kernel for x-derivative: [1, 0, -1]
    kernel_x = np.array([[1, 0, -1]])
    kernel_y = kernel_x.T  # Transpose for y-derivative

    # mode='same' keeps the output size equal to input size
    # boundary='symm' helps reducing artifacts at borders
    Ix = signal.convolve2d(I1, kernel_x, mode='same', boundary='symm')
    Iy = signal.convolve2d(I1, kernel_y, mode='same', boundary='symm')

    # 2. Compute temporal derivative It
    # It represents how much the intensity changed over time (between frames)
    It = I2 - I1

    # 3. Handle Borders (Windowing)
    # We discard pixels near the border because convolution is not valid there,
    # and to satisfy the assumption that the window moves together.
    w = window_size
    # Slice format: [start:end, start:end]
    Ix = Ix[w:-w, w:-w]
    Iy = Iy[w:-w, w:-w]
    It = It[w:-w, w:-w]

    # 4. Build the Lucas-Kanade Matrix Equation: A * [u, v] = b
    # A = [[sum(Ix^2), sum(Ix*Iy)],
    #      [sum(Ix*Iy), sum(Iy^2)]]

    Sigma_Ix2 = np.sum(Ix ** 2)
    Sigma_Iy2 = np.sum(Iy ** 2)
    Sigma_IxIy = np.sum(Ix * Iy)

    A = np.array([[Sigma_Ix2, Sigma_IxIy],
                  [Sigma_IxIy, Sigma_Iy2]])

    # b = [[-sum(Ix*It)],
    #      [-sum(Iy*It)]]

    Sigma_IxIt = np.sum(Ix * It)
    Sigma_IyIt = np.sum(Iy * It)

    b = np.array([[-Sigma_IxIt],
                  [-Sigma_IyIt]])

    # 5. Solve for (u, v)
    # We use pseudo-inverse or solve to handle singular matrices gracefully
    # If the matrix is singular (e.g., flat image with no edges), we cannot detect motion.
    try:
        # np.linalg.solve is numerically more stable than inv(A) @ b
        # However, checking eigenvalues is the robust way to ensure non-singularity.
        # For this exercise, simple exception handling or pinv is usually sufficient.

        # Check if matrix is invertible (det != 0) is risky with floats.
        # Let's try to solve it.
        u, v = np.linalg.solve(A, b).flatten()
        return u, v

    except np.linalg.LinAlgError:
        # Matrix is singular (e.g., blank image or aperture problem)
        return 0.0, 0.0
