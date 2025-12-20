import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates, convolve1d
import imageio
from skimage.color import rgb2gray

"------------------------------------------------------------------------------"
"-------------------------- Pyramid Implementation ----------------------------"
"------------------------------------------------------------------------------"
# Standard kernel for Burt & Adelson pyramids
REDUCE_KERNEL = np.array([1, 4, 6, 4, 1]) / 16.0


def _blur_single_channel(img, kernel):
    # TODO: Using 'reflect' is generally safer for pyramids than 'nearest',
    #       but stick to 'nearest' if that's what passed previous tests.
    temp = convolve1d(img, kernel, axis=1, mode='nearest')
    blurred = convolve1d(temp, kernel, axis=0, mode='nearest')
    return blurred


def blur(img, kernel):
    img = img.astype(np.float64, copy=False)
    if img.ndim == 2:
        return _blur_single_channel(img, kernel)
    blurred = np.zeros_like(img)
    for c in range(img.shape[2]):
        blurred[..., c] = _blur_single_channel(img[..., c], kernel)
    return blurred


def reduce(img):
    smoothed = blur(img, REDUCE_KERNEL)
    return smoothed[::2, ::2]


def gaussian_pyramid(img, num_of_levels):
    """Construct a Gaussian pyramid from the input image."""
    img = img.astype(np.float64)
    pyramid = [img]
    current_img = img

    for _ in range(1, num_of_levels):
        # Safety check: don't reduce if image is too small
        if min(current_img.shape[:2]) < 2:
            break
        reduced_img = reduce(current_img)
        pyramid.append(reduced_img)
        current_img = reduced_img

    return pyramid


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


def warp_image(im, u, v, theta):
    """
    Warps an image using translation (u, v) and rotation (theta).
    Used inside the optical_flow pyramid loop.

    Args:
        im: Input image (2D numpy array).
        u: Translation in x.
        v: Translation in y.
        theta: Rotation angle in radians.

    Returns:
        Warped image.
    """
    h, w = im.shape

    # 1. Create a grid of coordinates (x, y): pixel locations in the *destination* image.
    x_range = np.arange(w)
    y_range = np.arange(h)
    xv, yv = np.meshgrid(x_range, y_range)

    # 2. Center the coordinates around the center of the image, not (0,0).
    xv_centered = xv - w // 2
    yv_centered = yv - h // 2

    # 3. Apply Inverse Rotation: source_coord = Rotate(dest_coord) + Translation
    # Note: the rotation matrix logic consistent with our derivative calculation.
    c, s = np.cos(theta), np.sin(theta)
    rot_x = xv_centered * c + yv_centered * s
    rot_y = -xv_centered * s + yv_centered * c

    # 4. Apply Translation and Un-center
    # We add 'u' and 'v' because we are sampling from the shifted location.
    # (Backward mapping: if image moved by +10, we sample at index +10 to bring it back)
    src_x = rot_x + u + w // 2
    src_y = rot_y + v + h // 2

    # 5. Interpolate
    # Stack coordinates for map_coordinates (expects shape (2, N))
    # Row 0 is y (rows), Row 1 is x (cols)
    coords = np.stack([src_y, src_x])

    # map_coordinates handles the interpolation.
    # order=1 (linear) is fast and sufficient. prefilter=False for speed.
    warped_im = map_coordinates(im, coords, order=1, prefilter=False)

    return warped_im.reshape(h, w)


def optical_flow(im1, im2, step_size, border_cut):
    """
    Computes the rigid optical flow (u, v, theta) using Gaussian Pyramids.
    Args:
        im1: First image (2D numpy array).
        im2: Second image (2D numpy array).
        step_size: Minimum size of the smallest pyramid level.
        border_cut: Pixels to discard from borders during LK step.
    Returns:
        (u, v, theta): Estimated translation and rotation parameters.
    """
    # 1. Calculate required Number of Levels based on step_size
    # We want the smallest image to be at least 'step_size' pixels.
    # Formula: 2^levels * step_size <= min_dim
    min_dim = min(im1.shape[0], im1.shape[1])
    num_levels = int(np.log2(min_dim / step_size)) + 1
    num_levels = max(1, num_levels)

    # 2. Build Pyramids
    pyr1 = gaussian_pyramid(im1, num_levels)
    pyr2 = gaussian_pyramid(im2, num_levels)

    # 3. Initialize Flow
    u, v, theta = 0.0, 0.0, 0.0

    # 4. Iterate from top of pyramid to base
    for i in range(len(pyr1) - 1, -1, -1):
        curr_im1 = pyr1[i]
        curr_im2 = pyr2[i]

        # Scale up the translation from previous level
        if i < len(pyr1) - 1:
            u *= 2
            v *= 2
            # Theta does NOT scale

        # 5. Warp & Refine
        warped_im2 = warp_image(curr_im2, u, v, theta)
        du, dv, dtheta = lucas_kanade_step(curr_im1, warped_im2, border_cut)

        u += du
        v += dv
        theta += dtheta

    return u, v, theta


def stabilize_video(frames, step_size, win_size):
    """
    Stabilizes a video sequence by keeping only horizontal motion.

    Args:
        frames: List or Array of grayscale images (N, H, W).
        step_size: Parameter for optical_flow pyramids.
        win_size: Parameter for optical_flow window.

    Returns:
        stabilized_frames: List of warped images.
    """
    stabilized_frames = []

    # 1. Initialize Cumulative corrections
    # We treat the first frame as the anchor (0,0,0)
    current_x = 0.0
    current_y = 0.0
    current_theta = 0.0

    # Add first frame as is (it's the reference)
    stabilized_frames.append(frames[0])

    # 2. Loop over frames
    for i in range(len(frames) - 1):
        im1 = frames[i]
        im2 = frames[i + 1]

        # A. Find motion between im1 and im2
        u, v, theta = optical_flow(im1, im2, step_size, win_size)

        # B. Update "Where am I currently?" (Cumulative Path)
        current_x += u
        current_y += v
        current_theta += theta

        # C. Define Correction
        # We want to cancel out the drift in Y and Theta relative to the START.
        # So we want to bring the CURRENT frame (im2) back to y=0, theta=0.
        # FIX: We are warping im2. Its current absolute position is (current_y, current_theta).
        # To fix it, we need to shift by -current_y and rotate by -current_theta.

        fix_u = 0  # Keep horizontal motion (don't fix x)
        fix_v = -current_y  # Cancel vertical drift
        fix_theta = -current_theta  # Cancel rotation drift

        # D. Warp
        # Note: warp_image usually takes "inverse" mapping or "shift amount".
        # Make sure your warp_image handles these parameters correctly to "undo" the motion.
        warped_frame = warp_image(im2, fix_u, fix_v, fix_theta)

        stabilized_frames.append(warped_frame)

        print(
            f"Frame {i + 1}: Motion=({u:.2f}, {v:.2f}), Correction=({fix_v:.2f}, {fix_theta:.2f})")

    return stabilized_frames
