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


"------------------------------------------------------------------------------"
"-------------------------- Translation calc logic ----------------------------"
"------------------------------------------------------------------------------"


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


def stabilize_video(frames, step_size, border_cut):
    """
    Stabilizes a video sequence by keeping only horizontal motion.
    Based on instruction 2b: "only horizontal motion".

    Args:
        frames: List or Array of grayscale images (N, H, W).
        step_size: Parameter for optical_flow pyramids.
        border_cut: Parameter for optical_flow window.

    Returns:
        stabilized_frames: List of warped images (same length as input).
    """
    stabilized_frames = []

    # 1. Initialize Cumulative Path (Drift)
    # We assume the first frame is the anchor (0,0,0)
    current_y_drift = 0.0
    current_theta_drift = 0.0

    # The first frame is already "stable" relative to itself
    stabilized_frames.append(frames[0])

    # 2. Iterate through the video
    for i in range(len(frames) - 1):
        im1 = frames[i]
        im2 = frames[i + 1]

        # A. Calculate motion between consecutive original_frames (Chain Step)
        # u: dx, v: dy, theta: d_theta
        u, v, theta = optical_flow(im1, im2, step_size, border_cut)

        # B. Accumulate the vertical and rotational drift (relative to first frame)
        current_y_drift += v
        current_theta_drift += theta

        # C. Fix the current frame (im2)
        fix_u = 0.0  # Do not touch horizontal movement
        fix_v = current_y_drift  # Cancel total vertical drift
        fix_theta = current_theta_drift  # Cancel total rotation

        # D. Apply Warp
        # warp_image expects (u, v, theta) as the "shift to apply".
        warped_frame = warp_image(im2, fix_u, fix_v, fix_theta)

        stabilized_frames.append(warped_frame)

        # TODO - Optional: Print progress
        print(
            f"Frame {i + 1}/{len(frames) - 1}: Correction applied (dy={fix_v:.2f}, dth={np.rad2deg(fix_theta):.2f}°)")

    return stabilized_frames


def build_matrix(u, v, theta):
    """
    Creates a 3x3 Rigid transformation matrix from u, v, theta.
    Maps pixels from image (t) to image (t+1) coordinates if used directly,
    or accumulates motion if chained.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    # Rigid Body Transformation Matrix
    M = np.array([
        [c, -s, u],
        [s, c, v],
        [0, 0, 1]
    ])
    return M


def find_canvas_limits(frames, step_size, border_cut):
    """
    Calculates the cumulative motion to find the size of the panorama canvas.
    Args:
        frames: List or Array of grayscale images (N, H, W).
        step_size: Parameter for optical_flow pyramids.
        border_cut: Parameter for optical_flow window.
    Returns:
        absolute_transforms: List of 3x3 matrices mapping each frame to the first frame.
        canvas_size: (height, width) of the required canvas.
        offset: (y_offset, x_offset) to shift all original_frames into positive coordinates.
    """
    # Identify is initialized to Identity (No motion for first frame)
    current_T = np.eye(3)

    # List to store the absolute transform of EACH frame relative to the first frame
    absolute_transforms = [current_T]

    for i in range(len(frames) - 1):
        im1 = frames[i]
        im2 = frames[i + 1]

        # Get motion
        u, v, theta = optical_flow(im1, im2, step_size, border_cut)

        # Create matrix for this step (im_i -> im_{i+1})
        # Note: We need the inverse mapping logic usually for warping,
        # but for coordinate accumulation, we stick to the forward chain.
        M = build_matrix(u, v, theta)

        # Accumulate: T_global_next = T_global_curr @ M
        # (Assuming M maps from i to i+1)
        current_T = current_T @ M
        absolute_transforms.append(current_T)

    # Find min/max coordinates
    # We need to project the corners of every frame into the global coordinate system
    # to see how far they reach.
    h, w = frames[0].shape
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T  # Shape (3, 4)

    min_x, max_x = 0, w
    min_y, max_y = 0, h

    for T in absolute_transforms:
        # Transform corners: new_corners = T @ corners
        warped_corners = T @ corners

        # Normalize (though for Rigid it's always 1, good practice)
        warped_corners = warped_corners / warped_corners[2, :]

        xs = warped_corners[0, :]
        ys = warped_corners[1, :]

        min_x = min(min_x, xs.min())
        max_x = max(max_x, xs.max())
        min_y = min(min_y, ys.min())
        max_y = max(max_y, ys.max())

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    # The offset needed to shift everything to positive coordinates
    offset_x = -min_x
    offset_y = -min_y

    return absolute_transforms, (canvas_h, canvas_w), (offset_y, offset_x)


def warp_global(image, T_inv, canvas_shape):
    """
    Warps an image into the global canvas coordinates.
    Used for creating the panorama.

    Args:
        image: The source image (H, W).
        T_inv: The inverse transformation matrix (3x3) that maps:
               Canvas Pixel -> Source Image Pixel.
        canvas_shape: Target shape (H_canvas, W_canvas).
    """
    H_canvas, W_canvas = canvas_shape

    # 1. Create a grid of ALL pixels in the canvas
    # We create a meshgrid of (x, y) coordinates covering the whole canvas
    x = np.arange(W_canvas)
    y = np.arange(H_canvas)
    xv, yv = np.meshgrid(x, y)

    # 2. Stack to shape (3, N) for matrix multiplication
    # Vectors are [x, y, 1]
    ones = np.ones_like(xv)
    coords = np.stack([xv, yv, ones]).reshape(3, -1)

    # 3. Apply Inverse Transformation to find source coordinates
    # source_coords = T_inv @ canvas_coords
    src_coords_flat = T_inv @ coords

    # 4. Normalize (divide by z to get 2D coords)
    # In rigid motion z is usually 1, but good practice for homography
    z_vec = src_coords_flat[2, :]
    # Avoid division by zero
    z_vec[z_vec == 0] = 1e-10

    src_x = src_coords_flat[0, :] / z_vec
    src_y = src_coords_flat[1, :] / z_vec

    # 5. Reshape back to canvas grid shape
    src_x = src_x.reshape(H_canvas, W_canvas)
    src_y = src_y.reshape(H_canvas, W_canvas)

    # 6. Sample from the source image
    # map_coordinates expects (row_coords, col_coords) -> (y, x)
    # We use order=1 (linear interpolation) and cval=0 (black background)
    warped_image = map_coordinates(image, [src_y, src_x], order=1, cval=0,
                                   prefilter=False)

    return warped_image


def create_panorama(frames, step_size, border_cut):
    """
    Creates a panoramic mosaic by stitching all frames together (Step 3b).
    Naive implementation: Pastes frames on top of each other.
    """
    print("Calculating canvas limits...")
    # 1. Calculate limits and all global transforms
    abs_transforms, canvas_shape, (offset_y, offset_x) = find_canvas_limits(
        frames, step_size, border_cut)

    H_canvas, W_canvas = canvas_shape
    print(f"Canvas Size: {W_canvas}x{H_canvas}")

    # 2. Initialize Panorama (Black canvas)
    panorama = np.zeros((H_canvas, W_canvas))

    # 3. Create Offset Matrix
    # This shifts (0,0) to (offset_x, offset_y) so everything fits
    T_offset = np.eye(3)
    T_offset[0, 2] = offset_x  # x translation
    T_offset[1, 2] = offset_y  # y translation

    print("Stitching frames...")

    for i, frame in enumerate(frames):
        # A. Calculate the Total Transform (Frame -> Canvas)
        # T_rel maps Frame -> Global(0,0)
        # T_offset maps Global(0,0) -> Canvas(positive coords)
        T_rel = abs_transforms[i]
        T_total = T_offset @ T_rel

        # B. We need the Inverse for warping (Canvas -> Frame)
        T_inv = np.linalg.inv(T_total)

        # C. Warp the current frame onto the canvas
        warped_frame = warp_global(frame, T_inv, (H_canvas, W_canvas))

        # D. Paste onto panorama
        # np.maximum allows us to overlay the image w/o overwriting pixels with black borders.
        panorama = np.maximum(panorama, warped_frame)
        # panorama += warped_frame  # Alternative: simple addition (may cause brightness issues)

        # Progress log
        if i % 5 == 0 or i == len(frames) - 1:
            print(f"Pasted frame {i + 1}/{len(frames)}")

    return panorama
