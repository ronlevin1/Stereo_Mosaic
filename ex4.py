import os
import imageio
import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates, convolve1d
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


def stabilize_video(frames, step_size, border_cut, enable_rotation=False):
    """
    Stabilizes a video sequence.
    Allows disabling rotation correction to keep the horizon straight if rotation is negligible.

    Args:
        frames: List or Array of grayscale images (N, H, W).
        step_size: Parameter for optical_flow pyramids.
        border_cut: Parameter for optical_flow window.
        enable_rotation: Bool. If False, forces theta correction to be 0.

    Returns:
        stabilized_frames: List of warped images (same length as input).
    """
    print(f"Creating panorama with rotation_correction={enable_rotation}.")
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

        # A. Calculate motion between consecutive frames
        # u: dx, v: dy, theta: d_theta
        u, v, theta = optical_flow(im1, im2, step_size, border_cut)

        # B. Accumulate the vertical drift
        current_y_drift += v

        # C. Accumulate rotational drift ONLY if enabled
        if enable_rotation:
            current_theta_drift += theta
        else:
            # If disabled, we ignore the calculated theta for stabilization purposes
            # effectively assuming the camera remained parallel to the horizon.
            pass

            # D. Fix the current frame (im2)
        fix_u = 0.0  # Do not touch horizontal movement
        fix_v = current_y_drift  # Cancel total vertical drift
        fix_theta = current_theta_drift  # Cancel total rotation (or 0 if disabled)

        # E. Apply Warp
        warped_frame = warp_image(im2, fix_u, fix_v, fix_theta)

        stabilized_frames.append(warped_frame)

        # Progress print
        deg_print = np.rad2deg(fix_theta)
        print(
            f"Frame {i + 1}: Correction (dy={fix_v:.2f}, dth={deg_print:.2f}°)")

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
        # TODO: modify direction by some condition if needed
        u *= -1
        v *= -1
        theta *= -1
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


# TODO: integrate into main pipeline where needed
def apply_convergence_correction(absolute_transforms, focal_point=None):
    """
    Adjusts transforms so that 'focal_point' stays stationary.
    focal_point: (x, y) tuple in Frame 0 coordinates.
    """
    if focal_point is None:
        return absolute_transforms  # Default: Infinity alignment

    fx, fy = focal_point
    corrected_transforms = []

    for T in absolute_transforms:
        # 1. Where does the point land currently?
        # Project (fx, fy, 1) using T
        p_homog = T @ np.array([fx, fy, 1.0])
        px, py = p_homog[0] / p_homog[2], p_homog[1] / p_homog[2]

        # 2. Calculate shift needed to bring it back to (fx, fy)
        shift_x = fx - px
        shift_y = fy - py

        # 3. Create shift matrix
        T_shift = np.eye(3)
        T_shift[0, 2] = shift_x
        T_shift[1, 2] = shift_y

        # 4. Update the transform
        # We apply the shift AFTER the original transform
        T_new = T_shift @ T
        corrected_transforms.append(T_new)

    return corrected_transforms


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


def create_mosaic(frames, step_size, border_cut):
    """
    Creates a panorama using the Strips method (Section 4).
    FINAL VERSION: Handles First and Last noisy_frames specially to fill the whole canvas.
    """
    print("Calculating canvas limits...")
    frames = stabilize_video(frames, step_size, border_cut,
                             enable_rotation=True)
    abs_transforms, canvas_shape, (offset_y, offset_x) = find_canvas_limits(
        frames, step_size, border_cut)

    H_canvas, W_canvas = canvas_shape
    print(f"Canvas Size: {W_canvas}x{H_canvas}")

    panorama = np.zeros((H_canvas, W_canvas))

    T_offset = np.eye(3)
    T_offset[0, 2] = offset_x
    T_offset[1, 2] = offset_y

    h_img, w_img = frames[0].shape

    print("Stitching strips (with edge filling)...")

    for i, frame in enumerate(frames):
        # 1. Transform setup
        T_total = T_offset @ abs_transforms[i]
        T_inv = np.linalg.inv(T_total)

        # --- DYNAMIC STRIP WIDTH CALCULATION ---
        padding = 1  # Extra pixels to avoid seams or gaps
        if i < len(frames) - 1:
            curr_x = abs_transforms[i][0, 2]
            next_x = abs_transforms[i + 1][0, 2]
            dist = abs(next_x - curr_x)
            strip_width = int(np.ceil(dist)) + padding
        else:
            # Last frame fallback
            prev_x = abs_transforms[i - 1][0, 2]
            curr_x = abs_transforms[i][0, 2]
            dist = abs(curr_x - prev_x)
            strip_width = int(np.ceil(dist)) + padding

        strip_width = max(1, strip_width)

        # --- SPECIAL HANDLING FOR EDGES ---
        # Instead of always taking the center, we expand the selection for first/last noisy_frames.
        center_x = w_img // 2

        if i == 0:
            # FIRST FRAME: Take everything from pixel 0 up to the end of the strip
            # This fills the black void on the left
            strip_start_x = 0
            strip_end_x = center_x + (strip_width // 2)
            # print(f">>> Frame 0 (Left Edge): Start={strip_start_x}, End={strip_end_x}")  # DEBUG PRINT

        elif i == len(frames) - 1:
            # LAST FRAME: Take everything from the start of the strip to the end of the image
            # This fills the black void on the right
            strip_start_x = center_x - (strip_width // 2)
            strip_end_x = w_img
            # print(f">>> Frame {i} (Right Edge): Start={strip_start_x}, End={strip_end_x}")  # DEBUG PRINT

        else:
            # MIDDLE FRAMES: Take the standard strip from the center
            strip_start_x = center_x - (strip_width // 2)
            strip_end_x = strip_start_x + strip_width
            # print(f"Frame {i}: Start={strip_start_x}, End={strip_end_x}")

        # Define corners based on the calculated range
        strip_corners = np.array([
            [strip_start_x, 0, 1],
            [strip_end_x, 0, 1],
            [strip_end_x, h_img, 1],
            [strip_start_x, h_img, 1]
        ]).T

        # 2. Project Strip to Canvas
        warped_corners = T_total @ strip_corners
        warped_corners = warped_corners / warped_corners[2, :]

        min_x = int(np.floor(warped_corners[0, :].min()))
        max_x = int(np.ceil(warped_corners[0, :].max()))
        min_y = int(np.floor(warped_corners[1, :].min()))
        max_y = int(np.ceil(warped_corners[1, :].max()))

        min_x, max_x = max(0, min_x), min(W_canvas, max_x)
        min_y, max_y = max(0, min_y), min(H_canvas, max_y)
        # print(f"Frame {i}: Pasting at Canvas X range: {min_x} to {max_x}")

        if max_x <= min_x or max_y <= min_y:
            continue

        # 3. Create Grid
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)
        xv, yv = np.meshgrid(x_range, y_range)
        coords_roi = np.stack([xv, yv, np.ones_like(xv)]).reshape(3, -1)

        # 4. Back-Warp
        src_coords = T_inv @ coords_roi
        src_coords = src_coords / src_coords[2, :]
        src_x = src_coords[0, :]
        src_y = src_coords[1, :]

        # 5. Mask (Geometric only)
        is_valid_x = (src_x >= 0) & (src_x <= w_img - 1)
        is_valid_y = (src_y >= 0) & (src_y <= h_img - 1)
        valid_mask = is_valid_x & is_valid_y

        if not np.any(valid_mask):
            continue

        # 6. Sample & Paste
        roi_h, roi_w = max_y - min_y, max_x - min_x
        src_x_grid = src_x.reshape(roi_h, roi_w)
        src_y_grid = src_y.reshape(roi_h, roi_w)
        mask_grid = valid_mask.reshape(roi_h, roi_w)

        new_vals = map_coordinates(frame, [src_y_grid, src_x_grid], order=3,
                                   prefilter=False)

        pan_slice = panorama[min_y:max_y, min_x:max_x]
        pan_slice[mask_grid] = new_vals[mask_grid]
        panorama[min_y:max_y, min_x:max_x] = pan_slice

        if i % 10 == 0:
            print(f"Stitched {i}/{len(frames)}")

    return panorama


def load_video_frames(filename, inputs_folder='Exercise Inputs',
                      max_frames=None, downscale_factor=1):
    """
    Loads a video file, converts to grayscale, and normalizes to [0, 1].

    Args:
        filename (str): Name of the video file (e.g., 'my_video.mp4').
        inputs_folder (str): Relative folder path.
        max_frames (int): Optional limit to load only the first N frames (for testing).
        downscale_factor (int): Optional shrinking (e.g., 2 will reduce size by half).
                                Highly recommended for large videos to save runtime!

    Returns:
        np.array: A 3D array of shape (N, H, W) containing grayscale float frames.
    """
    # 1. Construct full path
    # Assuming the script runs from the root or src folder
    video_path = os.path.join(inputs_folder, filename)

    print(f"Loading video from: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Could not find video at {video_path}")

    # 2. Open Video Reader
    reader = imageio.get_reader(video_path)
    frames = []

    for i, frame in enumerate(reader):
        # Optional: Stop after N frames
        if max_frames and i >= max_frames:
            break

        # 3. Downscale (Simple subsampling) if requested
        if downscale_factor > 1:
            frame = frame[::downscale_factor, ::downscale_factor, :]

        # 4. Convert to Grayscale and Normalize
        if frame.ndim == 3:
            if frame.shape[2] == 4:  # Handle RGBA
                frame = frame[:, :, :3]
            gray_frame = rgb2gray(frame)
        else:
            gray_frame = frame.astype(np.float64) / 255.0

        frames.append(gray_frame)

    reader.close()

    frames_np = np.array(frames)
    print(
        f"Successfully loaded {len(frames_np)} frames. Shape: {frames_np.shape}")

    return frames_np


def save_panorama(panorama, output_path="outputs", filename="panorama.jpg"):
    """
    Saves the panorama to a specific folder and filename.
    Creates the folder if it doesn't exist.
    """
    # 1. Ensure the directory exists
    if not os.path.exists(output_path):
        print(f"Creating directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    # 2. Construct the full path (OS independent)
    # This handles slashes correctly on Mac (/) vs Windows (\)
    full_path = os.path.join(output_path, filename)

    print(f"Saving panorama to: {full_path}...")

    # 3. Process Image (Clip & Convert to uint8)
    # Critical step: prevents black images or color artifacts
    panorama_clipped = np.clip(panorama, 0, 1)
    panorama_uint8 = (panorama_clipped * 255).astype(np.uint8)

    # 4. Save
    try:
        imageio.imwrite(full_path, panorama_uint8)
        print("Success! Image saved.")
    except Exception as e:
        print(f"Error saving image: {e}")


# todo: main pipeline
def main():
    pass
