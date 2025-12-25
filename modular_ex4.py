import os
import numpy as np
import imageio
from scipy import signal
from scipy.ndimage import map_coordinates, convolve1d
from skimage.color import rgb2gray

"============================================================================="
"--------------- Utils & Low-Level Image Processing Functions ----------------"
"============================================================================="

REDUCE_KERNEL = np.array([1, 4, 6, 4, 1]) / 16.0


def load_video_frames(filename, inputs_folder='Exercise Inputs',
                      max_frames=None, downscale_factor=1):
    """
    Loads video as RGB float64 [0, 1].
    Always loads color! Conversion to gray happens only when needed.
    """
    video_path = os.path.join(inputs_folder, filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    frames = []

    print(f"Loading video: {filename}...")
    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break

        if downscale_factor > 1:
            frame = frame[::downscale_factor, ::downscale_factor, :]

        frame = frame.astype(np.float64) / 255.0

        # Ensure 3 channels
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=2)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        frames.append(frame)

    reader.close()
    return np.array(frames)


def save_panorama(panorama, output_path="outputs", filename="panorama.jpg"):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    full_path = os.path.join(output_path, filename)
    print(f"Saving to: {full_path}")

    panorama_clipped = np.clip(panorama, 0, 1)
    panorama_uint8 = (panorama_clipped * 255).astype(np.uint8)

    imageio.imwrite(full_path, panorama_uint8)


def _blur_single_channel(img, kernel):
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
    img = img.astype(np.float64)
    pyramid = [img]
    for i in range(num_of_levels - 1):
        pyramid.append(reduce(pyramid[-1]))
    return pyramid


def warp_image(im, u, v, theta):
    """
    Warps an image based on translation (u, v) and rotation (theta).
    Supports both Grayscale (H, W) and RGB (H, W, 3).
    """
    h, w = im.shape[:2]

    # Create coordinate grid
    x_range = np.arange(w)
    y_range = np.arange(h)
    xv, yv = np.meshgrid(x_range, y_range)

    # Rotation matrix logic
    cx, cy = w // 2, h // 2
    xv_c = xv - cx
    yv_c = yv - cy

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate backwards
    src_x = xv_c * cos_t + yv_c * sin_t + cx
    src_y = -xv_c * sin_t + yv_c * cos_t + cy

    # Translate (Inverse shift)
    src_x = src_x - u
    src_y = src_y - v

    coords = np.array([src_y.flatten(), src_x.flatten()])

    # Handle RGB vs Gray
    if im.ndim == 3:
        warped_channels = []
        for c in range(3):
            # Using order=1 for speed in stabilization step
            channel = im[:, :, c]
            warped_c = map_coordinates(channel, coords, order=1,
                                       mode='constant', cval=0)
            warped_channels.append(warped_c.reshape(h, w))
        return np.stack(warped_channels, axis=2)
    else:
        warped = map_coordinates(im, coords, order=1, mode='constant', cval=0)
        return warped.reshape(h, w)


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


"============================================================================="
"---------------------- Pipeline: High-Level Functions -----------------------"
"============================================================================="


# --- שלב 2: ייצוב ---
def stabilize_video(frames_rgb, step_size, border_cut, enable_rotation=True):
    """
    Stabilizes RGB video.
    Calculates motion on Grayscale version, applies warp on RGB.
    """
    print(
        f"Step 1: Stabilizing Video (Rotation={'ON' if enable_rotation else 'OFF'})...")
    stabilized_frames = [frames_rgb[0]]

    current_y = 0.0
    current_theta = 0.0

    for i in range(len(frames_rgb) - 1):
        # Convert to gray for calculation
        im1_gray = rgb2gray(frames_rgb[i])
        im2_gray = rgb2gray(frames_rgb[i + 1])

        # Calculate Motion
        u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)

        current_y += v
        if enable_rotation:
            current_theta += theta

        # Correction: Cancel Vertical & Rotation drift
        fix_u = 0
        fix_v = current_y
        fix_theta = current_theta

        # Warp RGB
        warped_rgb = warp_image(frames_rgb[i + 1], fix_u, fix_v, fix_theta)
        stabilized_frames.append(warped_rgb)

    return np.array(stabilized_frames)


# --- שלב 3: חישוב מסלול ---
def compute_camera_path(frames_rgb, step_size, border_cut,
                        convergence_point=None):
    """
    Computes global transforms.
    Supports Convergence Point (x,y) to align a specific object instead of background.
    """
    print("Step 2: Computing Camera Path...")
    transforms = [np.eye(3)]
    current_T = np.eye(3)

    if convergence_point is not None:
        cx, cy = convergence_point
        print(f"  > Tracking convergence point: ({cx}, {cy})")

    for i in range(len(frames_rgb) - 1):
        # Calc motion on gray
        im1_gray = rgb2gray(frames_rgb[i])
        im2_gray = rgb2gray(frames_rgb[i + 1])

        u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)

        # Build transform (im2 -> im1)
        M = build_matrix(u, v, theta)

        # Accumulate (im_i -> im_0)
        current_T = current_T @ M

        # --- Convergence Point Logic ---
        if convergence_point is not None:
            # Project point using current T (to see where it lands on canvas)
            p_hom = current_T @ np.array([cx, cy, 1])
            curr_px, curr_py = p_hom[0] / p_hom[2], p_hom[1] / p_hom[2]

            # Calculate shift needed to keep it at (cx, cy)
            dx = cx - curr_px
            dy = cy - curr_py

            # Update T with this shift
            S = np.eye(3)
            S[0, 2] = dx
            S[1, 2] = dy
            current_T = S @ current_T
        # -------------------------------

        transforms.append(current_T)

    return transforms


# --- שלב 4: חישוב גבולות ---
def compute_canvas_geometry(transforms, frame_h, frame_w):
    """
    Projects frame corners to find bounding box.
    """
    print("Step 3: Calculating Canvas Geometry...")
    corners = np.array([
        [0, 0, 1],
        [frame_w, 0, 1],
        [frame_w, frame_h, 1],
        [0, frame_h, 1]
    ]).T

    all_x = []
    all_y = []

    for T in transforms:
        proj = T @ corners
        proj = proj / proj[2, :]
        all_x.extend(proj[0, :])
        all_y.extend(proj[1, :])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    dx = -min_x
    dy = -min_y

    print(f"  > Canvas: {canvas_w}x{canvas_h}, Offset: ({dx:.1f}, {dy:.1f})")
    return canvas_h, canvas_w, dx, dy


# --- שלב 5: רינדור מתקדם (הפונקציה המרכזית) ---
def render_strip_panorama(frames_rgb, transforms, canvas_geometry,
                          strip_anchor=0.5,
                          strip_padding=2,
                          grayscale_out=False,
                          interp_order=1,
                          prefilter=False):
    """
    Stitches the final panorama with full control over aesthetics.

    Args:
        frames_rgb: Stabilized RGB frames.
        transforms: List of global matrices.
        canvas_geometry: (h, w, dx, dy).
        strip_anchor: 0.0 (Left), 0.5 (Center), 1.0 (Right).
        strip_padding: Extra pixels width for each strip (avoids black gaps).
        grayscale_out: If True, returns a 2D grayscale image.
        interp_order: Interpolation order (1=Linear, 3=Cubic).
        prefilter: For order>1, whether to prefilter (Sharper but slower).
    """
    canvas_h, canvas_w, dx, dy = canvas_geometry
    H_img, W_img = frames_rgb[0].shape[:2]

    # 1. Initialize Canvas
    if grayscale_out:
        panorama = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    else:
        panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)

    # Global Offset Matrix
    T_offset = np.eye(3)
    T_offset[0, 2] = dx
    T_offset[1, 2] = dy

    print(
        f"Step 4: Rendering (Anchor={strip_anchor}, Padding={strip_padding}, Gray={grayscale_out})...")

    # Center of strip relative to frame width
    curr_strip_center = W_img * strip_anchor

    for i in range(len(frames_rgb)):
        # --- A. Determine Strip Width ---
        if i < len(frames_rgb) - 1:
            # Distance to next frame center
            curr_x = transforms[i][0, 2]
            next_x = transforms[i + 1][0, 2]
            dist = abs(next_x - curr_x)

            # Width + Padding
            half_width = (dist / 2.0) + strip_padding

            # Handle first frame edge cases
            if i == 0 and strip_anchor < 0.5:
                s_start = 0
            else:
                s_start = int(curr_strip_center - half_width)

            s_end = int(curr_strip_center + half_width)
        else:
            # Last frame
            s_start = int(curr_strip_center - half_width)
            s_end = W_img

        # Clip to image boundaries
        s_start = max(0, s_start)
        s_end = min(W_img, s_end)

        if s_start >= s_end: continue

        # --- B. Geometric Mapping ---
        T_total = T_offset @ transforms[i]
        T_inv = np.linalg.inv(T_total)

        # Project strip to canvas to find bounding box (ROI)
        strip_corners = np.array([
            [s_start, 0, 1], [s_end, 0, 1],
            [s_end, H_img, 1], [s_start, H_img, 1]
        ]).T

        proj = T_total @ strip_corners
        proj /= proj[2, :]

        min_x_c = max(0, int(np.floor(min(proj[0]))))
        max_x_c = min(canvas_w, int(np.ceil(max(proj[0]))))
        min_y_c = max(0, int(np.floor(min(proj[1]))))
        max_y_c = min(canvas_h, int(np.ceil(max(proj[1]))))

        if max_x_c <= min_x_c or max_y_c <= min_y_c: continue

        # Create Grid
        xx, yy = np.meshgrid(np.arange(min_x_c, max_x_c),
                             np.arange(min_y_c, max_y_c))
        coords_canvas = np.stack([xx, yy, np.ones_like(xx)]).reshape(3, -1)

        # Back-project
        coords_src = T_inv @ coords_canvas
        coords_src /= coords_src[2, :]
        src_x_flat = coords_src[0]
        src_y_flat = coords_src[1]

        # Mask
        mask_strip = (src_x_flat >= s_start) & (src_x_flat < s_end)
        mask_img = (src_x_flat >= 0) & (src_x_flat < W_img) & (
                src_y_flat >= 0) & (src_y_flat < H_img)
        final_mask = mask_strip & mask_img

        if not np.any(final_mask): continue

        # Prepare sample coordinates (y, x)
        sample_y = src_y_flat[final_mask]
        sample_x = src_x_flat[final_mask]
        sample_coords = np.array([sample_y, sample_x])

        # --- C. Sampling & Pasting ---
        # Optimization: Prepare source image once
        src_frame = frames_rgb[i]
        if grayscale_out:
            if src_frame.ndim == 3:
                src_frame = rgb2gray(src_frame)
            # Sample single channel
            vals = map_coordinates(src_frame, sample_coords,
                                   order=interp_order, prefilter=prefilter,
                                   mode='constant', cval=0)

            # Paste
            # Trick to assign to masked 2D array slice
            roi = panorama[min_y_c:max_y_c, min_x_c:max_x_c]
            roi[final_mask.reshape(roi.shape)] = vals

        else:  # Color output
            for c in range(3):
                vals = map_coordinates(src_frame[:, :, c], sample_coords,
                                       order=interp_order, prefilter=prefilter,
                                       mode='constant', cval=0)

                roi = panorama[min_y_c:max_y_c, min_x_c:max_x_c, c]
                roi[final_mask.reshape(roi.shape)] = vals

    return panorama


"============================================================================="

if __name__ == '__main__':
    # # שלבים 1-4 מתבצעים פעם אחת (החלק הכבד)
    # frames = ex4.load_video(...)
    # stable = ex4.stabilize_video(...)
    # transforms = ex4.compute_camera_path(...)
    # geo = ex4.compute_canvas_geometry(...)
    #
    # # --- ניסוי א': פנורמה צבעונית מהמרכז עם חפיפה סטנדרטית ---
    # pan_color = ex4.render_strip_panorama(
    #     stable, transforms, geo,
    #     strip_anchor=0.5,
    #     strip_padding=2,
    #     grayscale_out=False
    # )
    #
    # # --- ניסוי ב': פנורמה אפורה מצד שמאל עם חפיפה גדולה (לתיקון קרעים) ---
    # pan_gray = ex4.render_strip_panorama(
    #     stable, transforms, geo,
    #     strip_anchor=0.0,
    #     strip_padding=5,  # חפיפה אגרסיבית יותר
    #     grayscale_out=True
    # )
    pass
