import os
import imageio
import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates, convolve1d
from skimage.color import rgb2gray

"------------------------------------------------------------------------------"
"-------------------------------- Utils ---------------------------------------"
"------------------------------------------------------------------------------"


def load_video_frames(filename, inputs_folder='Exercise Inputs',
                      max_frames=None, downscale_factor=1):
    """
    Loads video as RGB. Returns (N, H, W, 3).
    """
    video_path = os.path.join(inputs_folder, filename)
    print(f"Loading video from: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Could not find video at {video_path}")

    reader = imageio.get_reader(video_path)
    frames = []

    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break

        if downscale_factor > 1:
            frame = frame[::downscale_factor, ::downscale_factor, :]

        # Normalize to [0,1]
        frame = frame.astype(np.float64) / 255.0

        # Ensure RGB
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=2)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        frames.append(frame)

    reader.close()
    frames_np = np.array(frames)
    print(f"Loaded {len(frames_np)} RGB frames. Shape: {frames_np.shape}")
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


"------------------------------------------------------------------------------"
"-------------------------- Pyramid Implementation ----------------------------"
"------------------------------------------------------------------------------"
# Standard kernel for Burt & Adelson pyramids
REDUCE_KERNEL = np.array([1, 4, 6, 4, 1]) / 16.0


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
    """Construct a Gaussian pyramid from the input image."""
    img = img.astype(np.float64)
    pyramid = [img]
    current_img = img

    for _ in range(1, num_of_levels):
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
    Assumes inputs are Grayscale.
    """
    kernel_x = np.array([[1, 0, -1]]) / 2.0
    kernel_y = kernel_x.T

    Ix = signal.convolve2d(I1, kernel_x, mode='same', boundary='symm')
    Iy = signal.convolve2d(I1, kernel_y, mode='same', boundary='symm')
    It = I2 - I1

    h, w = I1.shape
    y, x = np.mgrid[0:h, 0:w]
    x = x - w // 2
    y = y - h // 2

    I_theta = y * Ix - x * Iy

    b = border_cut
    Ix = Ix[b:-b, b:-b]
    Iy = Iy[b:-b, b:-b]
    I_theta = I_theta[b:-b, b:-b]
    It = It[b:-b, b:-b]

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

    IxIt = np.sum(Ix * It)
    IyIt = np.sum(Iy * It)
    IthIt = np.sum(I_theta * It)

    B = np.array([[-IxIt], [-IyIt], [-IthIt]])

    try:
        res = np.linalg.solve(A, B).flatten()
        return res[0], res[1], res[2]
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


def warp_image(im, u, v, theta):
    """
    Warps an image using translation (u, v) and rotation (theta).
    UPDATED: Supports both Grayscale (2D) and RGB (3D).
    """
    h, w = im.shape[:2]

    # Create coordinate grid
    x_range = np.arange(w)
    y_range = np.arange(h)
    xv, yv = np.meshgrid(x_range, y_range)

    xv_centered = xv - w // 2
    yv_centered = yv - h // 2

    c, s = np.cos(theta), np.sin(theta)
    rot_x = xv_centered * c + yv_centered * s
    rot_y = -xv_centered * s + yv_centered * c

    src_x = rot_x + u + w // 2
    src_y = rot_y + v + h // 2

    coords = np.stack([src_y, src_x])

    # Check if RGB or Grayscale
    if im.ndim == 3:
        # Warp each channel separately
        warped_channels = []
        for ch in range(im.shape[2]):
            channel = im[:, :, ch]
            warped_c = map_coordinates(channel, coords, order=1,
                                       prefilter=False)
            warped_channels.append(warped_c.reshape(h, w))
        return np.stack(warped_channels, axis=2)
    else:
        # Original Grayscale logic
        warped_im = map_coordinates(im, coords, order=1, prefilter=False)
        return warped_im.reshape(h, w)


def optical_flow(im1, im2, step_size, border_cut):
    """
    Computes rigid optical flow.
    Expects Grayscale inputs.
    """
    min_dim = min(im1.shape[0], im1.shape[1])
    num_levels = int(np.log2(min_dim / step_size)) + 1
    num_levels = max(1, num_levels)

    pyr1 = gaussian_pyramid(im1, num_levels)
    pyr2 = gaussian_pyramid(im2, num_levels)

    u, v, theta = 0.0, 0.0, 0.0

    for i in range(len(pyr1) - 1, -1, -1):
        curr_im1 = pyr1[i]
        curr_im2 = pyr2[i]

        if i < len(pyr1) - 1:
            u *= 2
            v *= 2

        warped_im2 = warp_image(curr_im2, u, v, theta)
        du, dv, dtheta = lucas_kanade_step(curr_im1, warped_im2, border_cut)

        u += du
        v += dv
        theta += dtheta

    return u, v, theta


def stabilize_video(frames, step_size, border_cut, enable_rotation=False):
    """
    Stabilizes a video sequence.
    UPDATED: Handles RGB frames by converting to gray locally for calculation.
    """
    print(f"Stabilizing video (Rotation Correction={enable_rotation})...")
    stabilized_frames = []

    current_y_drift = 0.0
    current_theta_drift = 0.0

    stabilized_frames.append(frames[0])

    for i in range(len(frames) - 1):
        # 1. Convert to Gray for Optical Flow Calculation
        im1_gray = rgb2gray(frames[i]) if frames[i].ndim == 3 else frames[i]
        im2_gray = rgb2gray(frames[i + 1]) if frames[i + 1].ndim == 3 else \
            frames[i + 1]

        # 2. Calculate motion
        u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)

        current_y_drift += v
        if enable_rotation:
            current_theta_drift += theta

        # 3. Fix the current frame (Apply warp on RGB original)
        fix_u = 0.0
        fix_v = current_y_drift
        fix_theta = current_theta_drift

        warped_frame = warp_image(frames[i + 1], fix_u, fix_v, fix_theta)
        stabilized_frames.append(warped_frame)

        if i % 10 == 0:
            print(f"Processed frame {i}/{len(frames)}")

    return np.array(stabilized_frames)


def build_matrix(u, v, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    M = np.array([
        [c, -s, u],
        [s, c, v],
        [0, 0, 1]
    ])
    return M


def find_canvas_limits(frames, step_size, border_cut):
    """
    Calculates canvas limits.
    UPDATED: Handles RGB frames by converting to gray locally for calculation.
    """
    current_T = np.eye(3)
    absolute_transforms = [current_T]

    for i in range(len(frames) - 1):
        # Convert to Gray for Calc
        im1_gray = rgb2gray(frames[i]) if frames[i].ndim == 3 else frames[i]
        im2_gray = rgb2gray(frames[i + 1]) if frames[i + 1].ndim == 3 else \
            frames[i + 1]

        u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)

        # Invert for coordinate accumulation
        u *= -1
        v *= -1
        theta *= -1
        M = build_matrix(u, v, theta)

        current_T = current_T @ M
        absolute_transforms.append(current_T)

    h, w = frames[0].shape[:2]
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T

    min_x, max_x = 0, w
    min_y, max_y = 0, h

    for T in absolute_transforms:
        warped_corners = T @ corners
        warped_corners = warped_corners / warped_corners[2, :]
        xs = warped_corners[0, :]
        ys = warped_corners[1, :]
        min_x = min(min_x, xs.min())
        max_x = max(max_x, xs.max())
        min_y = min(min_y, ys.min())
        max_y = max(max_y, ys.max())

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))
    offset_x = -min_x
    offset_y = -min_y

    return absolute_transforms, (canvas_h, canvas_w), (offset_y, offset_x)


def create_mosaic(frames, step_size, border_cut, strip_anchor=0.5):
    """
    Creates a COLOR panorama.
    UPDATED: Calculates strip width based on the motion of the specific ANCHOR point,
    solving gaps when rotation causes edges to move faster than the center.
    """
    print(f"Step 1: Stabilizing Video (Anchor={strip_anchor})...")
    frames = stabilize_video(frames, step_size, border_cut,
                             enable_rotation=True)

    print("Step 2: Calculating Canvas Limits...")
    abs_transforms, canvas_shape, (offset_y, offset_x) = find_canvas_limits(
        frames, step_size, border_cut)

    H_canvas, W_canvas = canvas_shape
    print(f"Canvas Size: {W_canvas}x{H_canvas}")

    panorama = np.zeros((H_canvas, W_canvas, 3))

    T_offset = np.eye(3)
    T_offset[0, 2] = offset_x
    T_offset[1, 2] = offset_y

    h_img, w_img = frames[0].shape[:2]

    # מיקום ה-Anchor בפיקסלים בתוך הפריים הבודד
    target_center = int(w_img * strip_anchor)
    target_center = max(0, min(w_img - 1, target_center))

    print(f"Step 3: Stitching strips from x={target_center}...")

    for i, frame in enumerate(frames):
        T_total = T_offset @ abs_transforms[i]
        T_inv = np.linalg.inv(T_total)

        # --- תיקון: חישוב רוחב פס חכם (Smart Strip Width) ---
        padding = 3  # הגדלתי קצת לביטחון

        if i < len(frames) - 1:
            # 1. ניקח את הנקודה שאנחנו רוצים לגזור (ה-Anchor)
            # נשתמש בגובה האמצעי (h/2) כדי לייצג את העמודה
            p_anchor = np.array([target_center, h_img / 2, 1])

            # 2. נטיל את הנקודה הזו לקנבס דרך הטרנספורמציה של הפריים הנוכחי
            # (איפה ה-Anchor הזה יושב על הפנורמה?)
            p_curr_global = T_total @ p_anchor
            x_curr_global = p_curr_global[0] / p_curr_global[2]

            # 3. נטיל את אותה נקודה (Anchor) דרך הטרנספורמציה של הפריים *הבא*
            # (איפה ה-Anchor של הפריים הבא יושב על הפנורמה?)
            T_next_total = T_offset @ abs_transforms[i + 1]
            p_next_global = T_next_total @ p_anchor
            x_next_global = p_next_global[0] / p_next_global[2]

            # 4. המרחק ביניהם הוא בדיוק הרוחב שאנחנו צריכים לכסות!
            dist = abs(x_next_global - x_curr_global)
            strip_width = int(np.ceil(dist)) + padding

        else:
            # בפריים האחרון אין "פריים הבא", נשתמש ברוחב של הקודם
            # (זה פחות קריטי כי בפריים האחרון אנחנו לוקחים עד קצה התמונה ממילא)
            # אבל ליתר ביטחון נחשב לפי הפריים הקודם
            p_anchor = np.array([target_center, h_img / 2, 1])

            p_curr_global = T_total @ p_anchor
            x_curr_global = p_curr_global[0] / p_curr_global[2]

            T_prev_total = T_offset @ abs_transforms[i - 1]
            p_prev_global = T_prev_total @ p_anchor
            x_prev_global = p_prev_global[0] / p_prev_global[2]

            dist = abs(x_curr_global - x_prev_global)
            strip_width = int(np.ceil(dist)) + padding

        strip_width = max(1, strip_width)  # למנוע קריסה אם אין תזוזה

        # --- מכאן הכל נשאר זהה ללוגיקה הקודמת ---

        if i == 0:
            strip_start_x = 0
            strip_end_x = target_center + (strip_width // 2)

        elif i == len(frames) - 1:
            strip_start_x = target_center - (strip_width // 2)
            strip_end_x = w_img

        else:
            strip_start_x = target_center - (strip_width // 2)
            strip_end_x = strip_start_x + strip_width

        # Project Strip
        strip_corners = np.array([
            [strip_start_x, 0, 1], [strip_end_x, 0, 1],
            [strip_end_x, h_img, 1], [strip_start_x, h_img, 1]
        ]).T

        warped_corners = T_total @ strip_corners
        warped_corners /= warped_corners[2, :]

        min_x = max(0, int(np.floor(warped_corners[0, :].min())))
        max_x = min(W_canvas, int(np.ceil(warped_corners[0, :].max())))
        min_y = max(0, int(np.floor(warped_corners[1, :].min())))
        max_y = min(H_canvas, int(np.ceil(warped_corners[1, :].max())))

        if max_x <= min_x or max_y <= min_y:
            continue

        # Grid
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)
        xv, yv = np.meshgrid(x_range, y_range)
        coords_roi = np.stack([xv, yv, np.ones_like(xv)]).reshape(3, -1)

        # Back-Warp
        src_coords = T_inv @ coords_roi
        src_coords /= src_coords[2, :]
        src_x = src_coords[0, :]
        src_y = src_coords[1, :]

        # Mask
        is_valid_x = (src_x >= 0) & (src_x <= w_img - 1)
        is_valid_y = (src_y >= 0) & (src_y <= h_img - 1)
        valid_mask = is_valid_x & is_valid_y

        if not np.any(valid_mask):
            continue

        roi_h, roi_w = max_y - min_y, max_x - min_x
        src_x_grid = src_x.reshape(roi_h, roi_w)
        src_y_grid = src_y.reshape(roi_h, roi_w)
        mask_grid = valid_mask.reshape(roi_h, roi_w)

        # --- SAMPLE COLORS ---
        for c in range(3):
            new_vals = map_coordinates(frame[:, :, c],
                                       [src_y_grid, src_x_grid],
                                       order=1, prefilter=False)

            pan_slice = panorama[min_y:max_y, min_x:max_x, c]
            pan_slice[mask_grid] = new_vals[mask_grid]
            panorama[min_y:max_y, min_x:max_x, c] = pan_slice

        if i % 20 == 0:
            print(f"Stitched {i}/{len(frames)}")

    return panorama


def dynamic_mosaic(frames, step_size=16, border_cut=15):
    """
    Full pipeline to create and save a panorama from video frames.
    """
    movie_frames = []

    for anchor in np.linspace(0.2, 0.8, 10):
        print(f"Creating panorama for anchor {anchor:.2f}...")

        # קריאה לפונקציה המעודכנת
        pan = create_mosaic(frames, step_size=16, border_cut=15,
                            strip_anchor=anchor)

        # המרה ל-uint8 לצורך שמירה בוידאו
        pan_uint8 = (np.clip(pan, 0, 1) * 255).astype(np.uint8)
        movie_frames.append(pan_uint8)

    # 4. שמירה כקובץ וידאו (MP4)
    # append all frames in reverse order to create a back-and-forth effect
    movie_frames += movie_frames[::-1]
    return movie_frames
