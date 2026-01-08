import os

import imageio
import PIL.Image
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d, map_coordinates
from skimage.color import rgb2gray

MIN_PYRAMID_SIZE = 32
REDUCE_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

"------------------------------------------------------------------------------"
"-------------------------------- Utils ---------------------------------------"
"------------------------------------------------------------------------------"


def load_video_frames(filename, inputs_folder="Exercise Inputs",
                      spatial_downscale=1):
    """Load video frames from a file, with optional spatial downscaling.
    Params:
        filename (str): Video filename.
        inputs_folder (str): Folder containing the video.
        spatial_downscale (int): Keep every k-th pixel in both axes.
    Returns:
        np.ndarray: Frames array of shape (N, H, W, 3), float64 in [0, 1].
    """
    video_path = os.path.join(inputs_folder, filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    frames = []
    try:
        for idx, frame in enumerate(reader):
            if spatial_downscale > 1:  # in-frame, drop pixels
                frame = frame[::spatial_downscale, ::spatial_downscale, ...]
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Expected RGB frame (H,W,3), got shape {frame.shape} at index {idx}")
            frame = frame.astype(np.float64) / 255.0
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        raise ValueError("No frames loaded; check input parameters")
    return np.stack(frames, axis=0)


def estimate_motion_dir(motion_data):
    """
    Estimate motion between consecutive frames in a video.
    Returns a list of (u, v, theta) tuples.
    """
    avg_u = np.mean([m[0] for m in motion_data])
    is_right_to_left = avg_u > 0

    if is_right_to_left:
        print("Right-to-Left motion detected. Flipping data internally...")
        return "RTL"
    else:
        print("Left-to-Right motion detected.")
        return "LTR"


def warp_image(im, u, v, theta):
    """Warp an image by translation (u, v) and rotation theta around image center.
    Params:
        im (np.ndarray): (H,W) grayscale or (H,W,3) RGB image.
        u (float): X translation in pixels.
        v (float): Y translation in pixels.
        theta (float): Rotation in radians.
    Returns:
        np.ndarray: Warped image with same shape as input.
    """
    h, w = im.shape[:2]
    xv, yv = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    xc = xv - w / 2.0
    yc = yv - h / 2.0
    c, s = np.cos(theta), np.sin(theta)
    rx = xc * c + yc * s
    ry = -xc * s + yc * c
    src_x = rx + u + w / 2.0
    src_y = ry + v + h / 2.0
    coords = np.stack([src_y, src_x])

    if im.ndim == 3:
        warped_channels = [
            map_coordinates(im[..., ch], coords, order=1,
                            prefilter=False).reshape(h, w)
            for ch in range(3)
        ]
        return np.stack(warped_channels, axis=2)

    return map_coordinates(im, coords, order=1, prefilter=False).reshape(h, w)


def gaussian_pyramid(img, num_levels):
    """Build a Gaussian pyramid.
    Params:
        img (np.ndarray): Input image (H,W) or (H,W,C).
        num_levels (int): Max pyramid levels.
    Returns:
        list[np.ndarray]: Pyramid images from level 0 (original) downsampled.
    """
    pyramid = [img.astype(np.float64)]
    current = pyramid[0]
    for _ in range(1, num_levels):
        if min(current.shape[:2]) < MIN_PYRAMID_SIZE:
            break
        current = reduce(current)
        pyramid.append(current)
    return pyramid


def reduce(img):
    """Blur and downsample image by factor 2.
    Params:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Reduced image.
    """
    smoothed = blur(img, REDUCE_KERNEL)
    return smoothed[::2, ::2]


def blur(img, kernel):
    """Separable blur per channel.
    Params:
        img (np.ndarray): (H,W) or (H,W,C) image.
        kernel (np.ndarray): 1D blur kernel.
    Returns:
        np.ndarray: Blurred image, same shape as input.
    """
    img = img.astype(np.float64, copy=False)
    if img.ndim == 2:
        return _blur_single_channel(img, kernel)
    blurred = np.zeros_like(img)
    for ch in range(img.shape[2]):
        blurred[..., ch] = _blur_single_channel(img[..., ch], kernel)
    return blurred


def blur_video(frames, kernel):
    """Blur each frame in a video.
    Params:
        frames (np.ndarray): (N,H,W) or (N,H,W,3) video.
        kernel (np.ndarray): 1D blur kernel.
    Returns:
        np.ndarray: Blurred video, same shape as input.
    """
    blurred_frames = np.zeros_like(frames)
    for i in range(frames.shape[0]):
        blurred_frames[i] = blur(frames[i], kernel)
    return blurred_frames


def build_matrix(u, v, theta):
    """Build 3x3 rigid transform matrix for (u,v,theta).
    Params:
        u (float): X translation.
        v (float): Y translation.
        theta (float): Rotation (radians).
    Returns:
        np.ndarray: (3,3) homogeneous transform matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, u], [s, c, v], [0.0, 0.0, 1.0]], dtype=np.float64)


def _blur_single_channel(img, kernel):
    """Blur a single-channel image using separable 1D convolutions.
    Params:
        img (np.ndarray): (H,W) image.
        kernel (np.ndarray): 1D kernel.
    Returns:
        np.ndarray: Blurred image.
    """
    tmp = convolve1d(img, kernel, axis=1, mode="nearest")
    return convolve1d(tmp, kernel, axis=0, mode="nearest")


def _lucas_kanade_optimization(I1, I2, border_cut, u=0.0, v=0.0, theta=0.0,
                               max_iter=15, epsilon=1e-2):
    """Lucas-Kanade optimization for translation + small rotation.
    Params:
        I1 (np.ndarray): Reference grayscale image (H,W).
        I2 (np.ndarray): Target grayscale image (H,W).
        border_cut (int): Ignore a border of this many pixels.
        u (float): Initial x translation.
        v (float): Initial y translation.
        theta (float): Initial rotation (radians).
        max_iter (int): Max iterations.
        epsilon (float): Convergence threshold.
    Returns:
        tuple[float, float, float]: (u, v, theta) alignment from I2 to I1.
    """
    kernel_x = np.array([[1.0, 0.0, -1.0]]) / 2.0
    kernel_y = kernel_x.T
    Ix = signal.convolve2d(I1, kernel_x, mode="same", boundary="symm")
    Iy = signal.convolve2d(I1, kernel_y, mode="same", boundary="symm")

    h, w = I1.shape
    y, x = np.mgrid[0:h, 0:w]
    x = x - w / 2.0
    y = y - h / 2.0
    I_theta = y * Ix - x * Iy

    b = border_cut
    slices = (slice(b, -b if b > 0 else None), slice(b, -b if b > 0 else None))
    Ix = Ix[slices]
    Iy = Iy[slices]
    I_theta = I_theta[slices]

    Ix2 = np.sum(Ix * Ix)
    Iy2 = np.sum(Iy * Iy)
    Ith2 = np.sum(I_theta * I_theta)
    IxIy = np.sum(Ix * Iy)
    IxIth = np.sum(Ix * I_theta)
    IyIth = np.sum(Iy * I_theta)

    A = np.array(
        [[Ix2, IxIy, IxIth], [IxIy, Iy2, IyIth], [IxIth, IyIth, Ith2]],
        dtype=np.float64,
    )

    for _ in range(max_iter):
        warped = warp_image(I2, u, v, theta)
        It = warped - I1
        It = It[slices]

        IxIt = np.sum(Ix * It)
        IyIt = np.sum(Iy * It)
        IthIt = np.sum(I_theta * It)
        B = np.array([-IxIt, -IyIt, -IthIt], dtype=np.float64)

        try:
            res = np.linalg.solve(A, B)
            du, dv, dtheta = float(res[0]), float(res[1]), float(res[2])
        except np.linalg.LinAlgError:
            break

        u += du
        v += dv
        theta += dtheta

        if abs(du) < epsilon and abs(dv) < epsilon and abs(dtheta) < epsilon:
            break

    return u, v, theta


def _estimate_strip_width(transforms, idx, target_center, frame_h, frame_w,
                          strip_padding, T_offset):
    """Estimate strip width based on local horizontal motion.
    Params:
        transforms (Sequence[np.ndarray]): Per-frame (3,3) transforms.
        idx (int): Frame index.
        target_center (int): Strip center x (in source frame coords).
        frame_h (int): Frame height.
        frame_w (int): Frame width.
        strip_padding (int): Extra pixels to add.
        T_offset (np.ndarray): (3,3) offset transform to canvas coords.
    Returns:
        int: Strip width in pixels.
    """
    ys = [0.0, frame_h / 2.0, float(frame_h)]
    dists = []

    if len(transforms) == 1:
        return frame_w

    if idx < len(transforms) - 1:
        T_curr = T_offset @ transforms[idx]
        T_next = T_offset @ transforms[idx + 1]
    else:
        T_curr = T_offset @ transforms[idx - 1]
        T_next = T_offset @ transforms[idx]

    for y in ys:
        anchor = np.array([target_center, y, 1.0], dtype=np.float64)
        p_curr = T_curr @ anchor
        p_next = T_next @ anchor
        p_curr /= p_curr[2]
        p_next /= p_next[2]
        dists.append(abs(p_next[0] - p_curr[0]))

    max_dist = max(dists) if dists else frame_w
    strip_width = int(np.ceil(max_dist * 1.8)) + strip_padding
    return max(1, min(strip_width, frame_w))


"------------------------------------------------------------------------------"
"------------------------------------------------------------------------------"
"------------------------------------------------------------------------------"


def compute_motion(frames_rgb, border_cut, use_bottom_part=True):
    """Compute motion between consecutive frames.

    Params:
        frames_rgb (np.ndarray): Video frames (N,H,W,3) float in [0,1].
        border_cut (int): Border cut for LK.
        use_bottom_part (bool): If True uses only bottom 2/3 of frame.

    Returns:
        list[tuple[float,float,float]]: [(u,v,theta)] for each pair i->i+1.
    """
    motion_data = []

    im1_gray_full = rgb2gray(frames_rgb[0])
    for i in range(len(frames_rgb) - 1):
        im2_gray_full = rgb2gray(frames_rgb[i + 1])

        if use_bottom_part:
            h = im1_gray_full.shape[0]
            y0 = h // 3
            im1_gray = im1_gray_full[y0:, :]
            im2_gray = im2_gray_full[y0:, :]
        else:
            im1_gray = im1_gray_full
            im2_gray = im2_gray_full

        u, v, theta = align_pair(im1_gray, im2_gray, border_cut)
        motion_data.append((u, v, theta))

        im1_gray_full = im2_gray_full

    return motion_data


def stabilize_video(frames_rgb, motion_data, enable_rotation=True):
    """Stabilize rotations and Y translations across the video.
    Params:
        frames_rgb (np.ndarray): (N,H,W,3) frames.
        motion_data (list[tuple[float,float,float]]): Per-pair motion.
        enable_rotation (bool): If True apply theta stabilization.
    Returns:
        np.ndarray: Stabilized frames (N,H,W,3).
    """
    stabilized_frames = [frames_rgb[0]]
    current_v = 0.0
    current_theta = 0.0

    for i, (u, v, theta) in enumerate(motion_data):
        current_v += v
        if enable_rotation:
            current_theta += theta

        fix_u = 0
        fix_v = current_v
        fix_theta = current_theta

        warped = warp_image(frames_rgb[i + 1], fix_u, fix_v, fix_theta)
        stabilized_frames.append(warped)

    return np.array(stabilized_frames)


def compute_cumulative_transforms(motion_data, convergence_point=None):
    """Compose per-frame camera transforms from motion tuples.
    Params:
        motion_data (Sequence[tuple[float,float,float]]): (u,v,theta) per step.
        convergence_point (tuple[float,float] | None): Optional anchor point.
    Returns:
        list[np.ndarray]: Per-frame (3,3) transforms.
    """
    transforms = [np.eye(3)]
    current_T = np.eye(3)

    for (u, v, theta) in motion_data:
        M = build_matrix(-u, -v, -theta)
        current_T = current_T @ M
        transforms.append(current_T)

    # Convergence Point Logic (Optional)
    if convergence_point is not None:
        pass
        # transforms = _anchor_convergence(transforms, convergence_point)

    return transforms


def compute_canvas_geometry(transforms, frame_h, frame_w):
    """Compute panorama canvas size and offset so all warped frames fit.
    Params:
        transforms (Sequence[np.ndarray]): Per-frame (3,3) transforms.
        frame_h (int): Frame height.
        frame_w (int): Frame width.
    Returns:
        tuple[int,int,float,float]: (canvas_h, canvas_w, dx, dy).
    """
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [frame_w, 0.0, 1.0],
            [frame_w, frame_h, 1.0],
            [0.0, frame_h, 1.0],
        ]
    ).T

    min_x, min_y = 0.0, 0.0
    max_x, max_y = float(frame_w), float(frame_h)

    for T in transforms:
        warped = T @ corners
        warped /= warped[2, :]
        xs = warped[0, :]
        ys = warped[1, :]
        min_x = min(min_x, xs.min())
        max_x = max(max_x, xs.max())
        min_y = min(min_y, ys.min())
        max_y = max(max_y, ys.max())

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))
    dx = -min_x
    dy = -min_y
    return canvas_h, canvas_w, dx, dy


def render_strip_panorama(frames, transforms, canvas_geometry,
                          strip_anchor=0.5, strip_padding=2, interp_order=1,
                          prefilter=False, include_edge_full_strip=False):
    """Render a strip-based panorama (RGB only).
    Params:
        frames (np.ndarray): (N,H,W,3) frames.
        transforms (Sequence[np.ndarray]): (3,3) transform per frame.
        canvas_geometry (tuple[int,int,float,float]): (canvas_h, canvas_w, dx, dy).
        strip_anchor (float): Strip center position in [0,1].
        strip_padding (int): Extra strip width padding.
        interp_order (int): Interpolation order for map_coordinates.
        prefilter (bool): Prefilter flag for map_coordinates.
        include_edge_full_strip (bool): If True, use full-width strip for first/last frame.
    Returns:
        np.ndarray: Panorama canvas (float64) of shape (H,W,3).
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {frames.shape}")

    frame_h, frame_w = frames.shape[1:3]
    canvas_h, canvas_w, dy, dx = canvas_geometry

    pano_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)

    T_offset = np.eye(3, dtype=np.float64)
    T_offset[0, 2] = dx
    T_offset[1, 2] = dy

    target_center = int(np.clip(strip_anchor, 0.0, 1.0) * (frame_w - 1))

    for idx, frame in enumerate(frames):
        T_total = T_offset @ transforms[idx]
        T_inv = np.linalg.inv(T_total)

        strip_width = _estimate_strip_width(transforms, idx, target_center,
                                            frame_h, frame_w, strip_padding,
                                            T_offset)

        strip_start = max(0, target_center - strip_width // 2)
        strip_end = min(frame_w, strip_start + strip_width)

        if include_edge_full_strip:
            if idx == 0:
                strip_start = 0
            if idx == len(frames) - 1:
                strip_end = frame_w

        strip_corners = np.array(
            [
                [strip_start, 0.0, 1.0],
                [strip_end, 0.0, 1.0],
                [strip_end, frame_h, 1.0],
                [strip_start, frame_h, 1.0],
            ]
        ).T

        warped = T_total @ strip_corners
        warped /= warped[2, :]

        min_x = int(np.clip(np.floor(warped[0, :].min()), 0, canvas_w))
        max_x = int(np.clip(np.ceil(warped[0, :].max()), 0, canvas_w))
        min_y = int(np.clip(np.floor(warped[1, :].min()), 0, canvas_h))
        max_y = int(np.clip(np.ceil(warped[1, :].max()), 0, canvas_h))

        if max_x <= min_x or max_y <= min_y:
            continue

        xv, yv = np.meshgrid(np.arange(min_x, max_x, dtype=np.float64),
                             np.arange(min_y, max_y, dtype=np.float64))
        coords = np.stack([xv, yv, np.ones_like(xv)], axis=0).reshape(3, -1)
        src = T_inv @ coords
        src /= src[2, :]
        src_x = src[0, :]
        src_y = src[1, :]

        valid = (src_x >= 0) & (src_x <= frame_w - 1) & (src_y >= 0) & (
                src_y <= frame_h - 1)
        if not np.any(valid):
            continue

        roi_h = max_y - min_y
        roi_w = max_x - min_x
        src_x_grid = src_x.reshape(roi_h, roi_w)
        src_y_grid = src_y.reshape(roi_h, roi_w)
        mask_grid = valid.reshape(roi_h, roi_w)

        for ch in range(3):
            sampled = map_coordinates(frame[..., ch], [src_y_grid, src_x_grid],
                                      order=interp_order, prefilter=prefilter)
            target = pano_canvas[min_y:max_y, min_x:max_x, ch]
            target[mask_grid] = sampled[mask_grid]
            pano_canvas[min_y:max_y, min_x:max_x, ch] = target

    # Crop fully-black columns (left/right) before converting to uint8
    pano_canvas = crop_black_columns(pano_canvas, black_thresh=0, margin=10)
    return pano_canvas


def align_pair(im1, im2, border_cut):
    """Align two grayscale frames using a coarse-to-fine LK scheme.
    Params:
        im1 (np.ndarray): (H,W) reference.
        im2 (np.ndarray): (H,W) target.
        border_cut (int): Border cut for LK.
    Returns:
        tuple[float,float,float]: (u,v,theta).
    """
    min_dim = min(im1.shape[:2])
    levels = max(1, int(np.floor(np.log2(min_dim / MIN_PYRAMID_SIZE))) + 1)
    pyr1 = gaussian_pyramid(im1, levels)
    pyr2 = gaussian_pyramid(im2, levels)

    u = v = theta = 0.0
    # iterate pyramid from coarse to fine
    for level in reversed(range(len(pyr1))):
        if level < len(pyr1) - 1:
            u *= 2.0
            v *= 2.0
        u, v, theta = _lucas_kanade_optimization(pyr1[level], pyr2[level],
                                                 border_cut, u, v, theta)

    return u, v, theta


def load_frames_for_test(input_frames_path):
    """
    Load video frames from a directory for testing purposes.
    :param input_frames_path: Path to the directory containing video frames.
    :return: A numpy array of loaded video frames.
    """
    frames = []
    for filename in sorted(os.listdir(input_frames_path)):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(input_frames_path, filename)
            frame = imageio.v2.imread(frame_path)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Expected RGB frame (H,W,3), got shape {frame.shape} for {filename}")
            frame = frame.astype(np.float64) / 255.0
            frames.append(frame)
    if not frames:
        raise ValueError("No frames found in the specified directory.")
    return np.stack(frames, axis=0)


def crop_black_columns(img, black_thresh=0, margin=0):
    """Crop ONLY fully-black columns from left/right.
    Params:
        img (np.ndarray): (H,W) or (H,W,3) image.
        black_thresh (float): Pixels <= threshold are considered black.
        margin (int): Keep this many extra columns beyond detected content.
    Returns:
        np.ndarray: Cropped image.
    """
    if img.ndim == 2:
        non_black = img > black_thresh  # (H, W)
    elif img.ndim == 3:
        non_black = np.any(img > black_thresh, axis=2)  # (H, W)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # columns that have any non-black pixel
    col_has_content = np.any(non_black, axis=0)  # (W,)

    if not np.any(col_has_content):
        return img  # all columns are black (or below thresh)

    xs = np.where(col_has_content)[0]
    x0, x1 = xs[0], xs[-1] + 1

    if margin > 0:
        w = col_has_content.shape[0]
        x0 = max(0, x0 - margin)
        x1 = min(w, x1 + margin)

    if img.ndim == 2:
        return img[:, x0:x1]
    return img[:, x0:x1, :]


def dynamic_mosaic(frames, transforms, canvas, padding=2, start=0.2, stop=0.8,
                   num_views=10, back_n_forth=False,
                   include_edge_full_strip=False):
    """
    Create a dynamic mosaic video by rendering strip panoramas with varying strip anchors.
    Args:
        frames (np.ndarray): Input video frames of shape (N, H, W, 3) or (N, H, W).
        transforms (List[np.ndarray]): List of 3x3 transformation matrices for each frame.
        canvas (Tuple[int, int, float, float]): Canvas geometry as returned by compute_canvas_geometry.
        padding (int): Padding to add to the strip width.
        start (float): Starting anchor position (0.0 to 1.0).
        stop (float): Ending anchor position (0.0 to 1.0).
        num_views (int): Number of frames to generate between start and stop.
        back_n_forth (bool): If True, the video will play forward and then backward.
    Returns:
        :param frames: List of panorama frames as uint8 arrays.
    """
    movie_frames = []

    for anchor in np.linspace(start, stop, num_views):
        print(f"Creating panorama for anchor {anchor:.2f}...")
        pan = render_strip_panorama(frames, transforms, canvas,
                                    strip_anchor=anchor, strip_padding=padding,
                                    include_edge_full_strip=include_edge_full_strip)
        pan_uint8 = (np.clip(pan, 0, 1) * 255).astype(np.uint8)
        movie_frames.append(pan_uint8)

    # append frames in reverse order to create a back-and-forth effect
    if back_n_forth:
        movie_frames += movie_frames[::-1]
    return movie_frames


def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4
    :param input_frames_path: path to a dir with input video frames.
    We will test your code with a dir that has K frames, each in the format
    "frame_i:05d.jpg" (e.g., frame_00000.jpg, frame_00001.jpg, frame_00002.jpg, ...).
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorma frames (of size n_out_frames),
    each list item should be a PIL image of a generated panorama.
    """
    PADDING = 4
    BORDER_CUT = 15
    ENABLE_ROTATION = False
    INCLUDE_EDGE_FULL_STRIP = False
    NUM_OF_BLURS = 1
    START_ANCHOR = 0.2  # Safe margins
    STOP_ANCHOR = 0.8

    # 0. Load video & blur
    raw_frames = load_frames_for_test(input_frames_path)
    for _ in range(NUM_OF_BLURS):
        raw_frames = blur_video(raw_frames, REDUCE_KERNEL)

    # 1. Compute Motion
    motion_data = compute_motion(raw_frames, BORDER_CUT)

    # 1.2 invert if motion is right-to-left
    if estimate_motion_dir(motion_data) == "RTL":
        raw_frames = raw_frames[::-1]
        motion_data = [(-u, -v, -theta) for u, v, theta in motion_data[::-1]]

    # 2. Stabilize Video
    stable_frames = stabilize_video(raw_frames, motion_data,
                                    enable_rotation=ENABLE_ROTATION)
    stabilized_motion = [(u, 0, 0) for u, v, theta in motion_data]

    # 3. Motion composition: align all frames to same coordinate system
    transforms = compute_cumulative_transforms(stabilized_motion)
    geo = compute_canvas_geometry(transforms, raw_frames.shape[1],
                                  raw_frames.shape[2])

    # 4. Create Movie of Multi-Perspective mosaics
    movie_frames = dynamic_mosaic(stable_frames, transforms, geo,
                                  padding=PADDING,
                                  include_edge_full_strip=INCLUDE_EDGE_FULL_STRIP,
                                  start=START_ANCHOR, stop=STOP_ANCHOR,
                                  num_views=n_out_frames, back_n_forth=True)

    return [PIL.Image.fromarray(f) for f in movie_frames]

# tar file of video cmd:
# tar -cvf videos.tar viewpoint_input.mp4 viewpoint_result.mp4 good_input.mp4 good_result.mp4 bad_input.mp4 bad_result.mp4

# code file tar cmd:
# tar -cvf ex4.tar ex4.py requirements.txt