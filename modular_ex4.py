import os
from typing import List, Optional, Sequence, Tuple

import imageio
import PIL.Image
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d, map_coordinates
from skimage.color import rgb2gray

REDUCE_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

"------------------------------------------------------------------------------"
"-------------------------------- Utils ---------------------------------------"
"------------------------------------------------------------------------------"


def load_video_frames(filename: str, inputs_folder: str = "Exercise Inputs",
                      spatial_downscale: int = 1) -> np.ndarray:
    """
    Load video frames from a file, with optional spatial and temporal downscaling.
    """
    video_path = os.path.join(inputs_folder, filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    frames: List[np.ndarray] = []
    try:
        for idx, frame in enumerate(reader):
            if spatial_downscale > 1:  # in-frame, drop pixels
                frame = frame[::spatial_downscale, ::spatial_downscale, ...]
            frame = _ensure_rgb(frame).astype(np.float64) / 255.0
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        raise ValueError("No frames loaded; check input parameters")
    return np.stack(frames, axis=0)


def warp_image(im: np.ndarray, u: float, v: float, theta: float) -> np.ndarray:
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


def gaussian_pyramid(img: np.ndarray, num_levels: int) -> List[np.ndarray]:
    pyramid = [img.astype(np.float64)]
    current = pyramid[0]
    for _ in range(1, num_levels):
        if min(current.shape[:2]) < 2:
            break
        current = reduce(current)
        pyramid.append(current)
    return pyramid


def reduce(img: np.ndarray) -> np.ndarray:
    smoothed = blur(img, REDUCE_KERNEL)
    return smoothed[::2, ::2]


def blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64, copy=False)
    if img.ndim == 2:
        return _blur_single_channel(img, kernel)
    blurred = np.zeros_like(img)
    for ch in range(img.shape[2]):
        blurred[..., ch] = _blur_single_channel(img[..., ch], kernel)
    return blurred


def build_matrix(u: float, v: float, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, u], [s, c, v], [0.0, 0.0, 1.0]], dtype=np.float64)


def _blur_single_channel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    tmp = convolve1d(img, kernel, axis=1, mode="nearest")
    return convolve1d(tmp, kernel, axis=0, mode="nearest")


def _lucas_kanade_step(
        I1: np.ndarray,
        I2: np.ndarray,
        border_cut: int,
) -> Tuple[float, float, float]:
    kernel_x = np.array([[1.0, 0.0, -1.0]]) / 2.0
    kernel_y = kernel_x.T
    Ix = signal.convolve2d(I1, kernel_x, mode="same", boundary="symm")
    Iy = signal.convolve2d(I1, kernel_y, mode="same", boundary="symm")
    It = I2 - I1

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
    It = It[slices]

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
    IxIt = np.sum(Ix * It)
    IyIt = np.sum(Iy * It)
    IthIt = np.sum(I_theta * It)
    B = np.array([-IxIt, -IyIt, -IthIt], dtype=np.float64)

    try:
        res = np.linalg.solve(A, B)
        return float(res[0]), float(res[1]), float(res[2])
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], repeats=3, axis=2)
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    return frame


def _anchor_convergence(
        transforms: Sequence[np.ndarray],
        convergence_point: Tuple[float, float],
) -> List[np.ndarray]:
    anchor = np.array([convergence_point[0], convergence_point[1], 1.0])
    ref = anchor.copy()
    anchored: List[np.ndarray] = []
    for T in transforms:
        warped = T @ anchor
        warped /= warped[2]
        delta_x = warped[0] - ref[0]
        delta_y = warped[1] - ref[1]
        adjust = np.eye(3, dtype=np.float64)
        adjust[0, 2] = -delta_x
        adjust[1, 2] = -delta_y
        anchored.append(adjust @ T)
    return anchored


def _estimate_strip_width(
        transforms: Sequence[np.ndarray],
        idx: int,
        target_center: int,
        frame_h: int,
        frame_w: int,
        strip_padding: int,
        T_offset: np.ndarray,
) -> int:
    ys = [0.0, frame_h / 2.0, float(frame_h)]
    dists: List[float] = []

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
    strip_width = int(np.ceil(max_dist * 1.3)) + strip_padding
    return max(1, min(strip_width, frame_w))


"------------------------------------------------------------------------------"
"------------------------------------------------------------------------------"
"------------------------------------------------------------------------------"


def compute_motion(frames_rgb, step_size, border_cut):
    """
    Calculates raw_frames motion between consecutive frames.
    Returns a list of (u, v, theta) tuples.
    """
    motion_data = []

    im1_gray = rgb2gray(frames_rgb[0])
    for i in range(len(frames_rgb) - 1):
        im2_gray = rgb2gray(frames_rgb[i + 1])
        u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)
        motion_data.append((u, v, theta))
        im1_gray = im2_gray

    return motion_data


def stabilize_video(frames_rgb, motion_data, enable_rotation=True):
    """
    Stabilize rotations and Y-axis translations
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


def compute_camera_path(motion_data, convergence_point=None):
    """
    Compute the camera path as a list of transformation matrices.
    """
    transforms = [np.eye(3)]
    current_T = np.eye(3)

    for (u, v, theta) in motion_data:
        M = build_matrix(-u, -v, -theta)
        current_T = current_T @ M
        transforms.append(current_T)

    # todo: fix this !
    # mid_idx = len(transforms) // 2
    # mid_inv = np.linalg.inv(transforms[mid_idx])
    #
    # transforms = [mid_inv @ T for T in transforms]

    # Convergence Point Logic (אם קיים)
    if convergence_point is not None:
        transforms = _anchor_convergence(transforms, convergence_point)

    return transforms


def compute_canvas_geometry(
        transforms: Sequence[np.ndarray],
        frame_h: int,
        frame_w: int,
) -> Tuple[int, int, float, float]:
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


def render_strip_panorama(
        frames: np.ndarray,
        transforms: Sequence[np.ndarray],
        canvas_geometry: Tuple[int, int, float, float],
        strip_anchor: float = 0.5,
        strip_padding: int = 2,
        grayscale_out: bool = False,
        interp_order: int = 1,
        prefilter: bool = False,
) -> np.ndarray:
    if frames.ndim not in (3, 4):
        raise ValueError("frames must be (N,H,W,3) or (N,H,W)")
    if frames.ndim == 4 and frames.shape[-1] != 3:
        raise ValueError("RGB frames must have 3 channels")

    frame_h, frame_w = frames.shape[1:3]
    canvas_h, canvas_w, dy, dx = canvas_geometry

    canvas = (
        np.zeros((canvas_h, canvas_w), dtype=np.float64)
        if grayscale_out
        else np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    )

    T_offset = np.eye(3, dtype=np.float64)
    T_offset[0, 2] = dx
    T_offset[1, 2] = dy

    target_center = int(np.clip(strip_anchor, 0.0, 1.0) * (frame_w - 1))

    for idx, frame in enumerate(frames):
        T_total = T_offset @ transforms[idx]
        T_inv = np.linalg.inv(T_total)

        strip_width = _estimate_strip_width(
            transforms,
            idx,
            target_center,
            frame_h,
            frame_w,
            strip_padding,
            T_offset,
        )

        strip_start = max(0, target_center - strip_width // 2)
        strip_end = min(frame_w, strip_start + strip_width)
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

        xv, yv = np.meshgrid(
            np.arange(min_x, max_x, dtype=np.float64),
            np.arange(min_y, max_y, dtype=np.float64),
        )
        coords = np.stack([xv, yv, np.ones_like(xv)], axis=0).reshape(3, -1)
        src = T_inv @ coords
        src /= src[2, :]
        src_x = src[0, :]
        src_y = src[1, :]

        valid = (
                (src_x >= 0)
                & (src_x <= frame_w - 1)
                & (src_y >= 0)
                & (src_y <= frame_h - 1)
        )
        if not np.any(valid):
            continue

        roi_h = max_y - min_y
        roi_w = max_x - min_x
        src_x_grid = src_x.reshape(roi_h, roi_w)
        src_y_grid = src_y.reshape(roi_h, roi_w)
        mask_grid = valid.reshape(roi_h, roi_w)

        if grayscale_out:
            src_frame = rgb2gray(frame) if frame.ndim == 3 else frame
            sampled = map_coordinates(
                src_frame,
                [src_y_grid, src_x_grid],
                order=interp_order,
                prefilter=prefilter,
            )
            target = canvas[min_y:max_y, min_x:max_x]
            target[mask_grid] = sampled[mask_grid]
            canvas[min_y:max_y, min_x:max_x] = target
        else:
            for ch in range(3):
                sampled = map_coordinates(
                    frame[..., ch],
                    [src_y_grid, src_x_grid],
                    order=interp_order,
                    prefilter=prefilter,
                )
                target = canvas[min_y:max_y, min_x:max_x, ch]
                target[mask_grid] = sampled[mask_grid]
                canvas[min_y:max_y, min_x:max_x, ch] = target

    return canvas


def optical_flow(
        im1: np.ndarray,
        im2: np.ndarray,
        step_size: int,
        border_cut: int,
) -> Tuple[float, float, float]:
    MIN_PIXELS = 16.0
    min_dim = min(im1.shape[:2])
    levels = max(1, int(np.floor(np.log2(max(min_dim / MIN_PIXELS, 1.0)))) + 1)
    pyr1 = gaussian_pyramid(im1, levels)
    pyr2 = gaussian_pyramid(im2, levels)

    u = v = theta = 0.0
    for level in reversed(range(len(pyr1))):
        if level < len(pyr1) - 1:
            u *= 2.0
            v *= 2.0
        warped = warp_image(pyr2[level], u, v, theta)
        du, dv, dtheta = _lucas_kanade_step(pyr1[level], warped, border_cut)
        u += du
        v += dv
        theta += dtheta
    return u, v, theta


def dynamic_mosaic(frames, transforms, canvas,
                   padding=2, grayscale=False,
                   start=0.2, stop=0.8, num_views=10, back_n_forth=False):
    """
    Create a dynamic mosaic video by rendering strip panoramas with varying strip anchors.
    Args:
        frames (np.ndarray): Input video frames of shape (N, H, W, 3) or (N, H, W).
        transforms (List[np.ndarray]): List of 3x3 transformation matrices for each frame.
        canvas (Tuple[int, int, float, float]): Canvas geometry as returned by compute_canvas_geometry.
        padding (int): Padding to add to the strip width.
        grayscale (bool): Whether to output grayscale panoramas.
        start (float): Starting anchor position (0.0 to 1.0).
        stop (float): Ending anchor position (0.0 to 1.0).
        num_views (int): Number of frames to generate between start and stop.
        back_n_forth (bool): If True, the video will play forward and then backward.
    Returns:
        List[np.ndarray]: List of panorama frames as uint8 arrays.
        :param back_n_forth:
    """
    movie_frames = []

    for anchor in np.linspace(start, stop, num_views):
        print(f"Creating panorama for anchor {anchor:.2f}...")

        pan = render_strip_panorama(frames, transforms, canvas,
                                    strip_anchor=anchor,
                                    strip_padding=padding,
                                    grayscale_out=grayscale)

        pan_uint8 = (np.clip(pan, 0, 1) * 255).astype(np.uint8)
        movie_frames.append(pan_uint8)

    # append all frames in reverse order to create a back-and-forth effect
    if back_n_forth:
        movie_frames += movie_frames[::-1]
    return movie_frames


# TODO: test this.
def crop_panoramas_to_common_area(panoramas):
    """
    Implements Instruction 6: Neutralize the shift between panoramas.
    We find the common valid area across all panoramas and crop them.
    Since panoramas are mainly shifted horizontally, we focus on X cropping.
    """
    if not panoramas:
        return panoramas

    # Assuming all panoramas have the same height and mostly align vertically
    # We need to find the "valid" width range.

    # Heuristic:
    # The strip shift causes the image content to shift.
    # We want to keep the center intersection.

    h, w = panoramas[0].shape[:2]

    # In a proper stitching, usually:
    # Frame 0 (Leftmost strip) has valid pixels starting early but ending early.
    # Frame N (Rightmost strip) has valid pixels starting late but ending late.

    # Simple approach based on the instructions:
    # "Crop right columns from right panorama, left columns from left panorama"

    # Let's find the first non-black column of the Last Panorama (Rightmost view)
    # and the last non-black column of the First Panorama (Leftmost view).

    # Note: This depends on how your 'render' outputs the black background.
    # Assuming standard output where background is 0.

    # Find Left Crop limit (determined by the Rightmost View, which starts latest)
    # Actually, visual parallax works inversely to strip selection:
    # Left Strip = Right Viewpoint (Content moves Left)
    # Right Strip = Left Viewpoint (Content moves Right)

    # Let's keep it simple: Find the max 'first_col' and min 'last_col' across all frames

    max_first_col = 0
    min_last_col = w

    for pan in panoramas:
        # Convert to gray just for check
        if pan.ndim == 3:
            check_img = pan.mean(axis=2)
        else:
            check_img = pan

        # Sum columns to find where data exists
        col_sums = check_img.sum(axis=0)
        valid_cols = np.where(col_sums > 0)[0]

        if len(valid_cols) > 0:
            first_col = valid_cols[0]
            last_col = valid_cols[-1]

            max_first_col = max(max_first_col, first_col)
            min_last_col = min(min_last_col, last_col)

    # Check if we have a valid overlap
    if max_first_col >= min_last_col:
        print("Warning: No common overlap found. Returning original frames.")
        return panoramas

    print(
        f"Cropping panoramas to common width: {max_first_col} to {min_last_col}")

    cropped_panoramas = []
    for pan in panoramas:
        cropped_panoramas.append(pan[:, max_first_col:min_last_col])

    return cropped_panoramas


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
    PADDING = 2
    GRAYSCALE = False
    STEP_SIZE = 16
    BORDER_CUT = 15
    START_ANCHOR = 0.2  # Safe margins
    STOP_ANCHOR = 0.8

    # 1. Load & Stabilize
    raw = load_frames_for_test(input_frames_path)
    stable = stabilize_video(raw, step_size=STEP_SIZE, border_cut=BORDER_CUT,
                             enable_rotation=True)

    # 2. Compute Path
    matrices = compute_camera_path(stable, step_size=STEP_SIZE,
                                   border_cut=BORDER_CUT)
    geo = compute_canvas_geometry(matrices, raw.shape[1], raw.shape[2])

    # 3. Render
    movie_frames = dynamic_mosaic(stable, matrices, geo,
                                  num_views=n_out_frames,
                                  start=START_ANCHOR,
                                  stop=STOP_ANCHOR,
                                  padding=PADDING,
                                  grayscale=GRAYSCALE,
                                  back_n_forth=False)

    # 4. Post-Process (Neutralize Shift)
    # todo: test this function
    # movie_frames = crop_panoramas_to_common_area(movie_frames)

    # 5. Convert to PIL
    return [PIL.Image.fromarray(f) for f in movie_frames]
    # return movie_frames # numpy arrays for easier testing


def load_frames_for_test(input_frames_path):
    """
    Load video frames from a directory for testing purposes.
    :param input_frames_path: Path to the directory containing video frames.
    :return: A numpy array of loaded video frames.
    """
    frames = []
    for filename in sorted(os.listdir(input_frames_path)):
        if filename.endswith('.jpg'):
            frame_path = os.path.join(input_frames_path, filename)
            frame = imageio.v2.imread(frame_path)
            frame = _ensure_rgb(frame).astype(np.float64) / 255.0
            frames.append(frame)
    if not frames:
        raise ValueError("No frames found in the specified directory.")
    return np.stack(frames, axis=0)

# TODO: remove redundancy of calling optical_flow.
#       build a func compute_motion for the core math and call it from both
#       stabilize_video and compute_camera_path.
#       as is now:
#     in my code, these lines appear in 2 places:
#     im1_gray = rgb2gray(frames_rgb[0])
#         for idx in range(frames_rgb.shape[0] - 1):
#             im2_gray = rgb2gray(frames_rgb[idx + 1])
#             u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)
#     both in stabilize_video and compute_camera_path. in each of them i call optical_path. this is redundant.

# TODO:
#  - blur video as first step (especially Kessaria)
#  - split LK to rotation and translation components
#        a. rotation: with SIFT+RANSAC on horizontal lines\features, since they
#            are in same distance from camera
#        b. translation: with LK on the ROTATED frames
#  - make these ^ functions return te transforms mtx for future reusal.
#  - set middle frame as the reference for stabilization
#  - NOTE: banana in result is OK. the right form is 'smiling' banana :)

# todo - OPTIMIZE RUNTIME to ~100sec
#  V - in warp_image: set order=1, prefilter=False
#  - in render_strip_panorama: set order=2-3, prefilter=True for good quality
#  - in pyramid levels: stop earlier, when dims are < 16/32 pixels.
#  - reduce video resolution when computing transforms (e.g., by 2x)
#  - remove loops, use numpy vector operations,
#           reduce write/any access to disk
