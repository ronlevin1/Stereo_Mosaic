import os
from typing import List, Optional, Sequence, Tuple

import imageio
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d, map_coordinates
from skimage.color import rgb2gray

REDUCE_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

"------------------------------------------------------------------------------"
"-------------------------------- Utils ---------------------------------------"
"------------------------------------------------------------------------------"


def load_video_frames(
        filename: str,
        inputs_folder: str = "Exercise Inputs",
        max_frames: Optional[int] = None,
        downscale_factor: int = 1,
) -> np.ndarray:
    video_path = os.path.join(inputs_folder, filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    frames: List[np.ndarray] = []
    try:
        for idx, frame in enumerate(reader):
            if max_frames is not None and idx >= max_frames:
                break
            if downscale_factor > 1:
                frame = frame[::downscale_factor, ::downscale_factor, ...]
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


def stabilize_video(
        frames_rgb: np.ndarray,
        step_size: int,
        border_cut: int,
        enable_rotation: bool = True,
) -> np.ndarray:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(
            "stabilize_video expects RGB frames with shape (N,H,W,3)")

    stabilized_frames = [frames_rgb[0]]
    drift_v = 0.0
    drift_theta = 0.0

    for idx in range(frames_rgb.shape[0] - 1):
        im1_gray = rgb2gray(frames_rgb[idx])
        im2_gray = rgb2gray(frames_rgb[idx + 1])
        u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)
        drift_v += v
        if enable_rotation:
            drift_theta += theta
        warped = warp_image(frames_rgb[idx + 1], 0.0, drift_v, drift_theta)
        stabilized_frames.append(warped)

    return np.stack(stabilized_frames, axis=0)


def compute_camera_path(
        frames_rgb: np.ndarray,
        step_size: int,
        border_cut: int,
        convergence_point: Optional[Tuple[float, float]] = None,
) -> List[np.ndarray]:
    if frames_rgb.ndim != 4:
        raise ValueError("frames_rgb must be (N,H,W,3)")

    transforms: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    current_T = np.eye(3, dtype=np.float64)

    for idx in range(frames_rgb.shape[0] - 1):
        im1_gray = rgb2gray(frames_rgb[idx])
        im2_gray = rgb2gray(frames_rgb[idx + 1])
        u, v, theta = optical_flow(im1_gray, im2_gray, step_size, border_cut)
        M = build_matrix(-u, -v, -theta)
        current_T = current_T @ M
        transforms.append(current_T.copy())

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
    min_dim = min(im1.shape[:2])
    levels = max(1, int(np.log2(max(min_dim // step_size, 1))) + 1)
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
                   start=0.2, stop=0.8, num=10):
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
        num (int): Number of frames to generate between start and stop.
    Returns:
        List[np.ndarray]: List of panorama frames as uint8 arrays.
    """
    movie_frames = []

    for anchor in np.linspace(start, stop, num):
        print(f"Creating panorama for anchor {anchor:.2f}...")

        pan = render_strip_panorama(frames, transforms, canvas,
                                    strip_anchor=anchor,
                                    strip_padding=padding,
                                    grayscale_out=grayscale)

        pan_uint8 = (np.clip(pan, 0, 1) * 255).astype(np.uint8)
        movie_frames.append(pan_uint8)

    # append all frames in reverse order to create a back-and-forth effect
    movie_frames += movie_frames[::-1]
    return movie_frames
