### Prompt for AI / Copilot

**Project Objective:**
Create a Python module (`ex4.py`) for Video Panorama Stitching (Pushbroom Stereo) using Optical Flow.
The code must be modular, separating the logic into a strict pipeline of 5 steps.

**General Constraints:**

1. **Libraries:** Use `numpy`, `scipy.ndimage` (map_coordinates), `skimage.color` (rgb2gray), and `imageio`.
2. **Data Types:** All images should be processed as `float64` in the range `[0, 1]`.
3. **Color Handling:** The pipeline must support RGB video input. Motion calculations are done on Grayscale, but warping/rendering is done on RGB (unless specified otherwise).

---

### Module Architecture & API Requirements

Implement the following functions with these exact signatures and behaviors:

#### 1. Core Math Helpers

* **`optical_flow(im1, im2, step_size, border_cut)`**:
* Implement Iterative Lucas-Kanade with Gaussian Pyramids.
* Return `u, v` (translation) and `theta` (rotation).


* **`warp_image(im, u, v, theta)`**:
* Apply inverse warping based on rigid motion.
* **Crucial:** Must handle both 2D (H,W) and 3D (H,W,3) arrays.



#### 2. Pipeline Step 1: Loading

* **`load_video_frames(filename, ...)`**:
* Load video frames.
* Convert to `float64` [0,1].
* **Do NOT** convert to grayscale here. Return RGB frames (N, H, W, 3).



#### 3. Pipeline Step 2: Stabilization

* **`stabilize_video(frames_rgb, step_size, border_cut, enable_rotation=True)`**:
* Input: RGB frames.
* Logic:
1. Convert pair to Grayscale internally.
2. Calculate Optical Flow (`u, v, theta`).
3. Accumulate `dy` and `dtheta`.
4. If `enable_rotation` is False, force `dtheta` accumulation to 0.
5. Apply `warp_image` on the **RGB** frames to cancel `dy` and `dtheta` (keep `dx` untouched).


* Output: Stabilized RGB frames.


#### 4. Pipeline Step 3: Camera Path

* **`compute_camera_path(frames_rgb, step_size, border_cut, convergence_point=None)`**:
* Input: Stabilized RGB frames.
* Logic:
1. Calculate Optical Flow between stabilized frames (on gray versions).
2. Build 3x3 Homography matrices ().
3. Accumulate global transforms ().
4. **Feature:** If `convergence_point=(x,y)` is provided, adjust the global transform sequence so that this specific point remains stationary (canceling parallax for that depth), instead of the background (infinity).


* Output: List of 3x3 matrices.



#### 5. Pipeline Step 4: Geometry

* **`compute_canvas_geometry(transforms, frame_h, frame_w)`**:
* Logic: Project all 4 corners of all frames using the transforms.
* Calculate bounding box (`min_x, max_x, min_y, max_y`).
* Output: `canvas_h`, `canvas_w`, `dx`, `dy` (offsets to ensure positive coordinates).



#### 6. Pipeline Step 5: Rendering (The Stitching Logic)

* **`render_strip_panorama(frames, transforms, canvas_geometry, strip_anchor=0.5, strip_padding=2, grayscale_out=False, interp_order=1, prefilter=False)`**:
* **`strip_anchor`**: Float [0.0 - 1.0]. Determines the center of the strip relative to frame width ().
* 0.0 = Left strips, 0.5 = Center strips, 1.0 = Right strips.


* **`strip_padding`**: Integer. Adds extra width to each strip (overlap) to prevent black gaps between frames.
* **`grayscale_out`**: If True, the output panorama is 2D (H,W). Convert frames to gray on-the-fly during rendering to save memory. If False, output is 3D (H,W,3).
* **`interp_order`**: Parameter for `map_coordinates` (1=Linear, 3=Cubic).
* **Logic:**
1. Create canvas based on geometry.
2. For each frame, calculate dynamic strip width based on distance to the next frame's center.
3. Inverse map pixels from Canvas -> Frame.
4. Paste valid pixels.





---

### Usage Example (Main Block)

Please verify the code works with the following flow:

```python
# 1. Load & Stabilize
raw = load_video_frames("video.mp4")
stable = stabilize_video(raw, step_size=16, border_cut=15, enable_rotation=True)

# 2. Compute Path
matrices = compute_camera_path(stable, step_size=16, border_cut=15)
geo = compute_canvas_geometry(matrices, raw.shape[1], raw.shape[2])

# 3. Render (Example: Color panorama from center strips)
pan = render_strip_panorama(stable, matrices, geo, strip_anchor=0.5, strip_padding=2, grayscale_out=False)

# 4. Save
save_panorama(pan, "output.jpg")

```