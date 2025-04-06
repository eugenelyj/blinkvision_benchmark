import numpy as np
import cv2
import random
import imageio
import imageio.v3 as iio

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, rad_max=None, return_rad_max=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    if rad_max is None:
        rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    flow_vis = flow_uv_to_colors(u, v, convert_to_bgr)

    if return_rad_max:
        return flow_vis, rad_max
    else:
        return flow_vis


def apply_colormap(data, min_v=None, max_v=None):
    data = data.copy()
    min_v = min_v if min_v is not None else np.min(data)
    max_v = max_v if max_v is not None else np.max(data)
    data = (data - min_v) / (max_v - min_v)
    data = np.clip(data, 0, 1)
    data = (data * 255).astype(np.uint8)
    data = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)
    return data


def visualize_optical_flow(optical_flow_data_list):
    flow_vis_list = []
    for optical_flow_data in optical_flow_data_list:
        flow_vis = flow_to_image(optical_flow_data)
        flow_vis_list.append(flow_vis)
    return flow_vis_list

def visualize_depth(depth_data_list):
    depth_vis_list = []
    for depth in depth_data_list:
        depth = np.clip(depth, 0, np.percentile(depth, 99))
        depth_vis = apply_colormap(depth)
        depth_vis_list.append(depth_vis)
    return depth_vis_list

def visualize_trajectory(particle_track, rgb_imgs, track_info, segment_length=5, skip=50):
    track_start, track_end = track_info
    num_frame = track_end - track_start
    height, width = rgb_imgs[0].shape[:2]
    frame_list = []

    mask = np.ones([1, height, width], dtype=bool)

    colormap = np.array([[random.randint(0, 255) for _ in range(3)] for _ in range(1)])

    x = np.array(range(skip, width, skip))
    y = np.array(range(skip, height, skip))
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.flatten(), yv.flatten()

    traj_imgs = np.zeros((num_frame, height, width, 3), np.uint8)

    particle_data = np.array(particle_track)[..., :2]
    # particle status:
    # particle_VISIBLE = 1
    # particle_OCCLUDED = 0
    # particle_OUTBOUND = -1
    # particle_NO_CAST = -2
    # particle_BEHIND_CAMERA = -3
    valid_data = np.array(particle_track)[..., 3] >= -1

    for i in range(1, num_frame):
        jj = max(1, i-segment_length+1)
        for j in range(jj, i+1):
            for k in range(len(xv)):
                u, v = int(xv[k]), int(yv[k])
                if not mask[0,v,u]:
                    continue
                if not valid_data[j-1,v,u] or not valid_data[j,v,u]:
                    continue

                try:
                    cv2.line(traj_imgs[i], tuple(particle_data[j-1,v,u].astype(int)),
                                tuple(particle_data[j,v,u].astype(int)),
                                tuple(colormap[0].tolist()), thickness=2)
                except:
                    import ipdb; ipdb.set_trace()

    for local_idx, global_idx in enumerate(range(track_start, track_end)):
        rgb = rgb_imgs[global_idx]
        alpha = np.sum(traj_imgs[local_idx], -1) > 0
        alpha = np.stack([alpha] * 3, -1)
        rgb = alpha * traj_imgs[local_idx] + (1 - alpha) * rgb
        frame_list.append(rgb.astype(np.uint8))

    return frame_list


def make_video(data_list, outvid=None, fps=10, max_data_one_row=3, max_width=1000, is_color=True, format="mp4v"):
    video_len = len(data_list[0])
    if video_len < 1:
        return

    for i, data in enumerate(data_list):
        if len(data) != video_len:
            raise ValueError(f'data_list[{i}] has {len(data)} frames, but data_list[0] has {video_len} frames')

    frames_list = []
    for i in range(video_len):
        # concat data along the width axis, if the number of data is larger than max_data_one_row, then go to the next line
        frames = [data[i] for data in data_list]
        rows = []
        # First determine if we'll have multiple rows
        total_rows = (len(frames) + max_data_one_row - 1) // max_data_one_row
        needs_padding = total_rows > 1

        for j in range(0, len(frames), max_data_one_row):
            row = frames[j:j + max_data_one_row]
            # Pad row with empty frames only if we have multiple rows
            if needs_padding:
                while len(row) < max_data_one_row:
                    empty_frame = np.zeros_like(frames[0])
                    row.append(empty_frame)
                
            # Ensure all images in the row have the same height
            row_height = max(frame.shape[0] for frame in row)
            # Resize images if they exceed max_width
            scale = min(1.0, max_width / (sum(frame.shape[1] for frame in row)))
            resized_row = []
            for frame in row:
                if scale < 1.0:
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(row_height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                resized_row.append(frame)
            rows.append(cv2.hconcat(resized_row))
        
        # Concatenate all rows vertically
        img = cv2.vconcat(rows)
        frames_list.append(img)

    # Write all frames at once
    iio.imwrite(
        outvid,
        frames_list,
        fps=fps,
        codec='h264',  # Use h264 codec for better compatibility
        quality=7      # Quality from 0 to 10, 10 being the best
    )