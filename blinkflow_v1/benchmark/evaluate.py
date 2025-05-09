import numpy as np
import hashlib
import time
import torch
import os
import math
import csv
import cv2
import json
from PIL import Image


MAX_FLOW = 400
TAG_VALUE = 2502001 # full data
TAG_VALUE_2 = 2502002 # sampled data


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


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, return_rad_max=False, rad_max=None):
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
    rad_max = np.max(rad) if rad_max is None else rad_max
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    if return_rad_max:
        return flow_uv_to_colors(u, v, convert_to_bgr), rad_max
    else:
        return flow_uv_to_colors(u, v, convert_to_bgr)


def visualize_error_map(flow_pred, flow_gt):
    error_map = np.linalg.norm(flow_pred - flow_gt, axis=-1)
    error_map = error_map / np.max(error_map)
    error_map = (error_map * 255).astype(np.uint8)
    error_map = np.repeat(error_map[..., None], 3, axis=-1)

    return error_map


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg



class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            if key not in self._dict:
                self._dict[key] = RunningAverage()
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is not None:
            return {key: value.get_value() for key, value in self._dict.items()}
        else:
            return {}

def readFlow(fn, is_gt=False):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.int32, count=1)
        if magic != TAG_VALUE_2:
            raise ValueError('Magic number incorrect. Invalid .flo file')

        contains_vis = np.fromfile(f, np.int32, count=1)
        num_samples = np.fromfile(f, np.int32, count=1)
        data = np.fromfile(f, np.float32, count=2*int(num_samples))
        data_dict = {
            'flow': np.resize(data, (int(num_samples), 2))
        }

        if contains_vis and is_gt:
            rgb_image = np.fromfile(f, np.uint8, count=3*int(num_samples))
            event_image = np.fromfile(f, np.uint8, count=3*int(num_samples))
            data_dict['rgb'] = np.reshape(rgb_image, [480, 640, 3])
            data_dict['event'] = np.reshape(event_image, [480, 640, 3])

        return data_dict, bool(contains_vis)


def compute_out(flow_gt, pred, valid):
    epe = torch.sum((pred - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt() + 1e-6
    # 3.0 and 0.05 is tooken from KITTI devkit
    # Inliers are defined as EPE < 3 pixels or < 5%
    out = ((epe > 3.0) & ((epe / mag) > 0.05)).to(torch.float32)
    out = out.view(-1)[valid.view(-1)]
    out = 100 * torch.mean(out)

    return out.item()

def compute_angular_error(flow_gt, pred, valid):
    flow_gt = flow_gt / (torch.norm(flow_gt, p=2, dim=1, keepdim=True)+1e-6)
    pred = pred / (torch.norm(pred, p=2, dim=1, keepdim=True)+1e-6)
    dot_prod = flow_gt[:,0]*pred[:,0] + flow_gt[:,1]*pred[:,1]
    angles = torch.arccos(torch.clip(dot_prod, -1.0, 1.0))
    angles = angles.view(-1)[valid.view(-1)]
    angles = torch.mean(angles) * 180 / math.pi

    return angles.item()

def compute_errors(flow_gt, pred, valid):
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = torch.logical_and(valid, mag < MAX_FLOW)

    if torch.sum(valid) < 1:
        return {}

    epe = torch.sum((pred - flow_gt)**2, dim=1).sqrt()
    epe_total = epe.view(-1)[valid.view(-1)]

    mag_mask = (mag < 10)
    epe_s0_10 = epe.view(-1)[valid.view(-1) * mag_mask.view(-1)]

    mag_mask = (mag >= 10) * (mag <= 40)
    epe_s10_40 = epe.view(-1)[valid.view(-1) * mag_mask.view(-1)]

    mag_mask = (mag > 40)
    epe_s40plus = epe.view(-1)[valid.view(-1) * mag_mask.view(-1)]

    metrics = {
        'epe': epe_total.mean().item(),
        'out': compute_out(flow_gt, pred, valid),
        'AE': compute_angular_error(flow_gt, pred, valid),
        '1pe': (epe_total > 1).float().mean().item()*100,
        '2pe': (epe_total > 2).float().mean().item()*100,
        '3pe': (epe_total > 3).float().mean().item()*100,
        '5pe': (epe_total > 5).float().mean().item()*100,
    }
    if epe_s0_10.numel() > 0: metrics['epe_s0_10'] = epe_s0_10.mean().item()
    if epe_s10_40.numel() > 0: metrics['epe_s10_40'] = epe_s10_40.mean().item()
    if epe_s40plus.numel() > 0: metrics['epe_s40plus'] = epe_s40plus.mean().item()

    return metrics

def get_deepest_single_subfolder(path):
    """Recursively finds the deepest directory that contains only one subfolder."""
    current_path = path
    # while os.path.isdir(current_path) and len(os.listdir(current_path)) == 1:
    while True:
        if not os.path.isdir(current_path):
            return None
        list_content = os.listdir(current_path)
        # ignore __MACOSX and .*
        list_content = [item for item in list_content if item != '__MACOSX' and not item.startswith('.')]
        if len(list_content) == 1:
            current_path = os.path.join(current_path, list_content[0])
        else:
            break
    return current_path

if __name__ == '__main__':
    # argsparser
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate BlinkFlow')
    parser.add_argument('--gt_path', type=str, help='path to ground truth flow')
    parser.add_argument('--pred_path', type=str, help='path to predicted flow')
    parser.add_argument('--output_path', type=str, help='path to save cvv/jpg')
    parser.add_argument('--is_zip', action='store_true')
    args = parser.parse_args()
                        
    row_data = []
    metric_name = ['epe', 'out', 'AE', '1pe', '2pe', '3pe', '5pe']

    gt_path = args.gt_path
    pred_path = args.pred_path

    status_code = 'SUCCESS' # one of ['SUCCESS', 'FAILURE']
    error_msg = ''

    if args.is_zip:
        zip_path = args.pred_path
        # new a random hash name
        def generate_random_hash(length=32):
            current_time = str(time.time()).encode('utf-8')
            return hashlib.sha256(current_time).hexdigest()[:length]

        random_hash = generate_random_hash()
        uncompressed_path = os.path.join(args.output_path, random_hash)
        os.system(f'unzip -o {zip_path} -d {uncompressed_path}')
        pred_path = uncompressed_path

    # Replace the existing single-level check with the recursive function
    if os.path.isdir(pred_path):
        pred_path = get_deepest_single_subfolder(pred_path)
        if pred_path is None:
            json_data = {
                'code': 'FAILURE',
                'error_msg': 'Invalid zip file'
            }
            with open(f'{args.output_path}/result.json', 'w') as f:
                json.dump(json_data, f, indent=4)
            exit()
    else:
        json_data = {
            'code': 'FAILURE',
            'error_msg': 'output is not a folder'
        }
        with open(f'{args.output_path}/result.json', 'w') as f:
            json.dump(json_data, f, indent=4)
        exit()
        
    all_pixel_row = [['']+metric_name]

    print(f'Evaluating ...')
    os.system(f'mkdir -p {args.output_path}')

    for scene in os.listdir(gt_path):
        seq_metrics = RunningAverageDict()
        for seq in os.listdir(os.path.join(gt_path, scene)):
            file_list = os.listdir(os.path.join(gt_path, scene, seq))
            for file_name in file_list:
                pred_file_path = os.path.join(pred_path, scene, seq, file_name)
                if not os.path.exists(pred_file_path):
                    status_code = 'FAILURE'
                    error_msg = f'File {scene}/{seq}/{file_name} not found in the zip file'
                    break
                gt_file_path = os.path.join(gt_path, scene, seq, file_name)
                try:
                    pred_data, _ = readFlow(pred_file_path, is_gt=False)
                    gt_data, contains_vis = readFlow(gt_file_path, is_gt=True)
                except Exception as e:
                    status_code = 'FAILURE'
                    error_msg = str(e)
                    break
                pred_flow = pred_data['flow']
                gt_flow = gt_data['flow'][:, :2]

                # the valid mask means non-occluded pixels, we eval on all pixels
                valid = torch.ones([gt_flow.shape[0]], dtype=torch.bool)
                error_dict = compute_errors(torch.from_numpy(gt_flow), torch.from_numpy(pred_flow), valid)
                all_epe = error_dict['epe'] if 'epe' in error_dict else 0
                all_out = error_dict['out'] if 'out' in error_dict else 0
                seq_metrics.update(error_dict)

                if contains_vis:
                    pred_flow = np.reshape(pred_flow, [480, 640, 2])
                    gt_flow = np.reshape(gt_flow, [480, 640, 2])
                    error_map = visualize_error_map(pred_flow, gt_flow)
                    flow_vis = flow_to_image(pred_flow)
                    os.system(f'mkdir -p {args.output_path}/{scene}_{seq}_{file_name[:-4]}')
                    Image.fromarray(gt_data['rgb']).save(f'{args.output_path}/{scene}_{seq}_{file_name[:-4]}/rgb.png')
                    Image.fromarray(gt_data['event']).save(f'{args.output_path}/{scene}_{seq}_{file_name[:-4]}/event.png')
                    Image.fromarray(flow_vis).save(f'{args.output_path}/{scene}_{seq}_{file_name[:-4]}/flow.png')
                    Image.fromarray(error_map).save(f'{args.output_path}/{scene}_{seq}_{file_name[:-4]}/error.png')
                    with open(f'{args.output_path}/{scene}_{seq}_{file_name[:-4]}/err.csv', 'w', newline='') as csvfile:
                        err_data = [['EPE-All', 'Out-All'],
                                    [f'{all_epe:.3f}', f'{all_out:.3f}']]
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerows(err_data)

            if status_code == 'FAILURE':
                break

        if status_code == 'FAILURE':
            break

        seq_metrics = {k: round(v, 3) for k, v in seq_metrics.get_value().items()}

        all_pixel_row.append([scene])
        for metric in metric_name:
            all_pixel_row[-1].append(seq_metrics[metric])

    # write json
    json_data = {
        'code': status_code,
        'error_msg': error_msg
    }
    with open(f'{args.output_path}/result.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    if status_code == 'FAILURE':
        exit()

    # Writing to csv file
    with open(f'{args.output_path}/all.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(all_pixel_row)

    avg_row = [[''] + metric_name]
    avg_row += [[''] + [0] * len(metric_name)]
    for i in range(len(metric_name)):
        avg_metric = (all_pixel_row[1][i+1] + all_pixel_row[2][i+1] + all_pixel_row[3][i+1]) / 3
        avg_row[1][i+1] = avg_metric
       
    with open(f'{args.output_path}/avg.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(avg_row)
