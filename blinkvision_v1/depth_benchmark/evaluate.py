import numpy as np
import torch
import os
import math
import csv
import cv2
import json
from PIL import Image
import hashlib
import time


TAG_VALUE = 2502001 # full data
TAG_VALUE_2 = 2502002 # sampled data


def get_correct_path(path):
    while True:
        if not os.path.isdir(path):
            return None
        # list dir and filter out dir
        list_content = os.listdir(path)
        list_content = [item for item in list_content if os.path.isdir(os.path.join(path, item))]
        # ignore __MACOSX and .*
        list_content = [item for item in list_content if item != '__MACOSX' and not item.startswith('.')]
        if len(list_content) < 1:
            return None
        elif ('clean' in list_content and 'final' in list_content) or 'event' in list_content:
            return path
        elif len(list_content) == 1:
            path = os.path.join(path, list_content[0])
        else:
            return None

def visualize_error_map(depth_pred, depth_gt, sample_map):
    error_map = np.abs(depth_pred - depth_gt)
    error_map[~sample_map] = 0
    percentile_99 = np.percentile(error_map, 99)
    if percentile_99 > 0:
        error_map[error_map > percentile_99] = percentile_99
        error_map = error_map / percentile_99
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

def readDepth(fn, is_gt=False):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.int32, count=1)
        if magic != TAG_VALUE_2:
            raise ValueError('Magic number incorrect. Invalid .bin file')

        contains_vis = np.fromfile(f, np.int32, count=1)
        num_samples = np.fromfile(f, np.int32, count=1)
        data = np.fromfile(f, np.float32, count=int(num_samples))
        data_dict = {
            'depth': np.resize(data, (int(num_samples), 1))
        }

        if contains_vis and is_gt:
            sample_map = np.fromfile(f, np.bool_, count=int(num_samples))
            clean_image = np.fromfile(f, np.uint8, count=3*int(num_samples))
            final_image = np.fromfile(f, np.uint8, count=3*int(num_samples))
            event_image = np.fromfile(f, np.uint8, count=3*int(num_samples))
            data_dict['sample_map'] = np.reshape(sample_map, [480, 640])
            data_dict['clean'] = np.reshape(clean_image, [480, 640, 3])
            data_dict['final'] = np.reshape(final_image, [480, 640, 3])
            data_dict['event'] = np.reshape(event_image, [480, 640, 3])

        return data_dict, bool(contains_vis)


def depth_to_image(depth):
    depth = depth.astype(np.float32)
    max_depth = np.percentile(depth, 99)
    depth[depth > max_depth] = max_depth
    depth = (depth - np.min(depth)) / (max_depth - np.min(depth))
    depth = (depth * 255).astype(np.uint8)
    # colormap
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    return depth

# copied from https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/utils/misc.py
def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    inside_term = np.mean(err ** 2) - np.mean(err) ** 2
    # avoid nan
    if inside_term < 0 and inside_term > -1e-3:
        inside_term = 0
    silog = np.sqrt(inside_term) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return {
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'abs_rel': abs_rel,
        'rmse': rmse,
        'log_10': log_10,
        'rmse_log': rmse_log,
        'silog': silog,
        'sq_rel': sq_rel
    }

if __name__ == '__main__':
    # argsparser
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Depth')
    parser.add_argument('--gt_path', type=str, help='path to ground truth depth')
    parser.add_argument('--pred_path', type=str, help='path to predicted depth')
    parser.add_argument('--output_path', type=str, help='path to save csv/jpg')
    parser.add_argument('--is_zip', action='store_true')
    args = parser.parse_args()

    print(f'Evaluating ...')
    os.system(f'mkdir -p {args.output_path}')
                        
    row_data = []
    metric_name = ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10', 'rmse_log', 'silog', 'sq_rel']

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

    if os.path.isdir(pred_path):
        pred_path = get_correct_path(pred_path)
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
        
    if 'clean' in os.listdir(pred_path) and 'final' in os.listdir(pred_path):
        pattern_list = ['clean', 'final']
    elif 'event' in os.listdir(pred_path):
        pattern_list = ['event']
    else:
        json_data = {
            'code': 'FAILURE',
            'error_msg': 'Invalid zip file'
        }
        with open(f'{args.output_path}/result.json', 'w') as f:
            json.dump(json_data, f, indent=4)
        exit()


    for pattern in pattern_list:
        all_pixel_row = [['']+metric_name]
        os.system(f'mkdir -p {args.output_path}/{pattern}')
        for scene in os.listdir(gt_path):
            seq_metrics = RunningAverageDict()
            for seq in os.listdir(os.path.join(gt_path, scene)):
                file_list = os.listdir(os.path.join(gt_path, scene, seq))
                for file_name in file_list:
                    pred_file_path = os.path.join(pred_path, pattern, scene, seq, file_name)
                    if not os.path.exists(pred_file_path):
                        status_code = 'FAILURE'
                        error_msg = f'File {scene}/{seq}/{file_name} not found in the zip file'
                        break
                    gt_file_path = os.path.join(gt_path, scene, seq, file_name)
                    try:
                        pred_data, _ = readDepth(pred_file_path, is_gt=False)
                        gt_data, contains_vis = readDepth(gt_file_path, is_gt=True)
                    except Exception as e:
                        status_code = 'FAILURE'
                        error_msg = str(e)
                        break
                    pred_depth = pred_data['depth']
                    gt_depth = gt_data['depth']

                    # align the depth by median
                    gt_depth_median = np.median(gt_depth)
                    pred_depth_median = np.median(pred_depth)
                    pred_depth = pred_depth / pred_depth_median * gt_depth_median

                    if 'sample_map' in gt_data:
                        sample_map = gt_data['sample_map']
                        gt_larger_than_0 = gt_depth.reshape([480, 640]) > 1e-2
                        valid = sample_map.astype(np.bool_)
                        valid = np.logical_and(valid, gt_larger_than_0)
                        valid_flat = valid.reshape(-1)
                    else:
                        valid = np.ones_like(gt_depth, dtype=np.bool_)
                        gt_larger_than_0 = gt_depth > 1e-2
                        valid = np.logical_and(valid, gt_larger_than_0)
                        valid_flat = valid.reshape(-1)
                    # to avoid log(0)
                    pred_depth[pred_depth < 1e-6] = 1e-6
                    # too close
                    if np.sum(valid_flat) < 480 * 640 / 2:
                        continue
                    error_dict = compute_errors(gt_depth[valid_flat], pred_depth[valid_flat])
                    all_a1 = error_dict['a1'] if 'a1' in error_dict else 0
                    all_rmse = error_dict['rmse'] if 'rmse' in error_dict else 0

                    frame_index = file_name.split('.')[0]
                    renamed_error_dict = {}
                    for key, value in error_dict.items():
                        renamed_key = f'{key}'
                        renamed_error_dict[renamed_key] = value

                    seq_metrics.update(renamed_error_dict)

                    if contains_vis:
                        pred_depth = np.reshape(pred_depth, [480, 640])
                        gt_depth = np.reshape(gt_depth, [480, 640])
                        error_map = visualize_error_map(pred_depth, gt_depth, valid)
                        depth_vis = depth_to_image(pred_depth)
                        valid_mask = valid.astype(np.uint8) * 255
                        folder_name = f'{args.output_path}/{pattern}/{scene}_{seq}_{file_name[:-4]}'
                        os.system(f'mkdir -p {folder_name}')
                        if pattern == 'clean' or pattern == 'event':
                            Image.fromarray(gt_data['clean']).save(f'{folder_name}/rgb.png')
                        else:
                            Image.fromarray(gt_data['final']).save(f'{folder_name}/rgb.png')
                        Image.fromarray(valid_mask).save(f'{folder_name}/valid_mask.png')
                        Image.fromarray(gt_data['event']).save(f'{folder_name}/event.png')
                        Image.fromarray(depth_vis).save(f'{folder_name}/depth.png')
                        Image.fromarray(error_map).save(f'{folder_name}/error.png')
                        with open(f'{folder_name}/err.csv', 'w', newline='') as csvfile:
                            err_data = [['A1-All', 'RMSE-All'],
                                        [f'{all_a1:.3f}', f'{all_rmse:.3f}']]
                            csvwriter = csv.writer(csvfile)
                            csvwriter.writerows(err_data)

                if status_code == 'FAILURE':
                    # write json
                    json_data = {
                        'code': status_code,
                        'error_msg': error_msg
                    }
                    with open(f'{args.output_path}/result.json', 'w') as f:
                        json.dump(json_data, f, indent=4)
                    exit()

            seq_metrics = {k: round(v, 3) for k, v in seq_metrics.get_value().items()}

            all_pixel_row.append([scene])
            for metric in metric_name:
                all_pixel_row[-1].append(seq_metrics[metric])

        # Writing to csv file
        with open(f'{args.output_path}/{pattern}/all.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(all_pixel_row)

        avg_row = [[''] + metric_name]
        avg_row += [[''] + [0] * len(metric_name)]
        for i in range(len(metric_name)):
            avg_metric = 0
            for j in range(1, len(all_pixel_row)):
                avg_metric += all_pixel_row[j][i+1]
            avg_metric /= len(all_pixel_row) - 1
            avg_row[1][i+1] = avg_metric
        
        with open(f'{args.output_path}/{pattern}/avg.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(avg_row)

    if len(pattern_list) == 1 and pattern_list[0] == 'event':
        os.system(f'cp -r {args.output_path}/event/ {args.output_path}/clean/')
        os.system(f'cp -r {args.output_path}/event/ {args.output_path}/final/')
        os.system(f'rm -rf {args.output_path}/event')

    # write json
    json_data = {
        'code': status_code,
        'error_msg': error_msg
    }
    with open(f'{args.output_path}/result.json', 'w') as f:
        json.dump(json_data, f, indent=4)

