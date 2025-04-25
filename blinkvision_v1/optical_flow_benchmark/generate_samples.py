import os
import cv2
import h5py
import yaml
import random
import tqdm
import numpy as np
from pathlib import Path
from event_viz import event_to_rgb
import hashlib

TAG_VALUE = 2502001 # full data
TAG_VALUE_2 = 2502002 # sampled data

vis_list = []
this_file_path = os.path.abspath(__file__)
this_file_dir = os.path.dirname(this_file_path)
vis_list_path = os.path.join(this_file_dir, 'vis_list.txt')
if os.path.exists(vis_list_path):
    with open(vis_list_path, 'r') as f:
        for line in f:
            vis_list.append(line.strip())

def generate_sample_map(width, height, grid_size=4):
    """Generate a random sample map for a given size using efficient grid sampling."""
    sample_map = np.zeros((height, width), dtype=bool)
    
    # Calculate grid dimensions
    grid_h = (height + grid_size - 1) // grid_size  # Ceiling division for edge cases
    grid_w = (width + grid_size - 1) // grid_size
    
    # Generate random offsets for all grid cells at once
    offsets_x = np.random.randint(0, grid_size, size=(grid_h, grid_w))
    offsets_y = np.random.randint(0, grid_size, size=(grid_h, grid_w))
    
    # Create coordinate meshgrid for grid cells
    grid_y, grid_x = np.meshgrid(
        np.arange(grid_h) * grid_size,
        np.arange(grid_w) * grid_size,
        indexing='ij'
    )
    
    # Add random offsets and clip to image boundaries
    sample_y = np.clip(grid_y + offsets_y, 0, height - 1)
    sample_x = np.clip(grid_x + offsets_x, 0, width - 1)
    
    # Set the sampled points
    sample_map[sample_y, sample_x] = True
    
    return sample_map

def write_ground_truth(output_file, flow_data, additional_data):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    height, width = flow_data.shape[:2]

    with open(output_file, 'wb') as f:
        np.array(TAG_VALUE).astype(np.int32).tofile(f)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        flow_data.reshape(-1).tofile(f)
        if additional_data:
            additional_data['clean'].reshape(-1).tofile(f)
            additional_data['final'].reshape(-1).tofile(f)
            additional_data['event'].reshape(-1).tofile(f)


def write_sample_map(output_file, sample_map):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Pack boolean array into bits and save as binary file
    packed_array = np.packbits(sample_map)
    packed_array.tofile(output_file)

def load_rgb_img(rgb_path):
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def process_directory(mapping_file, input_dir, output_dir):

    seed = int(hashlib.sha256(str(TAG_VALUE).encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    input_path = Path(input_dir)

    output_path = Path(output_dir)
    sample_maps_dir = output_path / 'sample_maps'
    ground_truth_dir = output_path / 'ground_truth'
    
    # Create output directory if it doesn't exist
    sample_maps_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    seq_list = []
    with open(mapping_file, 'r') as f:
        lines = f.readlines()
        data = [line.strip('\n').split(', ') for line in lines]
        for (scene_pattern_a, scene_pattern_b) in data:
            category, seq = scene_pattern_b.rsplit('_', 1)
            if os.path.exists(os.path.join(input_path, scene_pattern_a)):
                seq_list.append(scene_pattern_a)
            else:
                seq_list.append(f'{category}/{seq}')


    data_list = []
    global vis_list
    if len(vis_list) == 0:
        generate_vis_list = True
    else:
        generate_vis_list = False

    for sub_path in seq_list:
        flow_dir = os.path.join(input_path, sub_path.replace('outdoor/', 'outdoor_gt/'), 'forward_flow_custom_stride')
        clean_dir = os.path.join(input_path, sub_path, 'clean_uint8')
        event_path = os.path.join(input_path, sub_path.replace('outdoor/', 'outdoor_event/'), 'events_left/events.h5')
        # config_path = os.path.join(input_path, sub_path, 'config.yaml')
        candidate_list = []
        for filename in os.listdir(flow_dir):
            if filename.endswith('.npz'):
                basename = filename.replace('.npz', '')
                image1_index, image2_index = basename.split('_')
                if int(image1_index) % 10 != 0:
                    continue
                flow_path = os.path.join(flow_dir, filename)
                image_path = os.path.join(clean_dir, f'{image1_index}.png')
                # check if all path exists
                if os.path.exists(flow_path) and os.path.exists(image_path) and os.path.exists(event_path):
                    data_list.append([flow_path, image1_index, image2_index, image_path, event_path, None])
                    candidate_list.append(flow_path)
                else:
                    if not os.path.exists(flow_path):
                        print(f'Missing files: {flow_path}')
                    if not os.path.exists(image_path):
                        print(f'Missing files: {image_path}')
                    if not os.path.exists(event_path):
                        print(f'Missing files: {event_path}')
        
        # random sample one from candidate_list
        if generate_vis_list and len(candidate_list) > 0:
            vis_list.append(candidate_list[np.random.randint(0, len(candidate_list))])
    
    if generate_vis_list:
        # random sample 50 from vis_list
        vis_list = random.sample(vis_list, 50)
        with open('vis.txt', 'w') as f:
            for vis in vis_list:
                vis = vis.replace(str(input_path)+'/', '').replace('/forward_flow_custom_stride', '').replace('.npz', '.flo')
                f.write(f'{vis}\n')

    for data in tqdm.tqdm(data_list, total=len(data_list), desc="Processing samples"):
        flow_path, image1_index, image2_index, image_path, event_path, config_path = data
        basename = f'{image1_index}_{image2_index}'
        rel_path = os.path.relpath(os.path.dirname(flow_path), input_path).replace('forward_flow_custom_stride', '')
        # with open(config_path, 'r') as f:
        #     config = yaml.safe_load(f)  # Use safe_load instead of load for security
        # fps = config['fps']  # Get fps from config
        fps = 20
        
        # Convert frame indices to timestamps in milliseconds
        start_ts = float(image1_index) * (1000.0 / fps)  # Convert to ms
        end_ts = float(image2_index) * (1000.0 / fps)    # Convert to ms

        # read data
        flow_raw_data = np.load(flow_path)['forward_flow'].astype(np.float32)
        flow_data, valid_mask = flow_raw_data[..., :2], flow_raw_data[..., 2]
        height, width = flow_data.shape[:2]

        ground_truth_rel_path = os.path.join(rel_path, f'{basename}.flo')
        additional_data = None
        if ground_truth_rel_path in vis_list or ground_truth_rel_path.replace('outdoor_gt', 'outdoor') in vis_list:
            # read rgb and event data
            event_data = h5py.File(event_path, 'r')
            event_stream = event_data['events']
            ms_to_idx = event_data['ms_to_idx']
            start_idx, end_idx = ms_to_idx[int(start_ts)], ms_to_idx[int(end_ts)]
            p = event_stream['p'][start_idx:end_idx]
            x = event_stream['x'][start_idx:end_idx]
            y = event_stream['y'][start_idx:end_idx]
            t = event_stream['t'][start_idx:end_idx]
            event_img = event_to_rgb(_x=x, _y=y, pol=p, t=t, H=height, W=width)
            clean_image = load_rgb_img(image_path)
            final_image = load_rgb_img(image_path.replace('clean_uint8', 'final_uint8'))
            additional_data = {}
            additional_data['clean'] = clean_image.astype(np.uint8)
            additional_data['final'] = final_image.astype(np.uint8)
            additional_data['event'] = event_img.astype(np.uint8)

            sample_map = np.ones((height, width), dtype=bool)
        else:
            # Generate sample map
            sample_map = generate_sample_map(width, height, grid_size=6)

        # set invalid flow to False
        sample_map = np.where(valid_mask == 0, False, sample_map)
        
        # Create output directory structure
        sample_file = sample_maps_dir / rel_path / f'{basename}.bin'
        write_sample_map(sample_file, sample_map)

        ground_truth_file = ground_truth_dir / rel_path / f'{basename}.flo'
        write_ground_truth(ground_truth_file, flow_data, additional_data)

if __name__ == "__main__":
    # This script is used to generate:
    # 1. sample maps for flow data
    # 2. pack ground truth data into a binary file
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', help='Path to the mapping file', default='blinkvision_v1/pytorch_dataloader/mapping_test.txt')
    parser.add_argument('--input_dir', help='Input directory containing .flo files', required=True)
    parser.add_argument('--output_dir', help='Output directory for sample maps', required=True)
    
    args = parser.parse_args()
    
    process_directory(args.mapping_file, args.input_dir, args.output_dir) 
