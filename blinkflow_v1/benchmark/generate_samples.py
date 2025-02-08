import os
import cv2
import h5py
import tqdm
import numpy as np
from pathlib import Path
from event_viz import event_to_rgb
import hashlib

TAG_VALUE = 2502001 # full data
TAG_VALUE_2 = 2502002 # sampled data

vis_list = [
    "FlyingObjects/000039/700_800.flo",
    "FlyingObjects/000042/200_300.flo",
    "E-Blender/anima/900_950.flo",
    "E-Blender/billiards/450_500.flo",
    "E-Blender/card/3000_3050.flo",
    "E-Blender/die/650_700.flo",
    "E-Blender/street/3700_3750.flo",
    "E-Tartan/20/2200_2250.flo",
    "E-Tartan/17/1050_1100.flo",
    "E-Tartan/14/2700_2750.flo",
    "E-Tartan/12/5100_5150.flo",
    "E-Tartan/01/3200_3250.flo",
]

def generate_sample_map(width, height):
    """Generate a random sample map for a given size using efficient grid sampling."""
    sample_map = np.zeros((height, width), dtype=bool)
    
    # Calculate grid dimensions
    grid_size = 4
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
            additional_data['rgb'].reshape(-1).tofile(f)
            additional_data['event'].reshape(-1).tofile(f)


def write_sample_map(output_file, sample_map):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Pack boolean array into bits and save as binary file
    packed_array = np.packbits(sample_map)
    packed_array.tofile(output_file)

def load_rgb_img(possible_rgb_path):
    if os.path.exists(possible_rgb_path):
        rgb_img = np.load(possible_rgb_path)

        if 'anima' not in possible_rgb_path:
            rgb_img = (rgb_img * 255).astype(np.uint8)

        if 'Blender' in possible_rgb_path:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        return rgb_img
    else:
        try:
            rgb_path = possible_rgb_path.replace('npy', 'png')
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            return rgb_img
        except:
            raise ValueError(f"No RGB image found at {possible_rgb_path} or {rgb_path}")


def process_directory(input_dir, output_dir):

    seed = int(hashlib.sha256(str(TAG_VALUE).encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    input_path = Path(input_dir)

    output_path = Path(output_dir)
    sample_maps_dir = output_path / 'sample_maps'
    ground_truth_dir = output_path / 'ground_truth'
    
    # Create output directory if it doesn't exist
    sample_maps_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # Count total lines first
    total_lines = sum(1 for _ in open(input_path / 'full_test_split.txt', 'r'))
    
    # Then process the file with known total
    with open(input_path / 'full_test_split.txt', 'r') as f:
        for line in tqdm.tqdm(f, total=total_lines, desc="Processing samples"):
            event_path, start_ts, end_ts, flow_path = line.strip().split()
            rel_path = os.path.dirname(flow_path).replace('forward_flow', '')
            event_path = os.path.join(input_path, event_path)
            flow_path = os.path.join(input_path, flow_path)
            basename = f'{start_ts}_{end_ts}'

            # read data
            flow_data = np.load(flow_path).astype(np.float32)[..., :2]
            height, width = flow_data.shape[:2]

            ground_truth_rel_path = os.path.join(rel_path, f'{basename}.flo')
            additional_data = None
            if ground_truth_rel_path in vis_list:
                # read rgb and event data
                event_data = h5py.File(event_path, 'r')
                event_stream = event_data['events']
                ms_to_idx = event_data['ms_to_idx']
                start_idx, end_idx = ms_to_idx[int(start_ts)], ms_to_idx[int(end_ts)]
                p = event_stream['p'][start_idx:end_idx]
                x = event_stream['x'][start_idx:end_idx]
                y = event_stream['y'][start_idx:end_idx]
                event_img = event_to_rgb(_x=x, _y=y, pol=p, H=height, W=width)
                rgb_path = flow_path.replace('forward_flow', 'hdr')
                rgb_img = load_rgb_img(rgb_path)
                additional_data = {}
                additional_data['rgb'] = rgb_img.astype(np.uint8)
                additional_data['event'] = event_img.astype(np.uint8)

                sample_map = np.ones((height, width), dtype=bool)
            else:
                # Generate sample map
                sample_map = generate_sample_map(width, height)
            
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
    parser.add_argument('input_dir', help='Input directory containing .flo files')
    parser.add_argument('output_dir', help='Output directory for sample maps')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir) 
