import os
import tqdm
import numpy as np
from pathlib import Path
import struct

TAG_VALUE = 25020223

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

def read_flo_file(file_path, relative_path, gt_input):
    """Read the flow file format used in the C++ code."""
    with open(file_path, 'rb') as f:
        # Read header
        tag = np.fromfile(f, dtype=np.int32, count=1)[0]
        if tag != TAG_VALUE:
            raise ValueError(f"Invalid tag {tag} in flow file: {file_path}, expected {TAG_VALUE}")
        
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read flow data
        flow_data = np.fromfile(f, dtype=np.float32, count=width * height * 2)
        flow_data = flow_data.reshape(height, width, 2)
        
        data = {
            'flow': flow_data,
        }

        contains_vis = False
        if relative_path in vis_list:
            contains_vis = True

            if gt_input:
                rgb_data = np.fromfile(f, dtype=np.uint8, count=width * height * 3)
                event_data = np.fromfile(f, dtype=np.uint8, count=width * height * 3)
                
                data.update({
                    'rgb': rgb_data,
                    'event': event_data
                })

        data['contains_vis'] = contains_vis
            
        return width, height, data

def sample_and_save_flow(flow_path, sample_map_path, output_path, rel_path, gt_input):
    """Sample flow data according to the sample map and save it."""
    # Read flow data
    width, height, data = read_flo_file(flow_path, rel_path, gt_input)
    flow_data = data['flow']
    flow_data = flow_data.reshape(height, width, 2)

    contains_vis = data['contains_vis']
    if contains_vis:
        sampled_flow = flow_data

        if gt_input:
            rgb_data = data['rgb']
            event_data = data['event']
    else:
        # Load sample map
        sample_map = load_sample_map(sample_map_path, height, width)
        
        # Sample the flow data
        sampled_flow = flow_data[sample_map].reshape(-1)

    # Prepare output data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write sampled data in the same format as C++ code
    with open(output_file, 'wb') as f:
        # Write header
        np.array(TAG_VALUE).astype(np.int32).tofile(f)
        np.array(contains_vis).astype(np.int32).tofile(f)
        np.array(len(sampled_flow)).astype(np.int32).tofile(f)
        
        # Write sampled flow data
        sampled_flow.tofile(f)

        if contains_vis and gt_input:
            rgb_data.tofile(f)
            event_data.tofile(f)

def load_sample_map(file_path, height, width):
    """Load a sample map from a binary file.
    
    Args:
        file_path: Path to the binary sample map file
        height: Height of the original image
        width: Width of the original image
        
    Returns:
        numpy.ndarray: Boolean array of shape (height, width)
    """
    # Read packed bits from binary file
    packed_array = np.fromfile(file_path, dtype=np.uint8)
    
    # Unpack bits back into boolean array
    sample_map = np.unpackbits(packed_array).astype(bool)

    assert len(sample_map) == height * width
    
    # Reshape and trim to original dimensions
    # (np.unpackbits might return more bits than needed due to byte alignment)
    return sample_map.reshape(height, width)

def process_directory(flow_dir, sample_maps_dir, output_dir, gt_input):
    flow_path = Path(flow_dir)
    sample_maps_path = Path(sample_maps_dir)
    output_path = Path(output_dir)
    
    total_files = sum(1 for _ in flow_path.rglob('*.flo'))
    for flo_path in tqdm.tqdm(flow_path.rglob('*.flo'), total=total_files):
        # Get relative path
        rel_path = flo_path.relative_to(flow_path)
        
        # Construct paths
        sample_map_path = sample_maps_path / rel_path.with_suffix('.bin')
        output_file = output_path / rel_path
        
        # Process the file
        sample_and_save_flow(flo_path, sample_map_path, output_file, str(rel_path), gt_input)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample flow data using pre-generated sample maps')
    parser.add_argument('flow_dir', help='Directory containing flow files')
    parser.add_argument('sample_maps_dir', help='Directory containing sample maps')
    parser.add_argument('output_dir', help='Output directory for sampled data')
    parser.add_argument('--gt_input', action='store_true', help='Use ground truth input')
    
    args = parser.parse_args()
    
    process_directory(args.flow_dir, args.sample_maps_dir, args.output_dir, args.gt_input) 
