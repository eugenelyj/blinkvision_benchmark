import os
import tqdm
import numpy as np
from pathlib import Path
import shutil
import struct

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
        
        data = {
            'flow': flow_data,
        }

        contains_vis = False
        flow_relative_path = relative_path.split('.')[0] + '.flo'
        if flow_relative_path in vis_list or flow_relative_path.replace('outdoor_gt', 'outdoor') in vis_list:
            contains_vis = True

            if gt_input:
                clean_data = np.fromfile(f, dtype=np.uint8, count=width * height * 3)
                final_data = np.fromfile(f, dtype=np.uint8, count=width * height * 3)
                event_data = np.fromfile(f, dtype=np.uint8, count=width * height * 3)

                data.update({
                    'clean': clean_data,
                    'final': final_data,
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
    sample_map = load_sample_map(sample_map_path, height, width)
    if data['contains_vis']:
        sampled_flow = flow_data.reshape(-1)
    else:
        sampled_flow = flow_data[sample_map].reshape(-1)

    contains_vis = data['contains_vis']
    if contains_vis and gt_input:
        clean_data = data['clean']
        final_data = data['final']
        event_data = data['event']

    # Prepare output data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write sampled data in the same format as C++ code
    with open(output_file, 'wb') as f:
        # Write header
        np.array(TAG_VALUE_2).astype(np.int32).tofile(f)
        np.array(contains_vis).astype(np.int32).tofile(f)
        num_samples = int(len(sampled_flow) / 2)
        np.array(num_samples).astype(np.int32).tofile(f)
        
        # Write sampled flow data
        sampled_flow.tofile(f)

        if contains_vis and gt_input:
            sample_map.tofile(f)
            clean_data.tofile(f)
            final_data.tofile(f)
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
    pattern_list = []
    if not gt_input:
        # should be clean&final or event folder under flow_dir
        list_content = os.listdir(flow_dir)
        if len(list_content) == 2 and 'clean' in list_content and 'final' in list_content:
            pattern_list = ['clean', 'final']
        elif len(list_content) == 1 and 'event' in list_content:
            pattern_list = ['event']
        else:
            raise ValueError(f"Invalid directory structure: {flow_dir}")
    else:
        pattern_list = ['']
        
    flow_path = Path(flow_dir)
    sample_maps_path = Path(sample_maps_dir)
    output_path = Path(output_dir)
    
    total_files = sum(1 for _ in sample_maps_path.rglob('*.bin'))
    for pattern in pattern_list:
        for sample_map_path in tqdm.tqdm(sample_maps_path.rglob('*.bin'), total=total_files):
            # Get relative path
            rel_path = sample_map_path.relative_to(sample_maps_path)
            
            # Construct paths
            flo_path = flow_path / pattern / rel_path.with_suffix('.flo')
            output_file = output_path / pattern / rel_path.with_suffix('.flo')
            
            # Process the file
            sample_and_save_flow(flo_path, sample_map_path, output_file, str(rel_path), gt_input)

    # if not gt_input, zip the output directory
    if not gt_input:
        print(f'Zipping the output directory: {output_path}')
        zip_path = output_path
        shutil.make_archive(zip_path, 'zip', output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample flow data using pre-generated sample maps')
    parser.add_argument('flow_dir', help='Directory containing flow files')
    parser.add_argument('sample_maps_dir', help='Directory containing sample maps')
    parser.add_argument('output_dir', help='Output directory for sampled data')
    parser.add_argument('--gt_input', action='store_true', help='Use ground truth input')
    
    args = parser.parse_args()
    
    process_directory(args.flow_dir, args.sample_maps_dir, args.output_dir, args.gt_input) 
