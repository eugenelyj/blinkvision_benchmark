import numpy as np
import os
import argparse

TAG_VALUE_2 = 2502002  # sampled data

def load_sample_map(file_path, height, width):
    """Load a sample map from a binary file.
    
    Args:
        file_path: Path to the binary sample map file
        height: Width of the original image
        width: Height of the original image
        
    Returns:
        numpy.ndarray: Boolean array of shape (height, width)
    """
    # Read packed bits from binary file
    packed_array = np.fromfile(file_path, dtype=np.uint8)
    
    # Unpack bits back into boolean array
    sample_map = np.unpackbits(packed_array).astype(bool)
    
    # Reshape and trim to original dimensions
    return sample_map.reshape(height, width)

def check_flow_file(file_path, sample_map_path):
    """
    Check if a flow file is valid according to the expected format.
    """
    try:
        with open(file_path, 'rb') as f:
            magic = np.fromfile(f, np.int32, count=1)[0]
            if magic != TAG_VALUE_2:
                return False, f'Invalid magic number in file {file_path}'

            contains_vis = np.fromfile(f, np.int32, count=1)[0]
            num_samples = np.fromfile(f, np.int32, count=1)[0]
            
            if not contains_vis:
                # Load the sample map from the corresponding .bin file
                sample_map = load_sample_map(sample_map_path, height=480, width=640)  # Assuming standard dimensions
                expected_samples = np.sum(sample_map)
                
                if num_samples != expected_samples:
                    return False, f'Number of samples ({num_samples}) does not match sample map sum ({expected_samples}) in file {file_path}'
            
            # Check if file has enough data for flow values
            data = np.fromfile(f, np.float32, count=2*int(num_samples))
            if len(data) != 2*int(num_samples):
                return False, f'Incomplete flow data in file {file_path}'

            return True, ''
    except Exception as e:
        return False, f'Error reading file {file_path}: {str(e)}'

def check_submission(sample_path, submission_path):
    """
    Verify submission structure matches the sample directory structure
    and validate flow files.
    """
    if not os.path.exists(submission_path):
        return False, "Submission path does not exist"

    # Check directory structure
    for scene in os.listdir(sample_path):
        scene_path = os.path.join(sample_path, scene)
        if not os.path.isdir(scene_path):
            continue

        submission_scene_path = os.path.join(submission_path, scene)
        if not os.path.exists(submission_scene_path):
            return False, f"Missing scene directory: {scene}"

        for seq in os.listdir(scene_path):
            seq_path = os.path.join(scene_path, seq)
            if not os.path.isdir(seq_path):
                continue

            submission_seq_path = os.path.join(submission_scene_path, seq)
            if not os.path.exists(submission_seq_path):
                return False, f"Missing sequence directory: {scene}/{seq}"

            # Check flow files
            for file_name in os.listdir(seq_path):
                if not file_name.endswith('.bin'):
                    continue

                submit_filename = file_name.replace('.bin', '.flo')
                submission_file = os.path.join(submission_seq_path, submit_filename)
                if not os.path.exists(submission_file):
                    return False, f"Missing flow file: {scene}/{seq}/{submit_filename}"

                # Get corresponding sample map path
                sample_map_file = os.path.join(seq_path, file_name)
                
                # Validate flow file format
                is_valid, error_msg = check_flow_file(submission_file, sample_map_file)
                if not is_valid:
                    return False, error_msg

    return True, "Submission check passed"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check BlinkFlow submission')
    parser.add_argument('--sample_path', type=str, required=True, 
                        help='path to sample map directory structure')
    parser.add_argument('--submission_path', type=str, required=True, 
                        help='path to submission directory')
    args = parser.parse_args()

    is_valid, message = check_submission(args.sample_path, args.submission_path)
    if is_valid:
        print("Success:", message)
        exit(0)
    else:
        print("Error:", message)
        exit(1)
