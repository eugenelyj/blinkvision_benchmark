from huggingface_hub import HfApi, login
import os
import argparse

def login_to_hf(token):
    """Login to Hugging Face using token"""
    login(token=token)


def get_seq_list(repo_id):
    api = HfApi()
    repo_contents = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    # Extract two-level paths from all files that have at least two levels
    two_level_paths = set()
    for path in repo_contents:
        parts = path.split('/')
        if len(parts) >= 2:  # Has at least two levels
            two_level_path = f"{parts[0]}/{parts[1]}"
            two_level_paths.add(two_level_path)
    
    # Convert set back to sorted list
    two_level_paths = sorted(list(two_level_paths))
    
    return two_level_paths

def download_from_hf(repo_id, seq_list, key_list, local_path):
    api = HfApi()
    
    # Download all matching files
    for seq in seq_list:
        for key in key_list:
            try:
                os.makedirs(os.path.join(local_path, seq), exist_ok=True)
                local_file_path = os.path.join(local_path, seq, key)
                api.hf_hub_download(
                    repo_id=repo_id,
                    filename=f'{seq}/{key}',
                    repo_type="dataset",
                    local_dir=local_path,
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded {seq}/{key} to {local_file_path}")
            except Exception as e:
                print(f"Error downloading {seq}/{key}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download files from Hugging Face repository")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--modality", type=str, required=True, help="rgb or event or all")
    parser.add_argument("--local_path", type=str, required=True, help="Local path to save the downloaded content")

    args = parser.parse_args()

    if not args.modality in ['rgb', 'event', 'all']:
        raise ValueError("modality must be rgb or event or all")

    key_list = []
    if args.modality == 'rgb':
        key_list = ['clean_uint8.zip', 'final_uint8.zip']
    elif args.modality == 'event':
        key_list = ['events_left']
    elif args.modality == 'all':
        key_list = ['clean_uint8.zip', 'final_uint8.zip', 'events_left']

    repo_id = 'BlinkVision/BlinkVision_Test'

    # Login to Hugging Face
    login_to_hf(args.token)

    # get the seq
    seq_list = get_seq_list(repo_id)

    # Download the content
    download_from_hf(repo_id, seq_list, key_list, args.local_path)

if __name__ == "__main__":
    main()
