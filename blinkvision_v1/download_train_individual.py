from huggingface_hub import HfApi, login
import os
import argparse

def login_to_hf(token):
    """Login to Hugging Face using token"""
    login(token=token)


def get_seq_list(all_seq, seq_name, repo_id):
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
    
    if seq_name and not seq_name in two_level_paths:
        raise ValueError(f"Sequence {seq_name} not found in repository {repo_id}")

    seq_list = two_level_paths if all_seq else [seq_name]
    
    return seq_list

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
    parser.add_argument("--seq_name", type=str, required=False, help="Sequence name")
    parser.add_argument("--key", type=str, required=False, help="Key")
    parser.add_argument("--all_seq", action="store_true", help="Download all sequences")
    parser.add_argument("--all_keys", action="store_true", help="Download all keys")
    parser.add_argument("--local_path", type=str, required=True, help="Local path to save the downloaded content")

    args = parser.parse_args()

    if not args.all_seq and not args.seq_name:
        raise ValueError("Either --all_seq or --seq_name must be provided")

    if not args.all_keys and not args.key:
        raise ValueError("Either --all_keys or --key must be provided")

    repo_id = 'BlinkVision/BlinkVision_Train_PerSeq'
    all_keys = ['clean_uint8', 'final_uint8', 'depth', 'normals', 'forward_flow', 'backward_flow', 'poses']

    # Login to Hugging Face
    login_to_hf(args.token)

    # get the seq
    seq_list = get_seq_list(args.all_seq, args.seq_name, repo_id)

    if args.key and not args.key in all_keys:
        raise ValueError(f"Key {args.key} not found in repository {repo_id}")
    key_list = all_keys if args.all_keys else [args.key]
    key_list = [f'{key}.npz' if key == 'poses' else f'{key}.zip' for key in key_list]

    # Download the content
    download_from_hf(repo_id, seq_list, key_list, args.local_path)

if __name__ == "__main__":
    main()
