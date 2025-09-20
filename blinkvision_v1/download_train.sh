#!/bin/bash

REPO_ID="BlinkVision/BlinkVision_train"

# Login to huggingface
huggingface-cli login

# Download the dataset
huggingface-cli download $REPO_ID --repo-type dataset --local-dir=./data

# Once you have downloaded, you need to concatenate the downloaded file chunks and decompress them. For example:
# cat outdoor_train_event_part_a* > outdoor_train_event.tar.gz
# tar -xvzf outdoor_train_event.tar.gz

echo "Dataset downloaded successfully to ./data directory"
