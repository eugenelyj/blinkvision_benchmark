#!/bin/bash

REPO_ID="BlinkVision/BlinkVision_train"

# Login to huggingface
huggingface-cli login

# Download the dataset
huggingface-cli download $REPO_ID --repo-type dataset --local-dir=./data

echo "Dataset downloaded successfully to ./data directory"
