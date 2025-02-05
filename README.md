# BlinkFlow

The related scripts are in the `flow_v1` folder.

This tree structure shows:
1. The main `flow_v1` directory
2. A `benchmark` subdirectory containing:
   - `run_inference.py`: Runs inference on the benchmark dataset
   - `sample_data.py`: Samples the infernece results before submission (also used for ground truth)
   - `check_submission.py`: Validates submission files against sample maps
   - `generate_samples.py`: Generates samples maps
   - `evaluate.py`: Evaluates the submission files on the benchmark website
   - `event_viz.py`: Visualizes the events
3. A `pytorch_dataloader` subdirectory containing dataset loader scripts
