# BlinkFlow

The related scripts are in the `blinkflow_v1` folder.

To submit your results, you need the following steps:
1. Run your results by referring to the `run_inference.py` script.
2. Process the results by running the `sample_data.py` script.
3. Check the submission by running the `check_submission.py` script.
4. Upload the results to the benchmark website.

The files under the `blinkflow_v1` directory:
1. A `benchmark` subdirectory containing:
   - `run_inference.py`: Runs inference on the benchmark dataset
   - `sample_data.py`: Samples the infernece results before submission (also used for ground truth)
   - `check_submission.py`: Validates submission files against sample maps
   - `generate_samples.py`: Generates samples maps
   - `evaluate.py`: Evaluates the submission files on the benchmark website
   - `event_viz.py`: Visualizes the events
2. A `pytorch_dataloader` subdirectory containing dataset loader scripts

# BlinkVision

The related scripts are in the `blinkvision_v1` folder.