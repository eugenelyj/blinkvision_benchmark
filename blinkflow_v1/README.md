
# Download

See `download_train.sh` and `download_test.sh`.

# Submission

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



# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{blinkflow_iros2023,
  title={BlinkFlow: A Dataset to Push the Limits of Event-based Optical Flow Estimation},
  author={Yijin Li, Zhaoyang Huang, Shuo Chen, Xiaoyu Shi, Hongsheng Li, Hujun Bao, Zhaopeng Cui, Guofeng Zhang},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  month = {October},
  year = {2023},
}

@inproceedings{blinkvision_eccv2024,
  title={BlinkVision: A Benchmark for Optical Flow, Scene Flow and Point Tracking Estimation using RGB Frames and Events},
  author={Yijin Li, Yichen Shen, Zhaoyang Huang, Shuo Chen, Weikang Bian, Xiaoyu Shi, Fu-Yun Wang, Keqiang Sun, Hujun Bao, Zhaopeng Cui, Guofeng Zhang, Hongsheng Li},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

