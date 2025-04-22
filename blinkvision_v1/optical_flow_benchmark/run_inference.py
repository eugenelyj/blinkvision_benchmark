import argparse
from datetime import datetime as dt
import math
import os
import sys
from pathlib import Path
import cv2
import imageio
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

# Local imports
# sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('blinkvision_v1')
from pytorch_dataloader.test_optical_flow_dataset import collate_event, BlinkVisionFlow, event2voxel

TAG_VALUE = 2502001 # full data

import sys
sys.path.append('core')

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



def load_model():
    # TODO: add your model here
    return None



def writeFlow(output_file, flow):
    # flow shape: (H, W, 2), in (u, v)
    flow = flow[..., :2]
    height, width = flow.shape[:2]
    with open(output_file, 'wb') as f:
        np.array(TAG_VALUE).astype(np.int32).tofile(f)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        flow.reshape(-1).tofile(f)

def run_frame_model():
    # TODO: add your data path here
    data_path = ''
    sample_map_path = ''
    config = {
        'event': False,
        'clean': True,
        'final': True,
    }

    # TODO: add your model here
    model = load_model()
    save_path = f'bv_flow_v2'

    test_set = BlinkVisionFlow(data_path=data_path, sample_map_path=sample_map_path, config=config)
    test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate_event,
                            drop_last=False)

    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        clean_img1 = batch['clean_img1'].cuda()
        clean_img2 = batch['clean_img2'].cuda()
        final_img1 = batch['final_img1'].cuda()
        final_img2 = batch['final_img2'].cuda()

        scene = batch['scene'][0]
        seq = batch['seq'][0]
        start_idx = batch['start_idx'][0]
        end_idx = batch['end_idx'][0]   
        os.makedirs(f'{save_path}/clean/{scene}/{seq}', exist_ok=True)
        os.makedirs(f'{save_path}/final/{scene}/{seq}', exist_ok=True)

        with torch.no_grad():
            # TODO: checkout to your model's input and output
            _, clean_flow = model(clean_img1, clean_img2, iters=20, test_mode=True)
            _, final_flow = model(final_img1, final_img2, iters=20, test_mode=True)
        
        clean_flow = clean_flow[0].permute(1,2,0).cpu().numpy()
        final_flow = final_flow[0].permute(1,2,0).cpu().numpy()

        clean_write_path = f'{save_path}/clean/{scene}/{seq}/{start_idx:06d}_{end_idx:06d}.flo'
        final_write_path = f'{save_path}/final/{scene}/{seq}/{start_idx:06d}_{end_idx:06d}.flo'
        writeFlow(clean_write_path, clean_flow)
        writeFlow(final_write_path, final_flow)

    print('Saved to', save_path)



def run_event_model():
    # TODO: add your data path here
    data_path = ''
    sample_map_path = ''
    config = {
        'event': True,
    }

    # TODO: add your model here
    model = load_model()
    save_path = f'bv_flow_v2'

    test_set = BlinkVisionFlow(data_path=data_path, sample_map_path=sample_map_path, config=config)
    test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate_event,
                            drop_last=False)

    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        # TODO: checkout to your event representation
        image1, image2 = event2voxel(batch, num_bins=15, height=480, width=640)
        image1, image2 = image1.cuda(), image2.cuda()
        if image1.ndim == 3:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)

        with torch.no_grad():
            # TODO: checkout to your model's input and output
            tup, flow_predictions = model(image1, image2)
        
        flow = flow_predictions[-1][0].cpu()
        flow = flow.permute(1, 2, 0).numpy()

        scene = batch['scene'][0]
        seq = batch['seq'][0]
        start_idx = batch['start_idx'][0]
        end_idx = batch['end_idx'][0]   
        os.makedirs(f'{save_path}/event/{scene}/{seq}', exist_ok=True)
        write_path = f'{save_path}/event/{scene}/{seq}/{start_idx:06d}_{end_idx:06d}.flo'
        writeFlow(write_path, flow)

    print('Saved to', save_path)



if __name__ == '__main__':
    run_frame_model()
    run_event_model()

