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
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_dataloader.dataset import collate_event, BlinkFlow, event2voxel

TAG_VALUE = 2502001 # full data

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

if __name__ == '__main__':

    # test_method = 'ours'
    for test_method in ['ours']:
        # TODO: add your data path here
        data_path = ''

        # TODO: add your model here
        model = load_model()
        save_path = f'eval_results'

        test_set = BlinkFlow(data_path=data_path, split='test')
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
                # TODO: checkout to your model's input
                tup, flow_predictions = model(image1, image2)
            
            flow = flow_predictions[-1][0].cpu()
            flow = flow.permute(1, 2, 0).numpy()

            scene = batch['scene'][0]
            seq = batch['seq'][0]
            time_window = batch['time_window'][0]
            os.makedirs(f'{save_path}/{scene}/{seq}', exist_ok=True)
            write_path = f'{save_path}/{scene}/{seq}/{time_window}.flo'
            writeFlow(write_path, flow)

        print('Saved to', save_path)

