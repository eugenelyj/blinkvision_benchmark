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

import torch.nn.functional as F
from torchvision.transforms import Compose

# Local imports
# sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('blinkvision_v1')
from pytorch_dataloader.test_depth_dataset import collate_event, BlinkVisionDepth, event2voxel

TAG_VALUE = 2502001 # full data


def load_depth_anything_v1(encoder='vits', device='cuda:0'):
    from depth_anything.dpt import DepthAnything

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(device).eval()
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    return depth_anything

def process_image(image, depth_anything, device='cuda:0'):
    # Import the Resize class from transform.py
    from depth_anything.util.transform import Resize

    # Assuming image is already a PyTorch tensor with shape (C, H, W) or (B, C, H, W)
    if image.ndim == 3:
        image = image.unsqueeze(0)  # Add batch dimension if not present
    
    # Get original dimensions
    _, _, h, w = image.shape
    
    # Convert PyTorch tensor to numpy format for the Resize class
    # Convert from (B, C, H, W) to (H, W, C) format
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Create sample dict expected by Resize class
    sample = {"image": image_np}
    
    # Create Resize transform with specified parameters
    resize_transform = Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    
    # Apply resize transformation
    sample = resize_transform(sample)
    resized_image_np = sample["image"]
    
    # Convert back to PyTorch tensor format (B, C, H, W)
    resized_image = torch.from_numpy(resized_image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Convert to float32 and normalize to [0, 1]
    resized_image = resized_image.to(dtype=torch.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    resized_image = (resized_image - mean) / std
    
    with torch.no_grad():
        depth = depth_anything(resized_image)

    # Interpolate back to original size
    depth = depth.unsqueeze(0)
    depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)
    depth = depth.squeeze(0).squeeze(0)
    
    return depth
    
 

def load_rpg_e2depth():
    # TODO: add your model here
    return None


def writeDepth(output_file, depth):
    # depth shape: (H, W)
    height, width = depth.shape[:2]
    with open(output_file, 'wb') as f:
        np.array(TAG_VALUE).astype(np.int32).tofile(f)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        depth.reshape(-1).tofile(f)

def run_frame_model():

    # We provide an example that run depth-anything-v1
    data_path = 'data/blinkvision/test_data'
    sample_map_path = 'data/blinkvision/benchmark/depth/sample_maps'
    config = {
        'event': False,
        'clean': True,
        'final': True,
    }

    model = load_depth_anything_v1()
    save_path = f'results/bv_depth_frame'

    test_set = BlinkVisionDepth(data_path=data_path, sample_map_path=sample_map_path, config=config)
    test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate_event,
                            drop_last=False)

    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        clean_img1 = batch['clean_img1'].cuda()
        final_img1 = batch['final_img1'].cuda()

        scene = batch['scene'][0]
        seq = batch['seq'][0]
        frame_idx = batch['frame_idx'][0]
        os.makedirs(f'{save_path}/clean/{scene}/{seq}', exist_ok=True)
        os.makedirs(f'{save_path}/final/{scene}/{seq}', exist_ok=True)

        with torch.no_grad():
            clean_depth = process_image(clean_img1, model)
            final_depth = process_image(final_img1, model)
        
        clean_depth = clean_depth.cpu().numpy()
        final_depth = final_depth.cpu().numpy()

        clean_write_path = f'{save_path}/clean/{scene}/{seq}/{frame_idx:06d}.bin'
        final_write_path = f'{save_path}/final/{scene}/{seq}/{frame_idx:06d}.bin'
        writeDepth(clean_write_path, clean_depth)
        writeDepth(final_write_path, final_depth)

    print('Saved to', save_path)



def run_event_model():
    # We provide an example that run rpg_e2depth, "Learning Monocular Dense Depth from Events"
    data_path = ''
    sample_map_path = ''
    config = {
        'event': True,
    }

    model = load_rpg_e2depth()
    save_path = f'results/bv_depth_event'

    test_set = BlinkVisionDepth(data_path=data_path, sample_map_path=sample_map_path, config=config)
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
        writeDepth(write_path, flow)

    print('Saved to', save_path)



if __name__ == '__main__':
    run_frame_model()
    # run_event_model()

