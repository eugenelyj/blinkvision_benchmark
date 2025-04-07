import h5py
import cv2
import numpy as np

def render(_x, _y, H, W):
    # assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    x, y = np.array(_x, dtype=np.int64), np.array(_y, dtype=np.int64)
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=1
    img[mask==0]=[255,255,255]
    # img[mask==-1]=[255,0,0]
    # img[mask==1]=[0,0,255]
    img[mask!=0]=[125,125,125]
    return img

def event_voxel_to_rgb(event_voxel, H=480, W=640):
    event_image = np.sum(event_voxel, axis=0)
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    img[event_image > 0] = [255,0,0]
    img[event_image < 0] = [0,0,255]
    return img

def event_to_rgb(_x, _y, pol, t, H, W):
    # assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    # Normalize timestamps to [0, 1]
    t_min = np.min(t)
    t_max = np.max(t)
    t = (t - t_min) / (t_max - t_min)
    
    x, y = np.array(_x, dtype=np.int64), np.array(_y, dtype=np.int64)
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = -np.ones((H,W),dtype='int32')
    t_map = np.zeros((H,W),dtype='float32')  # Add timestamp map
    
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]] = pol[mask1]
    t_map[y[mask1],x[mask1]] = t[mask1]  # Store timestamps
    
    # Scale color intensity based on timestamp
    img[mask==-1] = [255,255,255]  # Background remains white
    
    # For negative polarity (red), scale intensity with timestamp
    neg_mask = (mask == 0)
    img[neg_mask] = [
        255,  # R
        0,    # G
        0     # B
    ]
    img[neg_mask] = img[neg_mask] * (0.3 + 0.7 * t_map[neg_mask, None])  # Darker to brighter
    
    # For positive polarity (blue), scale intensity with timestamp
    pos_mask = (mask == 1)
    img[pos_mask] = [
        0,    # R
        0,    # G
        255   # B
    ]
    img[pos_mask] = img[pos_mask] * (0.3 + 0.7 * t_map[pos_mask, None])  # Darker to brighter
    
    return img
