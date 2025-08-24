import glob
import h5py
import numpy as np
import torch
import os
import yaml
import imageio.v3 as iio
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import Dataset, DataLoader
from event_utils import VoxelGrid
import os
from PIL import Image
import cv2
import csv
import tqdm
import numpy as np
import torch
import json
import h5py
import random
from scipy.spatial.transform import Rotation as R


import os
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
torch.set_num_threads(1)

# get the current file path
current_folder = os.path.dirname(os.path.abspath(__file__))

def load_data(seq_dir):

    pose_rescaled_list = [
        'indoor_people/AI33_001_280_people',
    ]

    for i in range(len(pose_rescaled_list)):
        pose_rescaled_list[i] = pose_rescaled_list[i].replace('/', '_')

    blender2opencv = np.float32([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])


    def convert_to_c2w(frame): # to get world-to-camera matrix
        translation = frame[0] 
        rotation = frame[1] 
        
        RR = R.from_euler('xyz', rotation, degrees=False).as_matrix()
        
        c2w = np.eye(4)
        c2w[:3, :3] = RR.T
        c2w[:3, 3] = -RR.T @ translation

        c2w = blender2opencv @ c2w

        return c2w

    def get_clean(seq_dir):
        clean_dir = f'{seq_dir}/clean_uint8'
        file_list = sorted(os.listdir(clean_dir))
        file_list = [file for file in file_list if file.endswith('.png')]
        file_list = [os.path.join(clean_dir, file) for file in file_list]

        img_list = []
        for file in file_list:
            with Image.open(file) as img:
                img_list.append(np.array(img))

        return {
            'clean': img_list
        }


    def get_depth(seq_dir):
        depth_dir = f'{seq_dir}/depth'
        file_list = sorted(os.listdir(depth_dir))
        file_list = [file for file in file_list if file.endswith('.npz')]
        file_list = [os.path.join(depth_dir, file) for file in file_list]

        data_list = []
        for file in file_list:
            data = np.load(file)
            data_list.append(data['depth'])
        
        return {
            'depth': data_list
        }

    def get_intrinsic(seq_dir):
        with open(os.path.join(seq_dir, 'metadata.json')) as f:
            metadata = json.load(f)
        intrinsic = np.array(metadata['K_matrix']).astype(np.float32)
        return {
            'intrinsic': intrinsic
        }

    def get_camera_pose(seq_dir):
        camera_pose_file = f'{seq_dir}/poses.npz'
        data = np.load(camera_pose_file)['camera_poses']
        data = [convert_to_c2w(pose) for pose in data]
        return {
            'camera_pose': data
        }

    seq_name = '_'.join(seq_dir.rsplit('/', 2)[-2:])
    try:
        return_dict = {}
        return_dict['seq_name'] = seq_name
        return_dict.update(get_clean(seq_dir))
        return_dict.update(get_depth(seq_dir))
        return_dict.update(get_intrinsic(seq_dir))
        return_dict.update(get_camera_pose(seq_dir))

        # pose_need_rescaled = seq_name in pose_rescaled_list
        pose_need_rescaled = True

        if pose_need_rescaled and 'camera_pose' in return_dict:
            for i in range(len(return_dict['camera_pose'])):
                return_dict['camera_pose'][i][:3, 3] = return_dict['camera_pose'][i][:3, 3] / 100

        return return_dict
    except Exception as e:
        print(f'Error loading data for {seq_dir}: {e}')
        return None


def render_3D(root_dir, save_dir):
    from plyfile import PlyData, PlyElement

    def get_pixel(H, W):
        # get 2D pixels (u, v) for image_a in cam_a pixel space
        u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
        # u_a = np.flip(u_a, axis=1)
        # v_a = np.flip(v_a, axis=0)
        pixels_a = np.stack([
            u_a.flatten() + 0.5, 
            v_a.flatten() + 0.5, 
            np.ones_like(u_a.flatten())
        ], axis=0)
        
        return pixels_a

    def unproject(rgb, depth, intrinsic, c2w):
        H, W = depth.shape
        mask = (depth > 0).reshape(-1)
        pixel = get_pixel(H, W).astype(np.float32)
        
        w2c = np.linalg.inv(c2w)

        points = (np.linalg.inv(intrinsic) @ pixel) * depth.reshape(-1)
        points = w2c[:3, :4] @ np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)

        points = points.T[mask]
        color = rgb.reshape(-1, 3)[mask]/255

        return points, color

    def write_ply(
        xyz,
        rgb,
        path='output.ply',
    ) -> None:
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        normals = np.zeros_like(xyz)
        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
        elements[:] = list(map(tuple, attributes))
        vertex_element = PlyElement.describe(elements, "vertex")
        ply_data = PlyData([vertex_element])
        ply_data.write(path)

    def render_point_cloud(points, color, intrinsic, c2w):
        import torch
        import imageio
        from pytorch3d.structures import Pointclouds
        from pytorch3d.utils import cameras_from_opencv_projection
        from pytorch3d.renderer import (
            PerspectiveCameras,
            PointsRasterizationSettings,
            PointsRenderer,
            PointsRasterizer,
            AlphaCompositor
        )

        device = torch.device("cuda:0")
        H, W = 480, 640
        
        # Convert points and colors to torch tensors with correct shapes
        points = torch.tensor(points, device=device, dtype=torch.float32).unsqueeze(0)  # Shape: (1, N, 3)
        color = torch.tensor(color, device=device, dtype=torch.float32).unsqueeze(0)   # Shape: (1, N, 3)
        
        # Create point cloud with batched inputs
        point_cloud = Pointclouds(points=points, features=color)  # points: (B, N, 3), features: (B, N, 3)
        
        # Rest of the camera setup
        R = torch.tensor(c2w[:3, :3], device=device, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 3, 3)
        tvec = torch.tensor(c2w[:3, 3], device=device, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 3)
        camera_matrix = torch.tensor(intrinsic, device=device, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 3, 3)
        image_size = torch.tensor([[H, W]], device=device, dtype=torch.float32)  # Shape: (1, 2)
        
        cameras = cameras_from_opencv_projection(
            R=R,  # Shape: (1, 3, 3)
            tvec=tvec,  # Shape: (1, 3)
            camera_matrix=camera_matrix,  # Shape: (1, 3, 3)
            image_size=image_size,  # Shape: (1, 2)
        )
        
        # Create rasterizer settings
        raster_settings = PointsRasterizationSettings(
            image_size=(H, W),
            radius=0.01,
            points_per_pixel=10,
            bin_size=0,  # Use naive rasterization without binning
            # Alternative settings if you want to keep binning:
            # bin_size=64,
            # max_points_per_bin=100000  # Increase from default
        )
        
        # Create renderer
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            compositor=AlphaCompositor()
        )
        
        # Render
        images = renderer(point_cloud)
        image = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        
        # Save
        imageio.imwrite(f'{save_dir}/{seq_name}_render.png', image)

    stride = 10

    with open('blinkvision_v1/pytorch_dataloader/mapping_test.txt', 'r') as f:
        mapping_test = f.readlines()
    scene_list = [line.strip().split(',')[0] for line in mapping_test]
    print('len(scene_list)', len(scene_list))
    error_scene_list = []
    for scene in scene_list:
        print(os.path.join(root_dir, scene, 'depth'))
        data = load_data(os.path.join(root_dir, scene))
        if data is None:
            continue

        seq_name = data['seq_name']
        print(seq_name)
        depth_list = data['depth']
        camera_pose_list = data['camera_pose']
        clean_list = data['clean']
        intrinsic = data['intrinsic']
        points_list = []
        color_list = []

        for i in range(stride, min(len(depth_list), stride*5), stride):
            depth = depth_list[i]
            camera_pose = camera_pose_list[i]
            clean = clean_list[i]
            points, color = unproject(clean, depth, intrinsic, camera_pose)
            points_list.append(points)
            color_list.append(color)
        # uniform downsample
        points = np.concatenate(points_list, axis=0)[::5]
        color = np.concatenate(color_list, axis=0)[::5]
        write_ply(points, color, path=f'{save_dir}/{seq_name}.ply')
        render_point_cloud(points, color, intrinsic, camera_pose_list[0])

    with open(f'error_scene_list.txt', 'w') as f:
        for scene in error_scene_list:
            f.write(scene + '\n')



if __name__ == '__main__':
    root_dir = 'data/blinkvision/test_data'
    save_dir = 'data/vis_test'
    os.makedirs(save_dir, exist_ok=True)
    render_3D(root_dir, save_dir)


