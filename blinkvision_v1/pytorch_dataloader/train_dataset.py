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
from vis import make_video, visualize_optical_flow, visualize_depth, visualize_trajectory, visualize_event

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

# define pytorch dataset
class BlinkvisionDataset(torch.utils.data.Dataset):
    def __init__(self, root, config):
        self.root = root
        self.config = config

        self.pose_rescaled_list = [
            'indoor_train/AI33_002_280_people',
            'indoor_train/AI33_007_280_people',
            'indoor_train/AI33_008_280_people',
            'indoor_train/AI48_001_people',
            'indoor_train/AI48_008_people',
            'indoor_train/AI48_009_people',
            'indoor_train/AI58_001_people',
            'indoor_train/AI58_006_people',
            'indoor_train/AI58_008_people',
            'indoor_train/AI33_002_280_tour',
            'indoor_train/AI33_007_280_tour',
            'indoor_train/AI33_008_280_tour',
            'indoor_train/AI48_001_tour',
            'indoor_train/AI48_008_tour',
            'indoor_train/AI48_009_tour',
            'indoor_train/AI58_001_tour',
            'indoor_train/AI58_006_tour',
            'indoor_train/AI58_008_tour',
        ]

        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        mapping_file_path = os.path.join(parent_directory, 'mapping_train.txt')

        with open(mapping_file_path, 'r') as f:
            lines = f.readlines()
            data = [line.strip('\n').split(', ') for line in lines]
            self.seq_list = []
            for (scene_pattern_a, scene_pattern_b) in data:
                category, seq = scene_pattern_b.rsplit('_', 1)
                pose_need_rescaled = True if scene_pattern_a in self.pose_rescaled_list else False
                self.seq_list.append([scene_pattern_a, f'{category}/{seq}', pose_need_rescaled])

            self.used_scene_pattern = None
            if os.path.exists(f'{self.root}/{self.seq_list[0][0]}'):
                self.used_scene_pattern = 'a'
            elif os.path.exists(f'{self.root}/{self.seq_list[0][1]}'):
                self.used_scene_pattern = 'b'
            else:
                raise ValueError(f'{self.root}/{self.seq_list[0][0]} or {self.root}/{self.seq_list[0][1]} does not exist')
        
        # there are two download links and have different name pattern
        new_list = []
        for (scene_pattern_a, scene_pattern_b, pose_need_rescaled) in self.seq_list:
            if self.used_scene_pattern == 'a':
                seq_dir = f'{self.root}/{scene_pattern_a}'
                seq_name = scene_pattern_a.rsplit('/', 1)[1]
            elif self.used_scene_pattern == 'b':
                seq_dir = f'{self.root}/{scene_pattern_b}'
                seq_name = scene_pattern_b.rsplit('/', 1)[1]
            new_list.append([seq_dir, seq_name, pose_need_rescaled])
        self.seq_list = new_list

        # some seq does not have event data
        if 'event' in self.config and self.config['event']:
            new_list = []
            for (seq_dir, seq_name, pose_need_rescaled) in self.seq_list:
                event_seq_dir = seq_dir
                event_seq_dir = event_seq_dir.replace('/indoor_train/', '/indoor_train_event_960/')
                event_seq_dir = event_seq_dir.replace('/outdoor_train/', '/outdoor_train_event/')
                if self.used_scene_pattern == 'a':
                    event_file = f'{event_seq_dir}/events_left/events.h5'
                else:
                    event_file = f'{event_seq_dir}/events_left_resize960/events.h5'
                if not os.path.exists(event_file):
                    continue
                new_list.append([seq_dir, seq_name, pose_need_rescaled])
            self.seq_list = new_list

    def get_event(self, seq_dir):
        if self.used_scene_pattern == 'a':
            event_file = f'{seq_dir}/events_left/events.h5'
        else:
            event_file = f'{seq_dir}/events_left_resize960/events.h5'
        if not os.path.exists(event_file):
            return {
                'event': None
            }
        try:
            fps = 20
            ms_interval = int(1000 / fps)
            hf = h5py.File(event_file)
            ms_to_idx = hf['ms_to_idx'][:]
            event_list = []
            for i in range(0, len(ms_to_idx), ms_interval):
                ts_start = i
                ts_end = min(i+ms_interval, len(ms_to_idx)-1)
                sidx = ms_to_idx[ts_start]
                eidx = ms_to_idx[ts_end]
                t = hf['events/t'][sidx:eidx].astype(np.float32)
                y = hf['events/y'][sidx:eidx].astype(np.float32)
                x = hf['events/x'][sidx:eidx].astype(np.float32)
                p = hf['events/p'][sidx:eidx].astype(np.float32)
                events = np.stack([x, y, t, p], axis=1)
                event_list.append(events)
        except Exception as e:
            print('#'*100)
            print(f'Corrupted event file: {event_file}')
            print('#'*100)
            return {
                'event': None
            }
                
        return {
            'event': event_list
        }
            

    def get_clean(self, seq_dir):
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

    def get_final(self, seq_dir):
        final_dir = f'{seq_dir}/final_uint8'
        file_list = sorted(os.listdir(final_dir))
        file_list = [file for file in file_list if file.endswith('.png')]
        file_list = [os.path.join(final_dir, file) for file in file_list]

        img_list = []
        for file in file_list:
            with Image.open(file) as img:
                img_list.append(np.array(img))

        return {
            'final': img_list
        }

    def get_optical_flow(self, seq_dir):
        optical_flow_dir = f'{seq_dir}/forward_flow'
        file_list = sorted(os.listdir(optical_flow_dir))
        file_list = [file for file in file_list if file.endswith('.npz')]
        file_list = [os.path.join(optical_flow_dir, file) for file in file_list]

        data_list = []
        for file in file_list:
            data = np.load(file)
            data_list.append(data['forward_flow'])

        return {
            'optical_flow': data_list
        }


    def get_optical_flow_custom_stride(self, seq_dir):
        forward_flow_custom_stride_dir = f'{seq_dir}/forward_flow_custom_stride'

        forward_flow_custom_stride_file_list = sorted(os.listdir(forward_flow_custom_stride_dir))
        forward_flow_custom_stride_file_list = [file for file in forward_flow_custom_stride_file_list if file.endswith('.npz')]
        forward_flow_custom_stride_file_list = [os.path.join(forward_flow_custom_stride_dir, file) for file in forward_flow_custom_stride_file_list]

        flow_list = []
        idx_list = []
        for forward_flow_file in forward_flow_custom_stride_file_list:
            basename = os.path.basename(forward_flow_file).split('.')[0]
            idx1, idx2 = basename.split('_')
            idx_list.append([int(idx1), int(idx2)])
            forward_flow_data = np.load(forward_flow_file)
            forward_flow_data = forward_flow_data['forward_flow']
            flow_list.append(forward_flow_data)

        return {
            'optical_flow_custom_stride': flow_list,
            'idx_optical_flow': idx_list
        }


    def get_particle(self, seq_dir):
        particle_dir = f'{seq_dir}/particle'
        particle_json = f'{seq_dir}/particle.json'

        particle_list = []

        with open(particle_json, 'r') as f:
            seg_list = json.load(f)


        for seg in seg_list:
            idx1, idx2 = seg
            data = []
            for i in range(idx1, idx2):
                partcile_file = f'{particle_dir}/{idx1:06d}_{i:06d}.npz'
                particle_data = np.load(partcile_file)
                particle_data = particle_data['particle']
                data.append(particle_data)
            particle_list.append(data)

        return {
            'particle': particle_list,
            'idx_particle': seg_list,
        }

    def get_depth(self, seq_dir):
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

    def get_intrinsic(self, seq_dir):
        with open(os.path.join(seq_dir, 'metadata.json')) as f:
            metadata = json.load(f)
        intrinsic = np.array(metadata['K_matrix']).astype(np.float32)
        return {
            'intrinsic': intrinsic
        }

    def get_camera_pose(self, seq_dir):
        camera_pose_file = f'{seq_dir}/poses.npz'
        data = np.load(camera_pose_file)['camera_poses']
        data = [convert_to_c2w(pose) for pose in data]
        return {
            'camera_pose': data
        }
        
        
    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        seq_dir, seq_name, pose_need_rescaled = self.seq_list[index]
        # loop the dict in config, if the key is true, then call the corresponding function
        ret_dict = {'seq_name': seq_name}
        for key, value in self.config.items():
            if value:
                if key == 'event':
                    event_seq_dir = seq_dir
                    event_seq_dir = event_seq_dir.replace('/indoor_train/', '/indoor_train_event_960/')
                    event_seq_dir = event_seq_dir.replace('/outdoor_train/', '/outdoor_train_event/')
                    sub_dict = getattr(self, f'get_{key}')(event_seq_dir)
                else:
                    sub_dict = getattr(self, f'get_{key}')(seq_dir)
                ret_dict.update(sub_dict)

        if pose_need_rescaled and 'camera_pose' in ret_dict:
            for i in range(len(ret_dict['camera_pose'])):
                ret_dict['camera_pose'][i][:3, 3] = ret_dict['camera_pose'][i][:3, 3] / 100

        return ret_dict


def render_sequential(root_dir, save_dir):
    config = {
        'clean': True,
        'final': True,
        'depth': True,
    }
    dataset = BlinkvisionDataset(root=root_dir, config=config)
    print(len(dataset))
    for data in dataset:
        seq_name = data['seq_name']
        clean_data_list = data['clean']
        final_data_list = data['final']
        depth_data_list = data['depth']

        print(seq_name)
        vis_depth_list = visualize_depth(depth_data_list)

        data_list = [clean_data_list, final_data_list, vis_depth_list]
        make_video(data_list, outvid=f'{save_dir}/{seq_name}.mp4')


def render_custom_stride(root_dir, save_dir):
    config = {
        'optical_flow_custom_stride': True,
        'particle': True,
        'clean': True,
    }
    dataset = BlinkvisionDataset(root=root_dir, config=config)
    for data in dataset:
        seq_name = data['seq_name']
        optical_flow_custom_stride_list = data['optical_flow_custom_stride']
        particle_list = data['particle']
        idx_optical_flow_list = data['idx_optical_flow']
        idx_particle_list = data['idx_particle']
        clean_list = data['clean']

        print(seq_name)

        clean_source = [clean_list[idx1] for idx1, idx2 in idx_optical_flow_list]
        clean_target = [clean_list[idx2] for idx1, idx2 in idx_optical_flow_list]
        flow = [data[..., :2] for data in optical_flow_custom_stride_list]
        status = [data[..., 2] for data in optical_flow_custom_stride_list]
        vis_optical_flow_list = visualize_optical_flow(flow)
        for i in range(len(vis_optical_flow_list)):
            vis_optical_flow_list[i][status[i] == 0] = 0

        data_list = [clean_source, clean_target, vis_optical_flow_list]
        make_video(data_list, outvid=f'{save_dir}/{seq_name}_flow.mp4')

        vis_particle_list = []
        for (track_info, sub_particle_list) in zip(idx_particle_list, particle_list):
            vis_particle_list.append(visualize_trajectory(sub_particle_list, clean_list, track_info))

        vis_particle_list = [item for sublist in vis_particle_list for item in sublist]

        data_list = [vis_particle_list]
        make_video(data_list, outvid=f'{save_dir}/{seq_name}_particle.mp4')

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
        H, W = 540, 960
        
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

    config = {
        'depth': True,
        'camera_pose': True,
        'intrinsic': True,
        'clean': True,
    }
    dataset = BlinkvisionDataset(root=root_dir, config=config)
    print('dataset length', len(dataset))
    stride = 10
    for data in dataset:
        seq_name = data['seq_name']
        print(seq_name)
        depth_list = data['depth']
        camera_pose_list = data['camera_pose']
        clean_list = data['clean']
        intrinsic = data['intrinsic']
        points_list = []
        color_list = []
        for i in range(0, min(len(depth_list), stride*5), stride):
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

def render_event(root_dir, save_dir):
    config = {
        'event': True,
        'clean': True,
    }

    dataset = BlinkvisionDataset(root=root_dir, config=config)
    print(len(dataset))
    for data in dataset:
        seq_name = data['seq_name']
        event_data_list = data['event']
        clean_data_list = data['clean']

        if event_data_list is None:
            continue

        print(seq_name)
        height, width = clean_data_list[0].shape[:2]
        vis_event_list = visualize_event(event_data_list, height, width)

        data_list = [clean_data_list[:-1], vis_event_list]
        make_video(data_list, outvid=f'{save_dir}/{seq_name}_event.mp4')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='tmp')
    parser.add_argument('--mode', type=str, default='render_3D')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'render_sequential':
        render_sequential(args.root_dir, args.save_dir)
    elif args.mode == 'render_custom_stride':
        render_custom_stride(args.root_dir, args.save_dir)
    elif args.mode == 'render_3D':
        render_3D(args.root_dir, args.save_dir)
    elif args.mode == 'render_event':
        render_event(args.root_dir, args.save_dir)

