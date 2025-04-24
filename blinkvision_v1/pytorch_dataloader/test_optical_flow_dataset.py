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
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
torch.set_num_threads(1)

# get the current file path
current_folder = os.path.dirname(os.path.abspath(__file__))


# NOTE: This loader load two consecutive event stream (in the range [t-interval, t] and [t, t+interval]) for predicting flow from time t to time t+interval
class BlinkVisionFlow(Dataset):
    def __init__(self, data_path, sample_map_path, config):
        assert os.path.isdir(data_path)

        self.height = 480
        self.width = 640
        self.fps = 20
        self.step_ms = int(1e3 / self.fps)
        self.config = config

        print(f'Scanning data ...')

        with open(os.path.join(current_folder, 'mapping_test.txt'), 'r') as f:
            mapping_list = f.readlines()
        mapping_list = [item.strip() for item in mapping_list]
        mapping_list = [item.split(',') for item in mapping_list]
        for i in range(len(mapping_list)):
            parts = mapping_list[i][1].rsplit('_', 1)
            mapping_list[i][1] = parts[0].strip() + '/' + parts[1].strip()

        self.metadata_list = []
        sample_map_list = sorted(glob.glob(os.path.join(sample_map_path, '**/*.bin'), recursive=True))
        print(len(sample_map_list))
        for map_path in sample_map_list:
            basename = os.path.basename(map_path).split('.')[0]
            relative_path = os.path.relpath(os.path.dirname(map_path), sample_map_path)
            scene = relative_path.split('/')[-2]
            seq = relative_path.split('/')[-1]
            folder_path = ''
            if os.path.exists(os.path.join(data_path, relative_path)):
                folder_path = os.path.join(data_path, relative_path)
            else:
                # find pattern in mapping_list
                for item in mapping_list:
                    if item[0] in relative_path:
                        folder_path = os.path.join(data_path, item[1])
                        break

            start_idx, end_idx = map(int, basename.split('_'))

            self.metadata_list.append({
                'folder_path': folder_path,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'scene': scene,
                'seq': seq,
            })


    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        metadata = self.metadata_list[index]
        folder_path = metadata['folder_path']
        start_idx = metadata['start_idx']
        end_idx = metadata['end_idx']

        return_dict = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'scene': metadata['scene'],
            'seq': metadata['seq'],
        }

        if 'clean' in self.config and self.config['clean']:
            clean_path = f'{folder_path}/clean_uint8'
            clean_img1 = iio.imread(f'{clean_path}/{start_idx:06d}.png')
            clean_img2 = iio.imread(f'{clean_path}/{end_idx:06d}.png')
            clean_img1 = clean_img1.transpose(2, 0, 1)
            clean_img2 = clean_img2.transpose(2, 0, 1)
            return_dict['clean_img1'] = clean_img1
            return_dict['clean_img2'] = clean_img2

        if 'final' in self.config and self.config['final']:
            final_path = f'{folder_path}/final_uint8'
            final_img1 = iio.imread(f'{final_path}/{start_idx:06d}.png')
            final_img2 = iio.imread(f'{final_path}/{end_idx:06d}.png')
            final_img1 = final_img1.transpose(2, 0, 1)
            final_img2 = final_img2.transpose(2, 0, 1)
            return_dict['final_img1'] = final_img1
            return_dict['final_img2'] = final_img2

        if 'event' in self.config and self.config['event']:

            event_path = f'{folder_path}/events_left/events.h5'
            time_window = [start_idx*self.step_ms, end_idx*self.step_ms]
            time_interval = time_window[1] - time_window[0]

            hf = h5py.File(event_path)
            ms_to_idx = hf['ms_to_idx'][:]
            ts_start_list = [time_window[0]-time_interval, time_window[0]]
            ts_end_list = [time_window[0], time_window[1]]
            event_list = []
            for i in range(2):
                ts_start = min(ts_start_list[i], ms_to_idx.shape[0]-1)
                ts_end = min(ts_end_list[i], ms_to_idx.shape[0]-1)
                if ts_start < 0:
                    return_dict[names[i]] = None
                    continue
                sidx = ms_to_idx[ts_start]
                eidx = ms_to_idx[ts_end]
                if sidx >= eidx:
                    return_dict[names[i]] = None
                    continue
                t = hf['events/t'][sidx:eidx].astype(np.float32)
                t = (t-np.min(t)) / (np.max(t)-np.min(t))
                y = hf['events/y'][sidx:eidx].astype(np.float32)
                x = hf['events/x'][sidx:eidx].astype(np.float32)
                p = hf['events/p'][sidx:eidx].astype(np.float32)
                events = np.stack([x, y, t, p], axis=1)
                return_dict[names[i]] = events

        return return_dict

def collate_event(batch):
    ret = {}
    for key in ['event_volume_old', 'event_volume_new']:
        if key not in batch[0]:
            continue
        event_list = []
        batch_index_list = []
        for i in range(len(batch)):
            if batch[i][key] is None:
                continue
            item = torch.from_numpy(batch[i][key])
            event_list.append(item)
            batch_index_list.append(torch.ones_like(item[:,0])*i)
        if len(event_list) == 0:
            ret[key] = None
            continue
        event = torch.cat(event_list, dim=0)
        batch_index = torch.cat(batch_index_list, dim=0)
        ret[key] = torch.cat([event, batch_index.unsqueeze(-1)], 1)

    for key in ['scene', 'seq', 'start_idx', 'end_idx']:
        ret[key] = [item[key] for item in batch]

    for key in ['clean_img1', 'clean_img2', 'final_img1', 'final_img2']:
        if key not in batch[0]:
            continue
        ret[key] = torch.stack([torch.from_numpy(np.array(item[key], copy=True)) for item in batch], dim=0)

    return ret

def event2voxel(batch, num_bins=15, height=480, width=640):
    voxel_grid = VoxelGrid((num_bins, height, width), normalize=True, device='cuda')
    if batch['event_volume_old'] is not None:
        event_old = batch['event_volume_old']
        image1 = voxel_grid.convert({
            'x': event_old[:,0],
            'y': event_old[:,1],
            't': event_old[:,2],
            'p': event_old[:,3],
            'batch_index': event_old[:,4].to(torch.long)
        })
    else:
        image1 = torch.zeros((num_bins, height, width)).cuda()

    if batch['event_volume_new'] is not None:
        event_new = batch['event_volume_new']
        image2 = voxel_grid.convert({
            'x': event_new[:,0],
            'y': event_new[:,1],
            't': event_new[:,2],
            'p': event_new[:,3],
            'batch_index': event_new[:,4].to(torch.long)
        })
    else:
        image2 = torch.zeros((num_bins, height, width)).cuda()

    return image1, image2


def test_dataflow():
    import tqdm
    root = 'data/blinkvision/test_data'
    sample_map_path = 'data/blinkvision/sample_maps'
    config = {
        'clean': True,
        'final': True,
        'event': True,
    }

    dataset = BlinkVisionFlow(data_path=root, sample_map_path=sample_map_path, config=config)
    test_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=collate_event,
                            drop_last=True)
    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        if i == 0:
            event_voxel_list = event2voxel(batch)
            print(batch['event_volume_new'].shape)
            print(event_voxel_list[1].shape)
            print(batch['clean_img1'].shape)
        if i > 10:
            break


if __name__ == '__main__':
    test_dataflow()


