import glob
import h5py
import hdf5plugin
import numpy as np
import torch
import os
import yaml
from event_utils import VoxelGrid
from torch.utils.data import Dataset, DataLoader

import os
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
torch.set_num_threads(1)


# NOTE: This loader load two consecutive event stream (in the range [t-interval, t] and [t, t+interval]) for predicting flow from time t to time t+interval
class BlinkFlow(Dataset):
    def __init__(self, data_path, split):
        assert os.path.isdir(data_path)
        assert split in ['training', 'test']

        self.height = 480
        self.width = 640
        data_path = os.path.join(data_path, split)
        self.split = split
        self.metadata_list = []

        print(f'Scanning data in {data_path}...')

        seq_list = sorted(glob.glob(os.path.join(data_path, '*/*')))
        for seq_path in seq_list:
            config_path = os.path.join(seq_path, 'config.yaml')
            if os.path.exists(config_path):
                f = open(config_path, 'r')
                cfg = yaml.safe_load(f)
                duration = cfg['duration']
                fps = cfg['fps']
            else:
                duration = 1.0
                fps = 10

            step_ms = int(1e3 / fps)
            hf = h5py.File(f'{seq_path}/events_left/events.h5')
            ms_to_idx = hf['ms_to_idx'][:]

            for idx in range(1, int(duration*fps)):
                if idx*step_ms >= len(ms_to_idx):
                    continue

                event_path = f'{seq_path}/events_left/events.h5'
                self.metadata_list.append({
                    'event_path': event_path,
                    'start_ms': int(step_ms*idx),
                    'end_ms': int(step_ms*(idx+1))
                })
                if split != 'test':
                    flow_path = f'{seq_path}/forward_flow/{idx:06d}.npy'
                    self.metadata_list[-1]['flow_path'] = flow_path

    def load_flow(self, flowfile):
        flow_32bit = np.load(flowfile)
        # valid flag labels forward-backward checking
        flow, valid2D = flow_32bit[:,:,:2], flow_32bit[:,:,2]
        # we train on all pixels
        valid2D = np.ones_like(valid2D)
        flow = np.transpose(flow, [2, 0, 1])
        return flow, valid2D

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        metadata = self.metadata_list[index]
        event_path = metadata['event_path']
        time_window = [metadata['start_ms'], metadata['end_ms']]
        time_interval = metadata['end_ms'] - metadata['start_ms']

        scene = event_path.split('/')[-4]
        seq = event_path.split('/')[-3]

        hf = h5py.File(event_path)
        ms_to_idx = hf['ms_to_idx'][:]
        ts_start_list = [time_window[0]-time_interval, time_window[0]]
        ts_end_list = [time_window[0], time_window[1]]
        event_list = []
        for i in range(2):
            ts_start = min(ts_start_list[i], ms_to_idx.shape[0]-1)
            ts_end = min(ts_end_list[i], ms_to_idx.shape[0]-1)
            sidx = ms_to_idx[ts_start]
            eidx = ms_to_idx[ts_end]
            t = hf['events/t'][sidx:eidx].astype(np.float32)
            t = (t-np.min(t)) / (np.max(t)-np.min(t))
            y = hf['events/y'][sidx:eidx].astype(np.float32)
            x = hf['events/x'][sidx:eidx].astype(np.float32)
            p = hf['events/p'][sidx:eidx].astype(np.float32)
            events = np.stack([x, y, t, p], axis=1)
            event_list.append(events)

        output = {
            names[0]: event_list[0],
            names[1]: event_list[1],
            'scene': scene,
            'seq': seq,
            'time_window': f'{time_window[0]}_{time_window[1]}',
        }

        if self.split != 'test':
            flow_path = metadata['flow_path']
            flow, valid = self.load_flow(flow_path)
            output['flow'] = flow
            output['valid'] = valid

        return output

def collate_event(batch):
    ret = {}
    for key in ['event_volume_old', 'event_volume_new']:
        event_list = []
        batch_index_list = []
        for i in range(len(batch)):
            item = torch.from_numpy(batch[i][key])
            event_list.append(item)
            batch_index_list.append(torch.ones_like(item[:,0])*i)
        event = torch.cat(event_list, dim=0)
        batch_index = torch.cat(batch_index_list, dim=0)
        ret[key] = torch.cat([event, batch_index.unsqueeze(-1)], 1)
    for key in ['scene', 'seq', 'time_window']:
        ret[key] = [item[key] for item in batch]

    if 'flow' in batch[0]:
        ret['flow'] = torch.from_numpy(np.stack([item['flow'] for item in batch], axis=0))

    if 'valid' in batch[0]:
        ret['valid'] = torch.from_numpy(np.stack([item['valid'] for item in batch], axis=0))

    return ret

def event2voxel(batch, num_bins=15, height=480, width=640):
    voxel_grid = VoxelGrid((num_bins, height, width), normalize=True, device='cuda')
    event_old = batch['event_volume_old'].cuda()
    event_new = batch['event_volume_new'].cuda()
    image1 = voxel_grid.convert({
        'p': event_old[:,0],
        't': event_old[:,1],
        'x': event_old[:,2],
        'y': event_old[:,3],
        'batch_index': event_old[:,4].to(torch.long)
    })
    image2 = voxel_grid.convert({
        'p': event_new[:,0],
        't': event_new[:,1],
        'x': event_new[:,2],
        'y': event_new[:,3],
        'batch_index': event_new[:,4].to(torch.long)
    })
    return image1, image2


def test_dataflow():
    import tqdm
    root = 'data/blinkflow/'

    dataset = BlinkFlow(data_path=root, split='training')
    train_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_event,
                            drop_last=True)
    for i, batch in enumerate(tqdm.tqdm(train_loader)):
        event_voxel = event2voxel(batch)
        if i > 10:
            pass

    dataset = BlinkFlow(data_path=root, split='test')
    test_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_event,
                            drop_last=True)
    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        event_voxel = event2voxel(batch)
        if i > 10:
            pass



if __name__ == '__main__':
    test_dataflow()

