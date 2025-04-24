import torch


class VoxelGrid():
    def __init__(self, input_size, normalize=True, norm_type='mean_std', device='cpu', keep_shape=False):
        assert len(input_size) == 3
        assert norm_type in ['mean_std', 'min_max']

        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False, device=device)
        self.nb_channels = input_size[0]
        self.normalize = normalize
        self.norm_type = norm_type
        self.keep_shape = keep_shape
        self.device = device

    def convert(self, events, chunk_size=1024*1024*50):
        """
        Convert events to voxel grid representation with memory-efficient chunking.
        
        Args:
            events: Dictionary containing event data
            chunk_size: Maximum number of events to process at once
        """
        C, H, W = self.voxel_grid.shape
        
        with torch.no_grad():
            # Determine batch size
            if 'batch_index' not in events.keys():
                bs = 1
                batch_index = torch.zeros_like(events['x'], dtype=torch.long)
            else:
                bs = torch.max(events['batch_index'])+1
                batch_index = events['batch_index']
            
            # Create output voxel grid on the specified device
            voxel_grid = torch.stack([self.voxel_grid]*bs, dim=0).to(self.device)
            
            # Create a copy of the timestamp data to avoid modifying input
            t_norm = events['t'].clone()
            
            # Normalize timestamps for each batch
            for i in range(bs):
                mask = batch_index == i
                if torch.sum(mask) < 1: continue
                if torch.sum(mask) < 2:
                    t_norm[mask] = 0
                    continue
                t_min = t_norm[mask][0]
                t_max = t_norm[mask][-1]
                t_norm[mask] = (C - 1) * (t_norm[mask]-t_min) / (t_max-t_min)
            
            # Get total number of events
            num_events = events['x'].shape[0]
            
            # Process events in chunks
            for start_idx in range(0, num_events, chunk_size):
                end_idx = min(start_idx + chunk_size, num_events)
                
                # Create chunk slices and move to device
                chunk_indices = slice(start_idx, end_idx)
                x_chunk = events['x'][chunk_indices].to(self.device)
                y_chunk = events['y'][chunk_indices].to(self.device)
                p_chunk = events['p'][chunk_indices].to(self.device)
                t_chunk = t_norm[chunk_indices].to(self.device)
                batch_chunk = batch_index[chunk_indices].to(self.device)
                
                # Process this chunk
                x0 = x_chunk.int()
                y0 = y_chunk.int()
                t0 = t_chunk.int()
                
                value = 2*p_chunk-1
                
                for xlim in [x0, x0+1]:
                    for ylim in [y0, y0+1]:
                        for tlim in [t0, t0+1]:
                            mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                            interp_weights = value * (1 - (xlim-x_chunk).abs()) * (1 - (ylim-y_chunk).abs()) * (1 - (tlim - t_chunk).abs())
                            
                            index = batch_chunk * C * H * W + \
                                    H * W * tlim.long() + \
                                    W * ylim.long() + \
                                    xlim.long()
                            
                            voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)
                            
                # Free chunk memory
                del x_chunk, y_chunk, p_chunk, t_chunk, batch_chunk, x0, y0, t0, value
                del mask, interp_weights, index
                
            # Normalize the voxel grid
            if self.normalize:
                for i in range(bs):
                    if self.norm_type == 'min_max':
                        maxv = torch.max(voxel_grid[i].abs())
                        voxel_grid[i] = voxel_grid[i] / maxv
                    elif self.norm_type == 'mean_std':
                        mask = torch.nonzero(voxel_grid[i], as_tuple=True)
                        if mask[0].size()[0] > 0:
                            mean = voxel_grid[i].mean()
                            std = voxel_grid[i].std()
                            if std > 0:
                                voxel_grid[i] = (voxel_grid[i] - mean) / std
                            else:
                                voxel_grid[i] = voxel_grid[i] - mean
        
        if bs == 1 and not self.keep_shape:
            voxel_grid = voxel_grid[0]
        
        return voxel_grid
