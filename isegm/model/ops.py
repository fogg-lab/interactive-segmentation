import torch
from torch import nn as nn
import numpy as np
import isegm.model.initializer as initializer

class qnode:
    def __init__(self):
        self.row = None
        self.col = None
        self.layer = None
        self.orig_row = None
        self.orig_col = None

def get_dist_maps(points: np.ndarray, height: int, width: int, norm_delimiter: float) -> dict:
    dxy = [-1, 0, 0, -1, 0, 1, 1, 0]
    q = []
    qhead = 0
    qtail = -1
    dist_maps = np.full((2, height, width), 1e6, dtype=np.float32, order="C")

    for i in range(points.shape[0]):
        x, y = round(points[i, 0]), round(points[i, 1])
        if x >= 0:
            qtail += 1
            q[qtail].row = x
            q[qtail].col = y
            q[qtail].orig_row = x
            q[qtail].orig_col = y
            if i >= points.shape[0] / 2:
                q[qtail].layer = 1
            else:
                q[qtail].layer = 0
            dist_maps[q[qtail].layer, x, y] = 0

    while qtail - qhead + 1 > 0:
        v = q[qhead]
        qhead += 1

        for k in range(4):
            x = v.row + dxy[2 * k]
            y = v.col + dxy[2 * k + 1]

            ndist = ((x - v.orig_row)/norm_delimiter) ** 2 + ((y - v.orig_col)/norm_delimiter) ** 2
            if (x >= 0 and y >= 0 and x < height and y < width and
                dist_maps[v.layer, x, y] > ndist):
                qtail += 1
                q[qtail].orig_col = v.orig_col
                q[qtail].orig_row = v.orig_row
                q[qtail].layer = v.layer
                q[qtail].row = x
                q[qtail].col = y
                dist_maps[v.layer, x, y] = ndist

    return dist_maps
    
def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False)

        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor
