"""
Gemoetry utils in bevformer
"""
import numpy as np
import torch
import torch.nn as nn


def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d',
                         bs=1, device='cuda', dtype=torch.float):
    """Get the reference points from Bev query or 2D BEV plane.
    Args:
        H, W: spatial shape of bev.
        Z: hight of pillar.
        D: sample D points uniformly from each pillar.
        dim: whether this is reference point from 3D query (for spatial
        cross attention) or 2d bev plane (for temporal attention)
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, num_keys, num_levels, 2).
    """
    # reference points in 3D space, used in spatial cross-attention (SCA)
    if dim == '3d':
        # normalized coordinates of z in the grid shape: (# of reference points, H, W)
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(
            num_points_in_pillar, H, W) / Z
        # normalized coordinates of x in the grid shape: (W, H, W)
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(
            num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(
            num_points_in_pillar, H, W) / H
        # stack them together to get the complete grids,
        # (# of reference points, H, W, 3)
        ref_3d = torch.stack((xs, ys, zs), -1)
        # ( # of ref, H*W, 3)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        # (bs, # of ref, H*W, 3)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    # reference points on 2D bev plane, used in temporal self-attention (TSA).
    elif dim == '2d':
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        # (b, h*w, 1, 2)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)

        return ref_2d


def point_sampling(reference_points, pc_range,  img_metas):
    """
    From the 3D query points find the image points.
    Args:
        reference_points: the grid for bev query reference points
        (b, # of ref, h*w, 3)
        pc_range: the BEV range in real world
        img_metas: the metadata (mostly lidar2cam needed)
    """
    lidar2img = []
    # the lidar2img contains cam_I * lidar_to_cam
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])

    # (b, 6, 4, 4)
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)
    reference_points = reference_points.clone()

    # get the reference points' real world coordinate whose origin is
    # the lidar pos of the vehicle
    reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]

    # construct homogenous transformation, (b, # of ref, h*w, 4)
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    # (#, b, h*w, 4)
    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    # (D, B, num_cam, h*w, 4, 1)
    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    # (D, B, num_cam, h*w, 4, 4)
    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
    # convert the 3D reference points to image coordinate
    # (D, B, num_cam, h*w, 4)
    reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                        reference_points.to(
                                            torch.float32)).squeeze(-1)
    eps = 1e-5
    # mask the points that are negative on z
    bev_mask = (reference_points_cam[..., 2:3] > eps)
    # normalize x, y using z to get the true image coordinate
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3],
        torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    # normalize x coordinate, 800 is the image width
    reference_points_cam[..., 0] /= 800
    reference_points_cam[..., 1] /= 600

    # filter out the points out of the image plane
    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))
    # Replaces NaN, positive infinity, and negative infinity values with 0s
    bev_mask = torch.nan_to_num(bev_mask)
    # (num_cam, b, h*w, D, 2)
    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    # (num_cam, b, h*w, D)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, bev_mask
