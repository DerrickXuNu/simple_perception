"""
Basic Modules for PointNet++
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Find N fasrthest points from the pointcloud as the centroid.

    Parameters
    ----------
    xyz : torch.Tensor
        Point clouds.

    npoint : int
        The number of centroids needed.

    Returns
    -------
    centroids: torch.Tensor
        Sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    # batch size, n points, c channels
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # the distance from each point to its closest centroid
    distance = torch.ones(B, N).to(device) * 1e10
    # shape (B), random pick the fasthest point in the first round
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # (0,1,2...,B)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    # find the fastest n points in the loop
    for i in range(npoint):
        # (B, N)
        centroids[:, i] = farthest
        # (B, 1, 3), the centriod in this iteration
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # the distance of each point to this centroid
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # we only update the distance of points whose closet centroid is this
        # iteration's centroid
        mask = dist < distance
        distance[mask] = dist[mask]
        # find the farthest point to all existing centroids. f
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Get the centroid x y z from the index.

    Parameters
    ----------
    points: input points data, [B, N, C]
    idx: sample index data, [B, S] or [B, S, nsample]

    Return
    ------
    new_points:, indexed points data, [B, S, C] or [B, S, n_sample, C]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(
        view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Find the query ball index for each point.

    Parameters
    ----------
    radius : float
        Raidus for the query ball.

    nsample : int
        Max number of points in local region.

    xyz : torch.Tensor
        All points, [B, N, C]

    new_xyz : torch.Tensor
        query points, [B, S, C].

    Returns
    -------
    group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # shape: [B, S, (0, 1, 2...,N)],
    group_idx = torch.arange(N,
                             dtype=torch.long).to(device).view(1, 1, N).repeat(
        [B, S, 1])
    # the square distance from each query ball center to all points
    # (B, S, N)
    sqrdists = square_distance(new_xyz, xyz)
    # assigning the points out of radius for each centroid N
    group_idx[sqrdists > radius ** 2] = N
    # (B, S, nsample), find the points that are within each query ball. We
    # only randomly get the first nsample points for eqch query ball
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # assume the first point in each query is valid
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # there may be some invalid points in the query ball
    mask = group_idx == N
    # change the invalid points index to the first point index
    # for each query ball
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Combining sampling and grouping together. First find the centroids, then
    get the query ball for each centroid, and eventually compute the features
    for each query ball.

    Parameters
    ----------
    npoint: int
        The number of centroids.

    radius: float
        The radius for each qeury ball.

    nsample: int
        The local maximum number of points.

    xyz: torch.Tensor
        input points position data, [B, N, 3]

    points: torch.Tensor
        input points feature, [B, N, D]

    Return
    ------
    new_xyz: sampled points position data, [B, npoint, nsample, 3]
    new_points: sampled points feature, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    # centroids index [B, npoint]
    fps_idx = farthest_point_sample(xyz, npoint)
    # centroid's xyz [B, npoint, 3]
    new_xyz = index_points(xyz, fps_idx)
    # find the points index for each query, [B, S, nsample]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)

    # [B, S, nsample, C]
    grouped_xyz = index_points(xyz, idx)
    # get the relative xyz for each point in the query
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        # [B, npoint, nsample, C+D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points],
                               dim=-1)
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Simply cancate the xyz geometric information and features.
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    A basic single-scalemodule for pointnet++. It will first do fastest
    sampling + query ball grouping, and then extract features from the
    grouped points.
    """

    def __init__(self,
                 npoint: int,
                 radius: float,
                 nsample: int,
                 in_channel: int,
                 mlp: list,
                 group_all: bool):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius,
                                                   self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, nsample, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    Multi-scale point feature extractor.
    """
    def __init__(self,
                 npoint: int,
                 radius_list: list,
                 nsample_list: list,
                 in_channel: int,
                 mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        # loop each scale
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            # loop each scale's mlp layer
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []

        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            # [B, D, K, S]
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            # [B, D', S]
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """
    This is for segmentation only. Perform upsampling and concat with
    point features from encoder layers.
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            # the input should be B, C, N, so use conv1d
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: sampled points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


if __name__ == '__main__':
    xyz = torch.rand(32, 3, 5000)
    radius = 0.2
    nsample = 32
    ncentriod = 2056

    sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3,
                                    [[16, 16, 32], [32, 32, 64]])
    sa2 = PointNetSetAbstractionMsg(256,
                                    [0.1, 0.2, 0.4],
                                    [16, 32, 128],
                                    32+64,
                                    [[32, 32, 64], [64, 64, 128], [64, 96, 128]])

    l1_xyz, l1_points = sa1(xyz, xyz)
    l2_xyz, l2_points = sa2(l1_xyz, l1_points)

    sg = PointNetFeaturePropagation(32+64+64+128+128, [256, 256])
    l3_points = sg(l1_xyz, l2_xyz, l1_points, l2_points)
