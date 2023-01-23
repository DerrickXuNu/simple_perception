import torch.nn as nn
import torch
import torch.nn.functional as F
from models.lidar_only.pointnet2.basic_modules import \
    PointNetSetAbstractionMsg, PointNetSetAbstraction, \
    PointNetFeaturePropagation


class PointNet2ClsSSG(nn.Module):
    """
    PointNet++ for classification with single-scale scale grouping.
    """

    def __init__(self, num_class, normal_channel=True):
        super(PointNet2ClsSSG, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=512,
                                          radius=0.2,
                                          nsample=32,
                                          in_channel=in_channel,
                                          mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128,
                                          radius=0.4,
                                          nsample=64,
                                          in_channel=128 + 3,
                                          mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None,
                                          radius=None,
                                          nsample=None,
                                          in_channel=256 + 3,
                                          mlp=[256, 512, 1024],
                                          group_all=True)
        # classification header
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class PointNet2ClsMSG(nn.Module):
    """
    PointNet++ for classification with multi-scale scale grouping.
    """

    def __init__(self, num_class, normal_channel=True):
        super(PointNet2ClsMSG, self).__init__()

        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstractionMsg(512,
                                             [0.1, 0.2, 0.4],
                                             [16, 32, 128],
                                             in_channel,
                                             [[32, 32, 64],
                                              [64, 64, 128],
                                              [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128,
                                             [0.2, 0.4, 0.8],
                                             [32, 64, 128],
                                             320,
                                             [[64, 64, 128],
                                              [128, 128, 256],
                                              [128, 128, 256]])

        # the last layer just simply concat xyz and features
        self.sa3 = PointNetSetAbstraction(None,
                                          None,
                                          None,
                                          640 + 3,
                                          [256, 512, 1024],
                                          True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # sampling, grouping, feature extraction
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # classification
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class PointNet2SegMSG(nn.Module):
    """
    PointNet++ for segmentation with multi-scale scale grouping.
    """

    def __init__(self, num_classes):
        super(PointNet2SegMSG, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 0,
                                             [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32],
                                             32 + 64,
                                             [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32],
                                             128 + 128, [[128, 196, 256],
                                                         [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32],
                                             256 + 256, [[256, 256, 512],
                                                         [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256,
                                              [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


if __name__ == '__main__':
    # mocked lidar points (batch, (xyz), num of points)
    xyz = torch.rand(32, 3, 5000)
    # mocked num of clas
    num_class = 7

    # classification with single-scale grouping
    net = PointNet2ClsSSG(num_class, False)
    pred, _ = net(xyz)
    print(pred.shape)

    # Classification with multi-scale grouping
    net = PointNet2ClsMSG(num_class, False)
    pred, _ = net(xyz)
    print(pred.shape)

    # Segmentation with multi-scale grouping
    net = PointNet2SegMSG(num_class)
    seg, _ = net(xyz)
    print(seg.shape)

