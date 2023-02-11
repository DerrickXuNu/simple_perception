"""
The modules for camera view transformations
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_modules import xavier_init

def feature_sampling(mlvl_feats,
                     reference_voxel,
                     pc_range,
                     img_metas,
                     img_depth=None,
                     num_sweeps=1,  # temporal dimension
                     num_cam=6):
    # extrinsic, augmentation on voxel level
    lidar2img, uni_rot_aug, uni_trans_aug = [], [], []
    # image level augmentation
    img_rot_aug, img_trans_aug = [], []

    if not isinstance(img_metas, list):
        img_metas = [img_metas]

    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
        if 'uni_rot_aug' in img_meta:
            uni_rot_aug.append(img_meta['uni_rot_aug'])
        if 'uni_trans_aug' in img_meta:
            uni_trans_aug.append(img_meta['uni_trans_aug'])
        if 'img_rot_aug' in img_meta:
            img_rot_aug.append(img_meta['img_rot_aug'])
        if 'img_trans_aug' in img_meta:
            img_trans_aug.append(img_meta['img_trans_aug'])
    # b, num_cam, 1 ,4, 4
    lidar2img = np.asarray(lidar2img)
    if num_sweeps > 1:
        lidar2img = lidar2img[:, :, :num_sweeps]

    # convert numpy to tensor
    lidar2img = reference_voxel.new_tensor(lidar2img)

    # Transfer to Point cloud range with X,Y,Z
    # the voxels' grid now represent the real-world coordinate
    reference_voxel[..., 0:1] = reference_voxel[..., 0:1] * (
            pc_range[3] - pc_range[0]) + pc_range[0]
    reference_voxel[..., 1:2] = reference_voxel[..., 1:2] * (
            pc_range[4] - pc_range[1]) + pc_range[1]
    reference_voxel[..., 2:3] = reference_voxel[..., 2:3] * (
            pc_range[5] - pc_range[2]) + pc_range[2]

    # Conduct inverse voxel augmentation first
    # todo: not explore train yet
    if len(uni_rot_aug) > 0:
        reference_voxel = reference_voxel @ torch.inverse(uni_rot_aug)[:, None,
                                            None]
    reference_aug = reference_voxel.clone()
    # make it homogenous coordinate
    reference_voxel = torch.cat(
        (reference_voxel, torch.ones_like(reference_voxel[..., :1])), -1)
    # bs, h * w * z, 4
    reference_voxel = reference_voxel.flatten(1, 3)
    B, num_query = reference_voxel.size()[:2]

    # todo: this seems never come
    if num_sweeps > 1:
        reference_voxel = reference_voxel.view(B, 1, 1, num_query, 4).repeat(1,
                                                                             num_cam,
                                                                             num_sweeps,
                                                                             1,
                                                                             1)
        reference_voxel = reference_voxel.reshape(B, num_cam * num_sweeps,
                                                  num_query, 4, 1)
        lidar2img = lidar2img.view(B, num_cam, num_sweeps, 1, 4, 4)
        lidar2img = lidar2img.reshape(B, num_cam * num_sweeps, 1, 4, 4)

    else:
        # b, num_cam, num_query, 4, 1
        reference_voxel = \
            reference_voxel.view(B, 1, num_query, 4).repeat(1, num_cam, 1,
                                                            1).unsqueeze(-1)
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)
    # project the voxels in lidar space to camera space
    # b, num_cam, num_query, 4
    reference_voxel_cam = torch.matmul(lidar2img, reference_voxel).squeeze(-1)
    # depth
    referenece_depth = reference_voxel_cam[..., 2:3].clone()
    mask = (referenece_depth > 1e-5)
    # u , v = x/z, y/z, get the 2d coordinates on image
    reference_voxel_cam = reference_voxel_cam[..., 0:2] / torch.maximum(
        reference_voxel_cam[..., 2:3],
        torch.ones_like(reference_voxel_cam[..., 2:3]) * 1e-5)

    # transfer if have image-level augmentation
    # todo: not explore train yet
    if len(img_rot_aug) > 0:
        img_rot_aug = torch.stack(img_rot_aug, dim=0).to(reference_voxel_cam)
        reference_voxel_cam = reference_voxel_cam @ img_rot_aug
    if len(img_trans_aug) > 0:
        img_trans_aug = torch.stack(img_trans_aug, dim=0).to(
            reference_voxel_cam)
        reference_voxel_cam = reference_voxel_cam + img_trans_aug[:, None,
                                                    None]

    # normalize the coordinate by the image shape
    reference_voxel_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_voxel_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    # F.grid sampling take -1, 1 as corners and 0, 0 center
    reference_voxel_cam = (reference_voxel_cam - 0.5) * 2

    # normalize depth
    if isinstance(img_depth, list):
        depth_dim = img_depth[0].shape[1]
    else:
        depth_dim = img_depth.shape[1]

    # sampling the depth
    referenece_depth /= depth_dim
    # make the depth center to be 0
    referenece_depth = (referenece_depth - 0.5) * 2

    mask = (mask & (reference_voxel_cam[..., 0:1] > -1.0)
            & (reference_voxel_cam[..., 0:1] < 1.0)
            & (reference_voxel_cam[..., 1:2] > -1.0)
            & (reference_voxel_cam[..., 1:2] < 1.0)
            & (referenece_depth > -1.0)
            & (referenece_depth < 1.0))

    mask = mask.view(B, num_cam * num_sweeps, 1, num_query, 1, 1).permute(0, 2,
                                                                          3, 1,
                                                                          4, 5)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        # B*N, num_query, 1, 2
        reference_points_cam_lvl = reference_voxel_cam.view(B * N, num_query,
                                                            1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        # bs, c, num_query, num_cam, 1
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2,
                                                                        3, 1,
                                                                        4)
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam * num_sweeps,
                                       1, len(mlvl_feats))

    # sample depth
    reference_points_cam = \
        torch.cat([reference_voxel_cam, referenece_depth], dim=-1)
    reference_points_cam = reference_points_cam.view(B * num_cam * num_sweeps,
                                                     1, num_query, 1, 3)
    if isinstance(img_depth, list):
        sampled_depth = []
        for lvl, depth in enumerate(img_depth):
            # num_cam * bs * len_sequence, 1, c, h, w
            depth = depth.unsqueeze(1)
            depth = F.grid_sample(depth, reference_points_cam)
            depth = depth.view(B, num_cam * num_sweeps, 1, num_query,
                               1).permute(0, 2, 3, 1, 4)
            sampled_depth.append(depth)
        # bs, 1, num_query, num_cam, 1, num_level
        sampled_depth = torch.stack(sampled_depth, -1)

    else:
        img_depth = img_depth.unsqueeze(1)
        sampled_depth = F.grid_sample(img_depth, reference_points_cam)
        sampled_depth = sampled_depth.view(B, num_cam * num_sweeps, 1,
                                           num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_depth = sampled_depth.unsqueeze(-1)

    return sampled_feats, sampled_depth, mask, reference_aug


class Uni3DViewTrans(nn.Module):
    """Implements the view transformer. It projects camera features to voxel space
    """

    def __init__(self,
                 num_cams=6,
                 num_convs=3,
                 num_points=5,
                 num_feature_levels=4,
                 num_sweeps=1,
                 cam_sweep_feq=12,
                 kernel_size=(1, 3, 3),
                 sweep_fusion=dict(type=""),
                 keep_sweep_dim=False,
                 embed_dims=128,
                 norm_cfg=None,
                 use_for_distill=False,
                 **kwargs):
        super(Uni3DViewTrans, self).__init__()

        self.conv_layer = []
        self.num_sweeps = num_sweeps
        self.sweep_fusion = sweep_fusion.get("type", "")
        self.keep_sweep_dim = keep_sweep_dim
        self.use_for_distill = use_for_distill
        self.num_cams = num_cams

        norm_op = nn.BatchNorm3d
        self.depth_proj = Uni3DDepthProj(embed_dims=embed_dims,
                                         num_levels=num_feature_levels,
                                         num_points=num_points,
                                         num_cams=num_cams,
                                         num_sweeps=num_sweeps,
                                         **kwargs)
        padding = tuple([(_k - 1) // 2 for _k in kernel_size])

        for k in range(num_convs):
            # this is for multi-frame fusion
            if "sepconv" in self.sweep_fusion:
                sep_kernel = num_points if 'sep_kernel' not in sweep_fusion else \
                    sweep_fusion['sep_kernel']
                conv = nn.Sequential(
                    nn.Conv3d(embed_dims,
                              embed_dims,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=True),
                    nn.Conv3d(embed_dims,
                              embed_dims,
                              kernel_size=(sep_kernel, 1, 1),
                              padding=((sep_kernel - 1) // 2, 0, 0),
                              stride=1),
                    norm_op(embed_dims) if "GN" not in norm_cfg[
                        "type"] else norm_op(norm_cfg["num_groups"],
                                             embed_dims),
                    nn.ReLU(inplace=True))
            else:
                conv = nn.Sequential(
                    nn.Conv3d(embed_dims,
                              embed_dims,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=True),
                    nn.BatchNorm3d(embed_dims),
                    nn.ReLU(inplace=True))
            # conv_trans_head_x
            self.add_module("{}_head_{}".format('conv_trans', k + 1), conv)
            self.conv_layer.append(conv)

        # only for multi frames
        if "sweep_cat" in self.sweep_fusion:
            self.trans_conv = nn.Sequential(
                nn.Conv3d(embed_dims * self.num_sweeps,
                          embed_dims,
                          kernel_size=1,
                          padding=0,
                          stride=1),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(inplace=True))

        if "with_time" in self.sweep_fusion:
            self.cam_sweep_time = 1.0 / cam_sweep_feq
            self.time_conv = nn.Sequential(
                nn.Conv3d(embed_dims + 1,
                          embed_dims,
                          kernel_size=1,
                          padding=0,
                          stride=1),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(inplace=True))

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        for layer in self.conv_layer:
            xavier_init(layer, distribution='uniform', bias=0.)

    def forward(self, mlvl_feats, **kwargs):
        """Forward function for `Uni3DViewTrans`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
        """
        # Sweep number and Cam number could be dynamic
        if self.num_sweeps > 1:
            num_sweep, num_cam = kwargs['img_metas'][0]['sweeps_ids'].shape

        else:
            num_sweep = self.num_sweeps
            num_cam = self.num_cams

        kwargs['num_sweep'] = num_sweep
        kwargs['num_cam'] = num_cam
        kwargs['batch_size'] = len(mlvl_feats[0])
        # project image features to voxel space
        voxel_space = self.depth_proj(mlvl_feats,
                                      img_depth=kwargs.pop('img_depth'),
                                      **kwargs)
        voxel_space = self.feat_encoding(voxel_space, **kwargs)

        return voxel_space

    def feat_encoding(self, voxel_space, **kwargs):
        num_sweep = kwargs["num_sweep"]

        # todo: not explored the time dimension yet
        if "with_time" in self.sweep_fusion:
            sweep_time = torch.stack([torch.from_numpy(_meta['sweeps_ids']) for _meta in kwargs['img_metas']], dim=0)
            sweep_time = self.cam_sweep_time * sweep_time[..., 0].repeat(len(voxel_space)//num_sweep, 1).to(device=voxel_space.device)
            sweep_time = sweep_time.reshape(-1, 1, 1, 1, 1).repeat(1,1,*voxel_space.shape[-3:])
            voxel_space = torch.cat([voxel_space, sweep_time], dim=1)
            voxel_space = self.time_conv(voxel_space)

        if 'sweep_sum' in self.sweep_fusion:
            voxel_space = voxel_space.reshape(-1, num_sweep, *voxel_space.shape[1:])
            voxel_space = voxel_space.sum(1)
            num_sweep = 1
        elif 'sweep_cat' in self.sweep_fusion:
            voxel_space = voxel_space.reshape(-1, num_sweep*voxel_space.shape[1], *voxel_space.shape[2:])
            voxel_space = self.trans_conv(voxel_space)
            num_sweep = 1


        # used for distill
        out_before_relu = []
        # refine the voxel features
        for _idx, layer in enumerate(self.conv_layer):
            if self.use_for_distill:
                out_mid = layer[:-1](voxel_space)
                out_before_relu.append(out_mid.clone())
                voxel_space = layer[-1](out_mid)
            else:
                voxel_space = layer(voxel_space)

        if self.keep_sweep_dim:
            # shape: (N, L, C, D, H, W)
            voxel_space = voxel_space.reshape(-1, num_sweep, *voxel_space.shape[1:])

        if self.use_for_distill:
            voxel_space = {'final': voxel_space,
                           'before_relu': out_before_relu}

        return voxel_space


class Uni3DDepthProj(nn.Module):
    """Depth project module used in UVTR.
    """

    def __init__(self,
                 embed_dims=256,
                 num_levels=4,  # multi-scale features
                 num_points=5,  # z value
                 num_cams=6,  # number of cameras
                 num_sweeps=1,  # lengh of sequence
                 pc_range=None,  # the camera bev range
                 voxel_shape=None,  # voxelization resolution
                 device='cuda',
                 **kwargs):
        super(Uni3DDepthProj, self).__init__()

        self.device = device
        self.pc_range = pc_range
        self.voxel_shape = voxel_shape
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.num_sweeps = num_sweeps

        # build voxel space with X,Y,Z
        _width = torch.linspace(0, 1, self.voxel_shape[0])
        _hight = torch.linspace(0, 1, self.voxel_shape[1])
        _depth = torch.linspace(0, 1, self.voxel_shape[2])
        # h, w, depth_sampling, 3
        self.reference_voxel = torch.stack(
            torch.meshgrid([_width, _hight, _depth]), dim=-1)

    def forward(self, mlvl_feats, img_depth=None, **kwargs):
        # mlvl_feats: [(b, num_cam, c, h, w), ...], image features from resnet
        # img_depth: [(b, num_cam, d, h, w), ...], image features from resnet
        bs = kwargs.get('batch_size', 1)
        num_sweep = kwargs.get('num_sweep', 1)
        num_cam = kwargs.get('num_cam', 6)
        device = mlvl_feats[0].device
        self.reference_voxel = self.reference_voxel.to(device)

        # bs, h, w, depth_sampling, 3
        reference_voxel = self.reference_voxel.unsqueeze(0).repeat(bs, 1, 1, 1,
                                                                   1)
        output, depth, mask, reference_voxel = feature_sampling(mlvl_feats,
                                                                reference_voxel,
                                                                self.pc_range,
                                                                kwargs[
                                                                    'img_metas'],
                                                                img_depth,
                                                                num_sweep,
                                                                num_cam)
        # bs, c, h*w, num_points on pillar, 1, num_level
        output = output.reshape(*output.shape[:2], -1, self.num_points,
                                *output.shape[-3:])
        mask = mask.reshape(*mask.shape[:2], -1, self.num_points,
                            *mask.shape[-3:])
        depth = depth.reshape(*depth.shape[:2], -1, self.num_points,
                              *depth.shape[-3:])
        depth = depth * mask
        output = output * depth

        if self.num_sweeps == 1:
            # sum up along the scale + num_cam dimension
            output = output.view(*output.shape[:4], -1).sum(-1)
            # shape: (N, C, W, H, D)
            output = output.reshape(*output.shape[:2], *self.voxel_shape)
            # reshape to (N, C, D, H, W)
            output = output.permute(0, 1, 4, 3, 2)
        else:
            # todo: not explre multi frames yet
            output = output.reshape(*output.shape[:4], num_cam, num_sweep, -1)
            output = output.transpose(-2, -1).reshape(*output.shape[:4], -1,
                                                      num_sweep).sum(-2)
            # shape: (N, C, W, H, D, S)
            output = output.reshape(*output.shape[:2], *self.voxel_shape,
                                    num_sweep)
            # permute to (N, S, C, D, H, W)
            output = output.permute(0, 5, 1, 4, 3, 2)
            # reshape to (N*S, C, D, H, W)
            output = output.reshape(-1, *output.shape[2:])

        return output.contiguous()
