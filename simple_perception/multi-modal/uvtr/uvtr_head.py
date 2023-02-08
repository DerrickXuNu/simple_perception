import copy
import torch
import torch.nn as nn

from loss import NMSFreeCoder, HungarianAssigner3D, FocalLoss, L1Loss
from transformers import Uni3DDETR
from view_modules import Uni3DViewTrans
from common_modules import constant_init, bias_init_with_prob

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class UVTRHead(nn.Module):
    def __init__(self,
                 *args,
                 num_classes, # the object classes
                 in_channels, # input feature channels
                 num_query=100, # number of object query
                 num_reg_fcs=2, #  Number of fully-connected layers used in  `FFN`, which is then used for the regression head. Default 2.
                 unified_conv=None,
                 sync_cls_avg_factor=False, # Whether to sync the avg_factor of   all ranks. Default to False.
                 view_cfg=None, # related to camera view projection
                 with_box_refine=False, # whether to refine the bbx predictions after decoder
                 transformer=None, # only contains decoder for object query
                 bbox_coder=None, # NMSfree box coder for Hungarian match
                 num_cls_fcs=2, # todo: seems not used
                 code_weights=None, # for bounding box prediction, [x, y, z, w, h, l, rot_x, rot_y, vx, vy]
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 **kwargs):
        super(UVTRHead, self).__init__()

        self.with_box_refine = with_box_refine
        self.code_size = 10
        self.code_weights = code_weights

        self.bbox_coder = NMSFreeCoder(**bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            # HungarianAssigner for training
            assigner = train_cfg['assigner']
            self.assigner = HungarianAssigner3D(**assigner)

        # the object query number
        self.num_query = num_query
        # default 10
        self.num_classes = num_classes
        self.in_channels = in_channels
        # regression layer number
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # notice: I simplified the cost functions
        # the original implemation is FocalLoss, L1Loss, and GIoU Loss,
        # which could have better performance
        self.loss_cls = FocalLoss(**loss_cls)
        self.loss_bbox = L1Loss(**loss_bbox)

        self.cls_out_channels = num_classes
        self.activate = nn.ReLU(inplace=True)
        self.transformer = Uni3DDETR(**transformer)
        self.embed_dims = self.transformer.embed_dims

        self._init_layers()

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        # this is used when both lidar and camera are used
        self.unified_conv = unified_conv
        if view_cfg is not None:
            self.view_trans = Uni3DViewTrans(**view_cfg)

        if self.unified_conv is not None:
            self.conv_layer = []
            for k in range(self.unified_conv['num_conv']):
                conv = nn.Sequential(
                            nn.Conv3d(view_cfg['embed_dims'],
                                    view_cfg['embed_dims'],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True),
                            nn.BatchNorm3d(view_cfg['embed_dims']),
                            nn.ReLU(inplace=True))
                self.add_module("{}_head_{}".format('conv_trans', k + 1), conv)
                self.conv_layer.append(conv)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, pts_feats, img_feats, img_metas, img_depth):
        """Forward function.
        Args:
            pts_feats: pointcloud features
            img_feats: [(bs, num_camera, c, h, w], xxx], where the length
                       of the list is the feature level
            img_metas: mainly use the lidar2img attribute
            img_depth: [(bs* num_camera, d, h, w], xxx], where the length
                       of the list is the feature level
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        query_embeds = self.query_embedding.weight
        with_image, with_point = True, True

        # if only lidar is used
        if img_feats is None:
            with_image = False
        elif isinstance(img_feats, dict) and img_feats['key'] is None:
            with_image = False

        # if only camera is used
        if pts_feats is None:
            with_point = False
        elif isinstance(pts_feats, dict) and pts_feats['key'] is None:
            with_point = False
            pts_feats = None

        # transfer image features to voxel level
        if with_image:
            img_feats = self.view_trans(img_feats, img_metas=img_metas,
                                        img_depth=img_depth)

        # shape: (N, L, C, D, H, W) todo: not explored yet
        if with_point:
            if len(pts_feats.shape) == 5:
                pts_feats = pts_feats.unsqueeze(1)

        # merge camera and lidar them using conv
        if self.unified_conv is not None:
            raw_shape = pts_feats.shape
            unified_feats = pts_feats.flatten(1, 2) + img_feats.flatten(1, 2)
            for layer in self.conv_layer:
                unified_feats = layer(unified_feats)
            img_feats = unified_feats.reshape(*raw_shape)
            pts_feats = None

        hs, init_reference, inter_references = self.transformer(
            pts_feats,
            img_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            # noqa:E501
            img_metas=img_metas,
        )
        # num_decode_layers. bs, num_query, c
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # bs, num_query, num_class
            outputs_class = self.cls_branches[lvl](hs[lvl])\
            # bs, num_query, 10
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # transfer to lidar system
            tmp[..., 0:1] = (
                        tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) +
                        self.pc_range[0])
            tmp[..., 1:2] = (
                        tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) +
                        self.pc_range[1])
            tmp[..., 4:5] = (
                        tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) +
                        self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # num_decod_layer, bs, num_query, num_class
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs