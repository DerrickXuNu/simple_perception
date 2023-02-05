import torch
import torch.nn as nn

from loss import NMSFreeCoder


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
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True), #
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
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


