import copy
import torch
import torch.nn as nn
from functools import partial
from loss import NMSFreeCoder, HungarianAssigner3D, FocalLoss, L1Loss
from transformers import Uni3DDETR
from view_modules import Uni3DViewTrans
from common_modules import constant_init, bias_init_with_prob


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


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

def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


class UVTRHead(nn.Module):
    def __init__(self,
                 *args,
                 num_classes,  # the object classes
                 in_channels,  # input feature channels
                 num_query=100,  # number of object query
                 num_reg_fcs=2,
                 # Number of fully-connected layers used in  `FFN`, which is then used for the regression head. Default 2.
                 unified_conv=None,
                 sync_cls_avg_factor=False,
                 # Whether to sync the avg_factor of   all ranks. Default to False.
                 view_cfg=None,  # related to camera view projection
                 with_box_refine=False,
                 # whether to refine the bbx predictions after decoder
                 transformer=None,  # only contains decoder for object query
                 bbox_coder=None,  # NMSfree box coder for Hungarian match
                 num_cls_fcs=2,  # todo: seems not used
                 code_weights=None,
                 # for bounding box prediction, [x, y, z, w, h, l, rot_x, rot_y, vx, vy]
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
            outputs_class = self.cls_branches[lvl](hs[lvl]) \
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

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9)
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # == num of object query
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        # label targets
        # (num_query,)
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        # check each query's corresponding class
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # (num_query, pred_num) note: sometime gt and prediction
        # output dimensions are different
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        # only the query has match has weight
        bbox_weights[pos_inds] = 1.0

        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        bbox_targets[pos_inds] = pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, shape [bs, num_query, 10].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # formated label
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # num_query*bs
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight

        if self.sync_cls_avg_factor:
            cls_avg_factor = cls_scores.new_tensor([cls_avg_factor])

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # after te normalization, the gt becomes (num_gt, 9->10)
        # the addtional dimension is because split rotation to rot_cos, rot_sin
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                       :10], bbox_weights[isnotnan, :10])
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        num_imgs = len(cls_scores_list)
        # labels_list: [(num_query,]
        # label_weights: [(num_query,]
        # bbox_targets_list: [(num_query, 9)]
        # bbox_weights_list: [(num_query, 10)]
        # pos_inds_list: [(num_gt,)]
        # neg_inds_list: [(num_query-num_gt,)]
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[LiDARInstance3DBoxes]): Ground truth bboxes for each image
                [(num_gt_1, 9)]
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape [(n,)...].
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 10 D-tensor with
                    normalized coordinate format [xc, yc, w, l, zc, h, rot.sin(), rot.cos(), vx, vy] and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)

        # gt_bboxes_list shape:
        # [(n, 9)] -> [xc, yc, w, l, zc, h, rot, vx, vy]
        # when compute loss, we only compare the first 8 dimension for both
        # prediction and gt

        # copy the gt boxes for num_dec_layers
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        # loop the dec layers for all batches
        # [(batch_1_dec_1+batch2_dec_1+...), (batch_1_dec_2+batch2_dec_2, ...)]
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # list of dictionary that contains: bbox pred, pred labels, and score
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []

        for i in range(num_samples):
            preds = preds_dicts[i]
            # after denormalize:
            # cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy ->
            # cx, cy, cz, w, l, h, rot, vx, vy
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
