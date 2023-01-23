"""
This include the high-level transformer design.
"""
from functools import partial
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.nn.init import normal_

from torchvision.transforms.functional import rotate
from encoders import BEVFormerEncoder
from decoders import DetectionTransformerDecoder, CustomMSDeformableAttention, \
    inverse_sigmoid
from attention_modules import MSDeformableAttention3D, TemporalSelfAttention, \
    xavier_init, LearnedPositionalEncoding
from loss import HungarianAssigner3D, normalize_bbox, denormalize_bbox, \
    FocalLoss, L1Loss, NMSFreeCoder
from mocked_img_meta import lidar2img, can_bus


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


class PerceptionTransformer(nn.Module):
    """ This includes both deformable encoder and decoder for bevformer.

    """

    def __init__(self,
                 num_feature_levels=4,  # multi-scale feature levels from FPN
                 num_cams=6,
                 two_stage_num_proposals=300,  # this is not used at all
                 encoder=None,  # the BEV encoder configuration, dict
                 decoder=None,  # the BEV decoder configuration, dict
                 embed_dims=256,  # the embedding size
                 rotate_prev_bev=True,
                 # rotate the bev from previous timestamp based on the rotation angle to align to the same coordinate
                 use_shift=True,
                 # x,y transplation to move the bev features from different timestamp to the same coordinate system
                 use_can_bus=True,
                 # whether use the can_bus to project to a high-dimension data
                 can_bus_norm=True,  # whether do normalization on can_bus
                 use_cams_embeds=True,  # todo:
                 rotate_center=[100, 100],
                 # the rotation center's position in the bev pos
                 **kwargs):

        super(PerceptionTransformer, self).__init__()
        self.encoder = BEVFormerEncoder(**encoder)
        self.decoder = DetectionTransformerDecoder(**decoder)

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        # initizalize the NN layers
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m,
                                                                    TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    def get_bev_features(
            self,
            mlvl_feats,  # multi-scale image features
            bev_queries,  # bev embedding
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],  # m/pixels
            bev_pos=None,  # bev position embedding
            prev_bev=None,  # bev query from last seconds
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        # (h*w, embed_dim) -> (h*w, bs, embed_dim)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        # (bs, h, w, c) -? (bs, hw, c)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                            for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                            for each in kwargs['img_metas']])
        # loop the batch
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in
             kwargs['img_metas']])

        # the followings are used to
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle

        # normalized shift in bev space
        shift_y = translation_length * \
                  np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
                  np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        # the shift will be applied to the sampling offset to align
        # the bev to the same coordinate in the current timestamp
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        # if there is bev query from previous frames
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # the angle is ego2global, so we rotate all of them
                    # to the same bev
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    # (c, h, w)
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    # rotate to the current frame.
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)

                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        # bs, 18
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        # bs, c
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        # this may not be useful from my view
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        # multi-scale level feature
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # num_cam, bs, h*w, c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(
                    feat.dtype)
            # feature level embedding add to feature
            feat = feat + self.level_embeds[None,
                          None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        # this is used to know the feature's level as they are concatenated
        # at the h*w dimension
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # (num_cam, H*W, bs, embed_dims)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        # temporal and spatial attetion iteratively applied
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs)

        return bev_embed

    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        bs = mlvl_feats[0].size(0)
        # num_query, embed_dim
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        # bs, num_query, embed_dim
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        # initialized reference points
        # bs, num_query, 3
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # inter_states: (n_decoder_iters, num_query, bs, embed)
        # the object query embedding after each decoder iteration
        # inter_references: (n_decoder_iters, bs, num_query, 3)
        # the 3D reference points for each object query
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)
        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


class BEVFormerHead(nn.Module):
    """ Include BEV embedding/pos initizer + PerceptionTransformer +
        cls/regress head + bounding box decoder + loss function.
        I merge the detrhead
        with bevformer head together.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
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
                 test_cfg=dict(max_per_img=100),
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 **kwargs):
        super(BEVFormerHead, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            # [预测框中心位置的x方向偏移，预测框中心位置的y方向偏移，预测框的宽，预测框的长，
            # 预测框中心位置的z方向偏移，预测框的高，旋转角的正弦值，旋转角的余弦值，x方向速度，y方向速度]
            self.code_size = 10

        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = NMSFreeCoder(**bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        # the BEV perception range in meters
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        # build loss function, cls/reg head, transformer
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            # HungarianAssigner for training
            assigner = train_cfg['assigner']
            self.assigner = HungarianAssigner3D(**assigner)
            sampler_cfg = dict(type='PseudoSampler')
            # I delete the sampler and directly use the useful functions
            # self.sampler = PseudoSampler(**sampler_cfg)
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
        self.positional_encoding = \
            LearnedPositionalEncoding(**positional_encoding)

        self.transformer = PerceptionTransformer(**transformer)
        self.embed_dims = self.transformer.embed_dims

        self._init_layers()

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
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            # copy the classification and regression branches N times
            # as the decoder will return N intermediate results
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        # the initialized BEV query and object query embedding
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)
        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)

    @staticmethod
    def bias_init_with_prob(prior_prob):
        """initialize conv/fc bias value according to a given probability value."""
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        return bias_init

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        bias_init = self.bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        # this is used to mask the pillars that do not have any valid projected
        # 2D points on any image
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        # this will be added to the bev geometry grid
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # this is used when computing previous bev features
        if only_bev:
            # the transformer just have encoder
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            # this will let the transformer go through encoder + decoder
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

        # bev_embed: the BEV query after encoder
        # hs:  (n_decoder_iters, num_query, bs, embed), the object query embedding after each decoder iteration
        # init_reference:  (bs, num_query, 3) The initialized reference points before decoder, which is
        # directly retrieved from linear projection + sigmoid on the object query
        # inter_references: (n_decoder_iters, bs, num_query, 3) the 3D reference points for each object query after decoder
        bev_embed, hs, init_reference, inter_references = outputs
        # n_cam, bs, # of query, embed_dim
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            # bs, # of query, num_class
            outputs_class = self.cls_branches[lvl](hs[lvl])
            # bs, # of query, 10
            # [xc, yc, w, l, zc, h, rot.sin(), rot.cos(), vx, vy]
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            # x, y position
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            # z position (center)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # from normalized scale to the real meters
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                              self.pc_range[0]) +
                             self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                              self.pc_range[1]) +
                             self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                              self.pc_range[2]) +
                             self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        # (num_decoder_iter, bs, num_query, num_class)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
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
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

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
