"""Attention modules in BEVFormer"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from geometry_utils import get_reference_points, point_sampling
from common_modules import xavier_init, constant_init, FFN


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, h*2, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,h*w, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,h*w, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_spatial_shapes = value_spatial_shapes.int()
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    # some nomralization
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        # find the feature value through sampling from sample_offset
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    # weighted sum through the attention weights
    # shape: (b, c, h*w)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable positional encoding added to BEV mask. For BEVFormer tiny,
    the mask (BEV pos) is 50 x 50 while for base version, it is 200 x 200
    """
    def __init__(self,
                 num_feats=128,
                 row_num_embed=50,
                 col_num_embed=50,
                 **kwargs):

        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        # mask: bev mask, (b, bev_h, bev_w)
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        # convert pre-defined grid to learnable position
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        # combine x and y direction-> (50, 128) + (50, 128) -> (b, 256, 50, 50)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)

        return pos


class TemporalSelfAttention(nn.Module):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. For BEVFormer miny, it is 1.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 **kwargs):

        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        # this is is used to get offsets for query for deformable attention
        # refer to DeformDeTR Figure2
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        # This is used to comupte the attention weights for query, refer to
        # DeformDETR figure2 'attention weights'
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        # add a linear projection to the value
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        # the LN layer
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True


    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): BEV query
                (bs, h*w, embed_dims).
            key (Tensor): Will be none if there is no history bev feature
            value (Tensor):  Will be none if there is no history bev feature
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): Bev_pos, shape: (bs, h*w, embed_dims)
            key_pos (Tensor): The same as query pos
            reference_points (Tensor):  reference points on 2D bev plane,
                shape: (bs*2, h*2, num_level, 2), I found num_level is always
                1 in BEVFormer tiny.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # this means there is no history bev feature
        if value is None:
            bs, len_bev, c = query.shape
            # (2*bs, h*w, c). Stack them because it consider one current
            # and one previous bev feature
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

        if identity is None:
            identity = query

        # add the learnable embedding bev_pos to bev query
        if query_pos is not None:
            query = query + query_pos

        # only consider current and one prev bev feature
        assert self.num_bev_queue == 2


        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape

        # shape (bs, h*w, 2c)
        # the first bs is actually from previous BEV query. So the query
        # becomes concat(prev_bev, current_bev)
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        # (2*bs, h*w, heads_num, c/heads_num)
        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)

        # compute the sampling offsets for query
        sampling_offsets = self.sampling_offsets(query)
        # (b, h*w, head, 2, 1, num_points, 2)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,
            self.num_bev_queue, self.num_levels, self.num_points, 2)
        # compute the attention weights for the query
        # (b, h*w, head, 2, num_points)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue,
            self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        # (b, h*w, head, 2, 1, num_points)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)
        # move the temporal dimension to batch dimension for deformable attention
        # ( b*2, h*2, head, 1, num_points)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        # ( b*2, h*2, head, 1, num_points, 2)
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        # 2d coordinate (x,y)
        if reference_points.shape[-1] == 2:
            # [[bev_h, bev_w]]
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            # the reference_points are normalized already. So we should
            # normalize the sampling locations as well
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        # todo: not sure about this
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5

        # apply deformable transformer basic operation
        # output is the weighted sum value from the sampling points from value
        # shape : (b*num_bev_queue, hw, c)
        output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)
        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        return self.dropout(output) + identity


class MSDeformableAttention3D(nn.Module):
    """The deformable attention module for spatial cross-attention
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention, i.e., N_ref, Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1):
        super().__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        # this will only be used for the built-in gpu version of deformable
        # attention, which will not used in our codebase
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        # deformable attention's sampling projection
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        # deformable attention
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = None
        # initial weights. we need special intialization for the sampling
        # offset
        self.init_weights()


    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True


    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs*# of cam, h*w, embed_dims).
            key (Tensor): The key tensor with shape (flattent image feature)
                `(bs*# of cam,  h*w,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs*# of cam,  h*w,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        # bs*cam, h*w, heads, heads_c
        value = value.view(bs, num_value, self.num_heads, -1)
        # bs*cam, H*W, heads, 1, D, 2
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        #  bs*cam, H*W, heads, D
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        # the weight for diiferent sampling points
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            # bs, num_query, 1, 1, 1, num_Z_anchors, 2
            reference_points = reference_points[:, :, None, None, None, :, :]

            # find the normalized pixel position for each sampling offsets
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = \
                sampling_offsets.shape

            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels,
                num_all_points // num_Z_anchors, num_Z_anchors, xy)

            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        return output


class SpatialCrossAttention(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 deformabble_attention=dict(
                     embed_dims=256,
                     num_levels=1),
                 **kwargs):
        super(SpatialCrossAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)
        # evaluation range
        self.pc_range = pc_range
        # embedding dimension
        self.embed_dims = embed_dims
        self.num_cams = num_cams

        # the deformable attention
        self.deformable_attention = MSDeformableAttention3D(**deformabble_attention)
        # used to refine the attention feature
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        # initial weights
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)


    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): BEV Query of Transformer with shape
                (bs, hw, embed_dims).
            key (Tensor): The flatten image features`
                `(num_cam, H*W, bs, embed_dims)`.
            value (Tensor): The flatten image features`
               `(num_cam, H*W, bs, embed_dims)`.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_points, h*w, 3),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            bev_mask (Tensor): (num_cam, b, h*w, num_points)
                This is used to mask the query pillar that does not has
                any projected points on camera images.
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        # num of sampling points
        D = reference_points_cam.size(3)

        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            # maske_per_img: bs, h*w, D
            # as long as there is one point among all D has corresponding image
            # pixels, we shouldn't ignore this bev query index
            # shape: (n,), it indicates all the n valid indexes
            # it is the same for all batches since the ego is always at the center
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            # loop each camera
            # (b, h*w, D, 4)
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                # make sparse query->dense query
                queries_rebatch[j, i, :len(index_query_per_img)] = query[
                    j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = \
                    reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape
        # (bs*num_cam, h*w, c)
        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # normalization for the new query
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

