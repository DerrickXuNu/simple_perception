import torch
import torch.nn as nn
import torch.nn.functional as F

from common_modules import xavier_init, constant_init


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttention, self).__init__()
        if 'dropout' in kwargs:
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Identity()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class UniCrossAtten(nn.Module):
    """
    Cross attention module in UVTR. I deleted some unnecessary params that
    are never used
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_sweeps=1,
                 num_points=1,
                 dropout=0.1,
                 batch_first=False,
                 **kwargs):
        super(UniCrossAtten, self).__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')

        self.dropout = nn.Dropout(dropout)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_sweeps = num_sweeps

        self.attention_weights = nn.Linear(embed_dims, num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
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
                level_start_index=None,
                **kwargs):
        """Forward Function of UniCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        # ATTENTION: reference_points is decoupled from sigmoid function!
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            # query pos is obtained from query embedding
            query = query + query_pos

        pts_value = value['pts_value']
        img_value = value['img_value']
        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        # change to (bs, num_query, num_points)
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.sigmoid()
        # normalize X, Y, Z to [-1,1]
        # #bs, num_query, 3
        reference_points_voxel = (reference_points.sigmoid() - 0.5) * 2

        embed_uni = []
        # shape: (N, L, C, D, H, W)
        if img_value is not None:
            # b*level, C, D, H, W
            img_value = img_value.view(-1, *img_value.shape[2:])
            reference_points_voxel = reference_points_voxel.view(-1, 1, 1,
                                                                 *reference_points_voxel.shape[
                                                                  -2:])
            # (bs, level, 1, 900, 3) note: I perseonally think the 3rd dimension should be num of points instead of 1
            # it seems to be hard-coded here
            reference_points_voxel = reference_points_voxel.repeat(1,
                                                                   len(img_value) // len(
                                                                       query),
                                                                   1, 1, 1)
            # without height
            if len(img_value.shape) == 4:
                # sample image feature in bev space
                embed_img = F.grid_sample(img_value,
                                          reference_points_voxel.reshape(-1, *reference_points_voxel.shape[-3:])[...,:2])
            else:
                # sample image feature in voxel space
                embed_img = F.grid_sample(img_value,
                                          reference_points_voxel.reshape(-1, 1,
                                                                         *reference_points_voxel.shape[
                                                                          -3:]))
            # shape: bs, level, c, num of query
            embed_img = embed_img.reshape(len(query), -1, embed_img.shape[1],
                                          embed_img.shape[-1])
            embed_img = embed_img.permute(0, 3, 2, 1)
            embed_uni.append(embed_img)

        # shape: (N, C, D, H, W)
        # todo: not explored yet
        if pts_value is not None:
            # without height
            if len(pts_value.shape) == 4:
                reference_points_voxel = reference_points_voxel.view(-1,1,*reference_points_voxel.shape[-2:])[...,:2]
            else:
                pts_value = pts_value.view(-1, *pts_value.shape[2:])
                reference_points_voxel = reference_points_voxel.view(-1, 1, 1, *reference_points_voxel.shape[-2:])
            # sample image feature in voxel space
            embed_pts = F.grid_sample(pts_value, reference_points_voxel)
            embed_pts = embed_pts.reshape(len(query), -1, embed_pts.shape[1], embed_pts.shape[-1])
            embed_pts = embed_pts.permute(0, 3, 2, 1)
            embed_uni.append(embed_pts)

        # concat embeddings different modalities
        embed_uni = torch.cat(embed_uni, dim=-1)
        # bs, num of query, c, number of points
        output = embed_uni * attention_weights.unsqueeze(-2)
        # bs, num of query, c
        output = output.sum(-1)

        output = output.permute(1, 0, 2)
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(reference_points).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat