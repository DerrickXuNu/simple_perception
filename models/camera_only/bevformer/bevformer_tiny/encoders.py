import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_modules import TemporalSelfAttention, SpatialCrossAttention
from common_modules import FFN
from geometry_utils import get_reference_points, point_sampling


class BEVFormerLayer(nn.Module):
    def __init__(self,
                 attn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 ffn_cfgs=dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),

                 ),
                 **kwargs):
        """
        Each BEVFormer layer contains a temporal-attention, a normalization,
        a spatial cross-attention layer and a FFN.
        """
        super().__init__()
        # check how many attention modules in the layer
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()

        # we need to revise the default ffn config based on the new arguments
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                ffn_cfgs[new_name] = kwargs[ori_name]

        index = 0
        for operation_name in operation_order:
            if operation_name == 'self_attn':
                attention = TemporalSelfAttention(**attn_cfgs[index])
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1
            if operation_name == 'cross_attn':
                attention = SpatialCrossAttention(**attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims
        # FFNs
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        # FFN configuration are the same for all
        ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]

        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims

            self.ffns.append(FFN(**ffn_cfgs[ffn_index]))

        # normalizations
        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(nn.LayerNorm(self.embed_dims))

        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [bs, num_queries, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): (num_cam, H*W, bs, embed_dims) image feature
            value (Tensor): (num_cam, H*W, bs, embed_dims)
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        # the index indicate different operations
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spatial cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class BEVFormerEncoder(nn.Module):
    def __init__(self, transformerlayers, num_layers,
                 return_intermediate=False,
                 pc_range=None, num_points_in_pillar=4, **kwargs):
        # transformerlayers: a list of dictionary containing
        # all BEVFormerLayer configs
        super(BEVFormerEncoder, self).__init__()
        self.return_intermediate = return_intermediate

        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(BEVFormerLayer(**transformerlayers[i]))

        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate

    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        intermediate = []

        # (bs, num_points_in_pillar, h*w, 3)
        # the BEV grid coordinates in 3D space
        # each (x,y) will sampling # points in pillar in the Z axis
        ref_3d = get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),
            device=bev_query.device, dtype=bev_query.dtype)

        # (bs, h*w, 1, 2)
        # the BEV grid corrdinates in 2D space. The z axis is ignored
        ref_2d = get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1),
            device=bev_query.device, dtype=bev_query.dtype)

        # reference_points_cam: (num_cam, bs, h*w, # points in pillar, 2)
        # This is the corresponding coordinate in camera image space
        # for each BEV query pillar
        # bev_mask: (num_cam, bs, h*w, 4) The mask is to indicate whether
        # certain query pillar has valid projected camera coordinates
        reference_points_cam, bev_mask = point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        # the shift is for temporal fusion. It will shift all BEV features
        # to the same coordinate system
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        # a learnable embedding to be added to query
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        if prev_bev is not None:
            # when there is already computed bev feature from last timestamp
            # bs, h*w, c
            prev_bev = prev_bev.permute(1, 0, 2)
            # bs*2, hw, c
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
            # the shift_ref_2d is nearly the same as ref_2d as it is really
            # small
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


if __name__ == '__main__':
    # unit test
    encoder_config = {'type': 'BEVFormerEncoder', 'num_layers': 3,
                      'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                      'num_points_in_pillar': 4, 'return_intermediate': False,
                      'transformerlayers': {'type': 'BEVFormerLayer',
                                            'attn_cfgs': [
                                                {
                                                    'type': 'TemporalSelfAttention',
                                                    'embed_dims': 256,
                                                    'num_levels': 1},
                                                {
                                                    'type': 'SpatialCrossAttention',
                                                    'pc_range': [-51.2, -51.2,
                                                                 -5.0, 51.2,
                                                                 51.2, 3.0],
                                                    'deformable_attention': {
                                                        'type': 'MSDeformableAttention3D',
                                                        'embed_dims': 256,
                                                        'num_points': 8,
                                                        'num_levels': 1},
                                                    'embed_dims': 256}],
                                            'feedforward_channels': 512,
                                            'ffn_dropout': 0.1,
                                            'operation_order': (
                                                'self_attn', 'norm',
                                                'cross_attn', 'norm', 'ffn',
                                                'norm')}}
    encoder = BEVFormerEncoder(**encoder_config)
    print('passed')

    bev_queries = torch.rand(2500, 1, 256)
    feat_flatten = torch.rand(6, 375, 1, 256)
    bev_h = 50
    bev_w = 50
    bev_pos = torch.rand(2500, 1, 256)
    spatial_shapes = torch.Tensor([[15, 25]])
    level_start_index = 0
    prev_bev = None
    shift = torch.Tensor([[0., 0.]])

    lidar2img = [np.array([[6.21263804e+02, 4.20619097e+02, 1.75501092e+01,
                            -1.58917754e+02],
                           [-8.49768101e+00, 2.69007904e+02, -6.12510490e+02,
                            -3.11435602e+02],
                           [-1.22683565e-02, 9.98410605e-01, 5.50068379e-02,
                            -3.88263580e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            1.00000000e+00]]),
                 np.array([[6.82637282e+02, -3.09298948e+02, -1.97312124e+01,
                            -2.38504762e+02],
                           [1.90286945e+02, 1.60754178e+02, -6.19571483e+02,
                            -3.39648941e+02],
                           [8.43124328e-01, 5.36676561e-01, 3.34609709e-02,
                            -5.89521056e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            1.00000000e+00]]),
                 np.array([[1.57166902e+01, 7.51541527e+02, 3.96837713e+01,
                            -1.20451140e+02],
                           [-1.93894320e+02, 1.60813695e+02, -6.18813125e+02,
                            -3.32763074e+02],
                           [-8.23783500e-01, 5.65440018e-01, 4.07226190e-02,
                            -5.10620747e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            1.00000000e+00]]),
                 np.array([[-4.01872037e+02, -4.25465256e+02, -1.35112723e+01,
                            -4.46751496e+02],
                           [-5.18454677e+00, -2.22522690e+02, -4.07517382e+02,
                            -3.60091437e+02],
                           [-8.05056156e-03, -9.99190905e-01, -3.94046079e-02,
                            -1.04151481e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            1.00000000e+00]]),
                 np.array([[-5.93347201e+02, 4.61549460e+02, 2.66114321e+01,
                            -3.03441280e+02],
                           [-2.31253960e+02, -5.12626076e+01, -6.26260928e+02,
                            -2.82746456e+02],
                           [-9.47544132e-01, -3.19610210e-01, 3.07117610e-03,
                            -4.39541723e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            1.00000000e+00]]),
                 np.array([[1.42927147e+02, -7.34563027e+02, -3.00422588e+01,
                            -1.57191915e+02],
                           [2.22884337e+02, -6.10021376e+01, -6.25052672e+02,
                            -2.95054182e+02],
                           [9.24216363e-01, -3.81856010e-01, -3.17844874e-03,
                            -4.72933290e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            1.00000000e+00]])]
    # can_bus:
    #     can_bus[:3] = translation -- regarding the global frame origin
    #     can_bus[3:7] = rotation -- regarding to the global frame origin in quaternion
    #     patch_angle = quaternion_yaw(rotation) / np.pi * 180 -- convert the quaternion to degree
    #     can_bus[-2] = patch_angle / 180 * np.pi
    #     can_bus[-1] = patch_angle
    can_bus = [0., 0., 0., 0.97147692, 0.97147692, 0.97147692, 0.97147692,
               0.12370334, -0.17086322, 9.71368351, 0.01170966, 0.02093362,
               -0.02357822, 9.00888615, 0., 0., 5.80457608, 0.]
    kwargs = {'img_metas': [{'lidar2img': lidar2img,
                              'can_bus': can_bus}]}

    bev_embed = encoder(
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
        **kwargs
    )

    print('test passed')

