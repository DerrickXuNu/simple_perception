# -----------------------Configuration--------------------------------
# BEVFormer Encoder
encoder_config = {
    'type': 'BEVFormerEncoder',
    'num_layers': 3,
    # the bev evaluation range in meters
    'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    # we sample 4 points in pillar on the z-axis
    'num_points_in_pillar': 4,
    'return_intermediate': False,
    'transformerlayers': {
        'type': 'BEVFormerLayer',
        'attn_cfgs': [
            {
                'type': 'TemporalSelfAttention',
                'embed_dims': 256,
                'num_levels': 1
            },
            {
                'type': 'SpatialCrossAttention',
                'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                'deformable_attention': {
                    'type': 'MSDeformableAttention3D',
                    'embed_dims': 256,
                    'num_points': 8,
                    'num_levels': 1
                },
                'embed_dims': 256
            }
        ],
        'feedforward_channels': 512,
        'ffn_dropout': 0.1,
        'operation_order': (
            'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    }
}
# BEVFormer decoder configuration
decoder_config = {
    'type': 'DetectionTransformerDecoder',
    'num_layers': 6,
    'return_intermediate': True,
    'transformerlayers': {
        'type': 'DetrTransformerDecoderLayer',
        'attn_cfgs': [
            {
                'type': 'MultiheadAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'dropout': 0.1
            },
            {
                'type': 'CustomMSDeformableAttention',
                'embed_dims': 256,
                'num_levels': 1
            }
        ],
        'feedforward_channels': 512,
        'ffn_dropout': 0.1,
        'operation_order': (
            'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    }
}
# The transformer config
transformer = {'type': 'PerceptionTransformer',
               # Whether to rotate the previous bev features
               # to align them with current BEV feature
               'rotate_prev_bev': True,
               # Whether to shift the previous bev features
               # to align them with current BEV feature
               'use_shift': True,
               # if true, the ego vehicle's pose and speed info
               # will be passed to a mlp and add to the bev embed
               'use_can_bus': True,
               # this will be used for camera embedding,
               # level embedding, initialized reference points for
               #
               'embed_dims': 256,
               'encoder': encoder_config,
               'decoder': decoder_config
               }
# bbx coder, conver the output to bbx format
bbx_coder = {'type': 'NMSFreeCoder',
             # this is ued to filter out the prediction bbx out of range
             'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2,
                                   10.0],
             'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
             # maximum predict object number
             'max_num': 300,
             'score_threshold': 0.5,
             # not useful
             'voxel_size': [0.2, 0.2, 8],
             'num_classes': 10}
# this will be added to bev pos, which will be added to bev query
positional_encoding = {'type': 'LearnedPositionalEncoding',
                       'num_feats': 128,
                       'row_num_embed': 50,
                       'col_num_embed': 50}

# classification loss
loss_cls = {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0,
            'alpha': 0.25,
            'loss_weight': 2.0}
# regression loss
loss_bbox = {'type': 'L1Loss', 'loss_weight': 0.25}

# gt assigner (for hungarian matching)
assigner = {'type': 'HungarianAssigner3D',
            'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0},
            'reg_cost': {'type': 'BBox3DL1Cost', 'weight': 0.25},
            'iou_cost': {'type': 'IoUCost', 'weight': 0.0},
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]}
# training cfg for head
train_cfg = {'grid_size': [512, 512, 1],  # acually useless
             'voxel_size': [0.2, 0.2, 8],  # actually useless
             'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
             'out_size_factor': 4,
             'assigner': assigner
             }
test_cfg = None

head_cfg = {
    'type': 'BEVFormerhead',
    # the bev query size
    'bev_h': 50,
    'bev_w': 50,
    # object query number
    'num_query': 900,
    'num_classes': 10,
    'in_channels': 256,
    'sync_cls_avg_factor': True,
    'with_box_refine': True,
    'as_two_stage': False,
    'transformer': transformer,
    'bbox_coder': bbx_coder,
    'positional_encoding': positional_encoding,
    'loss_cls': loss_cls,
    'loss_bbox': loss_bbox,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg
}

# Img backbone config
backbone_config = {'depth': 50,  # ResNet50
                   'pretrained': False
                   }
# Img Neck config
neck_config = {'in_channels': [2048],
               'out_channels': 256
               }

# merge all parts together:
bevformer_config = {
    'img_backbone': backbone_config,
    'img_neck': neck_config,
    'pts_bbox_head': head_cfg
}