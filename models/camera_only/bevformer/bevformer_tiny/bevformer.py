"""
This is for those who does not have large GPU but would like to know the
core part
"""



if __name__ == '__main__':
    # model configuration
    # image backbone to extract features from rgb image
    img_backbone_config = {
        'type': 'ResNet',
        'depth': 50,  # resnet50
        'pretrained': True,  # pretrained or not
        'out_indices': [3],  # the third stages' output will be used
    }
    # image neck component. Used to refine the features from the backbone
    img_neck_config = {
        'type': 'FPN',  # Feature Pyramid Network
        'in_channel': [2048],  # the input features' channel
        'out_channel': 256,
        'add_extra_output': 'on_output',  # where to add an extra output conv
    }
    # BEV Encoder:
    bev_encoder_config = {
        'type': 'BEVFormerEncoder',
        'num_layers': 3,  # 3 layers encoder, the same for both temporal and sptial attention
        'range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # the bev map covered range
        'ffd_forward_channels': 512,
        'ffn_dropuout': 0.1,
        'TemporalSelfAttention': {  # this is for temporal dimension BEV Feature fusion, i.e., Section 3.4 in the paper
            'embed_dims': 256,
            'num_levels': 1
        },
        'SpatialCrossAttention': {# this refers to the part in Section 3.3
            'embed_dims': 256,
            'num_cams': 6,
            'num_points': 8, # todo: the number of reference point for deformable transformer
        }
    }


    #--------------------------- Data Mock ---------------------------------#

    # mocked image data (b, # of continuous frames, # of camera, 3 (rgb), h, w)
    img = torch.rand(1, 3, 6, 3, 480, 800)
    # the length
    len_queue = img.size(1)
    # the previous images
    prev_img = img[:, :-1, ...]
    # current image
    img = img[:, -1, ...]

    # we first retrieve