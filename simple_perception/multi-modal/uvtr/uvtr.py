"""
UVTR model that contains image backbone, neck, viewtrans, and lidar backbone
"""
import os
import torch
import torch.nn as nn

from easydict import EasyDict
from uvtr_head import UVTRHead
from common_modules import ResnetEncoder, FPN, GridMask


class UVTR(nn.Module):
    """UVTR."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 depth_head=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 load_img=None,
                 load_pts=None,
                 **kwargs):
        # todo: not implement the point based features yet
        super(UVTR, self).__init__()
        if pts_bbox_head:
            # uvtr head
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = UVTRHead(**pts_bbox_head)
        # image backbone
        if img_backbone:
            self.image_backbone = ResnetEncoder(img_backbone)
        # image neck for feature augmentation
        if img_neck:
            self.img_neck = FPN(**img_neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # for image augmentation
        if self.with_img_backbone:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False,
                                      ratio=0.5, mode=1, prob=0.7)
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels = in_channels[0]

            self.input_proj = nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=1)

            if "SimpleDepth" in depth_head.type:
                self.depth_dim = depth_head.model.depth_dim
                self.depth_net = nn.Conv2d(out_channels,
                                           self.depth_dim, kernel_size=1)
            else:
                raise NotImplementedError
            self.depth_head = depth_head
            self.use_grid_mask = use_grid_mask

        self.load_img = load_img
        self.load_pts = load_pts

    def init_weights(self):
        """Initialize weights of the depth head."""
        if not self.with_img_backbone:
            return

    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images.

                This function takes in a batch of images (`img`) and
                corresponding image metadata (`img_metas`) and returns the
                extracted features for each image.

        Args:
        img (Tensor): A batch of images with shape (B, N, C, H, W),
        where B is the batch size, N is the number of instances in an image,
        C is the number of channels, H is the height, and W is the width.

        img_metas (List[dict]): A list of metadata for each image in the
        batch, where each dictionary contains information such as the
        original shape of the image, scaling factors, etc.

        Returns:
            list: A list of feature maps,
            where each feature map has shape (B, N, C', H', W').
        """
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = self.input_proj(img_feat)
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def pred_depth(self, img, img_metas, img_feats=None):
        """Predict depth maps of images."""
        # If the image features have not been extracted yet, return None
        if img_feats is None:
            return None

        # Get batch size of input image
        B = img.size(0)

        # Reshape the input image if necessary
        if img.dim() == 5 and img.size(0) == 1:
            # Squeeze the first dimension if it is of size 1
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            # Reshape the input image to (B * N, C, H, W)
            img = img.view(B * N, C, H, W)

        # Predict the depth maps using the depth head
        if self.depth_head.type == "SimpleDepth":
            depth = []
            for _feat in img_feats:
                # Pass the image features through the depth network
                _depth = self.depth_net(_feat.view(-1, *_feat.shape[-3:]))
                # Apply softmax to the depth predictions
                _depth = _depth.softmax(dim=1)
                # Append the depth predictions to the output list
                depth.append(_depth)
        else:
            # Raise an error if the depth head type is not supported
            raise NotImplementedError

        # Return the list of depth maps
        return depth

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        if hasattr(self, "img_backbone"):
            img_feats = self.extract_img_feat(img, img_metas)
            img_depth = self.pred_depth(img=img, img_metas=img_metas,
                                        img_feats=img_feats)
        else:
            img_feats, img_depth = None, None

        if hasattr(self, 'pts_voxel_encoder'):
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = None

        return pts_feats, img_feats, img_depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      depth_prob=None,
                      depth_coord=None,
                      img=None,
                      **kwargs):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # feature extraction
        pts_feat, img_feats, img_depth = self.extract_feat(points=points,
                                                           img=img,
                                                           img_metas=img_metas)
        # view transfer + decoder
        outs = self.pts_bbox_head(pts_feat, img_feats, img_metas, img_depth)
        # loss calculation
        loss_inputs = [gt_bboxes_3d, gt_labels_3d,
                       depth_prob, depth_coord, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    def forward(self, train=True, **kwargs):
        if train:
            return self.forward_train(**kwargs)
        else:
            return self.forward_train(**kwargs)


if __name__ == '__main__':
    # -----------------------Configuration--------------------------------
    from configs.config_util import Config

    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    camera_config = Config.fromfile(os.path.join(current_dir,
                                                 'configs/base_camera_config.py'))._cfg_dict
    camera_config = EasyDict(camera_config['model'])

    # -----------------------Mocked data ---------------------------------
    # real image meta example
    from mocked_img_meta import img_metas
    # mocked input image and gt
    batch_size = 1
    n_gt = 32
    len_queue = 3
    num_camera = 6

    imgs = torch.rand(batch_size, num_camera, 3, 928, 1600)
    # regression, cx, cy, cz, w, l, h, rot, vx, vy
    gt_bboxes_3d = [torch.rand(n_gt, 9)] * batch_size
    # 10 classes
    gt_labels_list = [torch.randint(0, 10, (n_gt,))] * batch_size
    img_metas = [img_metas * len_queue] * batch_size

    data_dict = {
        'img_metas': img_metas,
        'img': imgs,
        'gt_bboxes_3d': gt_bboxes_3d,
        'gt_labels_3d': gt_labels_list
    }

    uvtr_camera = UVTR(**camera_config)
    print('here')