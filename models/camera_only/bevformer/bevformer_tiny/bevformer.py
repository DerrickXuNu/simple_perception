"""
BEVFormer main function
"""
import numpy as np

import torch
import torch.nn as nn
import copy

from common_modules import ResnetEncoder, FPN
from transformers import BEVFormerHead


class BEVFormer(nn.Module):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None):
        super(BEVFormer, self).__init__()
        # image backbone create
        self.img_backbone = ResnetEncoder(img_backbone)
        # image neck creation
        self.img_neck = FPN(img_neck)
        # bevformer head build
        self.pts_bbox_head = BEVFormerHead(**pts_bbox_head)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                # N is the camera num
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        # img neck
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B),
                                  C, H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape

            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            # multi-scale features: [(b, len_queue, num_cam, C, H, W ), ]
            img_feats_list = self.extract_feat(img=imgs_queue,
                                               len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None):
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
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # encoder + decoder output
        output = self.pts_bbox_head(img_feats,
                                    img_metas,
                                    prev_bev,
                                    only_bev=False)
        loss = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, output,
                         img_metas=img_metas)

        return loss

    def forward_test(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None):
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
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # encoder + decoder output
        output = self.pts_bbox_head(img_feats,
                                    img_metas,
                                    prev_bev,
                                    only_bev=False)
        # bbx_prediction, bbx_score, bbx_label
        ret_list = self.pts_bbox_head.get_bboxes(output, img_metas)

        return ret_list

    def forward(self, train=True, **kwargs):
        if train:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


if __name__ == '__main__':
    # -----------------------Configuration--------------------------------
    from config import bevformer_config

    # -----------------------Mocked data ---------------------------------
    # real image meta example
    from mocked_img_meta import img_metas
    # mocked input image and gt
    batch_size = 1
    n_gt = 32
    len_queue = 3
    num_camera = 6

    imgs = torch.rand(batch_size, len_queue, num_camera, 3, 480, 800)
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

    # -----------------------Model Build ---------------------------------
    bevformer = BEVFormer(**bevformer_config)

    # During train
    # final loss and the loss for every decoder layer
    loss = bevformer(train=True, **data_dict)
    # During test
    # [(300 * 9), (300,), (300,)]
    output_list = bevformer(train=False, **data_dict)



