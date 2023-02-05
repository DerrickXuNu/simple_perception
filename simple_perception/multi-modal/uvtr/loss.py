import torch
import torch.nn as nn


class NMSFreeCoder(nn.Module):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None, #
                 post_center_range=None, # used to filter the bbx out of range
                 max_num=100, # maximum objects number in a scenario
                 score_threshold=None,
                 num_classes=10):
        super(NMSFreeCoder, self).__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass


    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # max_num = self.max_num
        #
        # cls_scores = cls_scores.sigmoid()
        # # find the top k best candidates
        # scores, indexs = cls_scores.view(-1).topk(max_num)
        # labels = indexs % self.num_classes
        # bbox_index = indexs // self.num_classes
        # bbox_preds = bbox_preds[bbox_index]
        # todo: not finished yet
        pass
