# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import Sequential

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class VisionTransformerFgClsHead(ClsHead):
    """Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Number of the dimensions for hidden layer. Only
            available during pre-training. Default None.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to Tanh.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_dim=None,
                 act_cfg=dict(type='Tanh'),
                 init_cfg=dict(type='Constant', layer='Linear', val=0),
                 *args,
                 **kwargs):
        super(VisionTransformerFgClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))

    def init_weights(self):
        super(VisionTransformerFgClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)

    def simple_test(self, x):
        """Test without augmentation."""
        x = x[-1]
        _, cls_token = x
        cls_score = self.layers(cls_token)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        x = x[-1]
        _, cls_token = x
        cls_score = self.layers(cls_token)
        cls_loss = self.loss(cls_score, gt_label, **kwargs)
        contrast_loss = self.contrast_loss(cls_token, gt_label.view(-1))
        losses = cls_loss + contrast_loss
        return losses

    def contrast_loss(self, tokens, gt_label):
        alpha = 0.4
        B = tokens.shape[0]
        tokens = F.normalize(tokens)
        cor_matrix = tokens.mm(tokens.t())
        pos_gt_matrix = torch.stack([gt_label == tag for tag in gt_label]).float()
        neg_gt_matrix = 1 - pos_gt_matrix
        pos_cor_matrix = 1 - cor_matrix
        neg_cor_matrix = cor_matrix - alpha
        neg_cor_matrix[neg_cor_matrix < 0] = 0
        loss = (pos_cor_matrix * pos_gt_matrix).sum() + (neg_cor_matrix * neg_gt_matrix).sum()
        loss /= (B * B)
        return loss



