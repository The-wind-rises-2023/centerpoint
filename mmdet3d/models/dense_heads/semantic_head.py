# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core import multi_apply


@HEADS.register_module()
class SemanticHead(BaseModule):
    """Semantic segmentation head for pixel-wise segmentation.

    Args:
        in_channels (int): The number of input channel.
        num_classes (int): The number of class.
        loss_seg (dict): Config of segmentation loss.
        loss_part (dict): Config of part prediction loss.
    """

    def __init__(self,
                 in_channels,
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 out_channels=None,
                 num_classes=3,
                 seg_score_thr=0.3,
                 init_cfg=None,
                 loss_seg=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0)):
        super(SemanticHead, self).__init__(init_cfg=init_cfg)
        self.train_cfg=train_cfg
        # if isinstance(weight, list):
        #     weight = torch.Tensor(weight).view(-1)
        # elif isinstance(weight, np.array):
        #     weight = torch.from_numpy(weight).view(-1)
        
        if train_cfg:
            self.weight = train_cfg['semantic_code_weights']
        self.num_classes = num_classes
        self.seg_score_thr = seg_score_thr
        self.deconv=nn.ConvTranspose2d(in_channels,in_channels,stride=2,kernel_size=2)
        self.seg_cls_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels//4,in_channels//4,stride=2,kernel_size=2),
            nn.Conv2d(in_channels//4, in_channels//16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, num_classes, kernel_size=3, padding=1, bias=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels,in_channels,stride=2,kernel_size=2),
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels//4,in_channels//4,stride=2,kernel_size=2),
            nn.Conv2d(in_channels//4, in_channels//8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels//8,in_channels//8,stride=2,kernel_size=2),
            nn.Conv2d(in_channels//8, in_channels//16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//16),
            nn.ReLU(inplace=True))
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
            nn.Conv2d(in_channels//16, num_classes, kernel_size=3, padding=1, bias=True))

        self.loss_seg = build_loss(loss_seg)

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            tuple[list[torch.Tensor]]: Predicted boxes and classification
                scores.
        """
        # todo: add feature name
        # import ipdb;ipdb.set_trace()
        # seg_preds = self.seg_cls_layer(self.deconv(feat_dict['x'][0]))
        x1 = self.up1(feat_dict['x'][0])
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        seg_preds = self.maxpool_conv(x3)
        feat_dict.update({'seg_preds': seg_preds})
        return feat_dict

    def loss(self, seg_preds, seg_targets):
        """Calculate pixel-wise segmentation losses.

        Args:
            semantic_results (dict): Results from semantic head.

                - seg_preds: Segmentation predictions.

            semantic_targets (dict): Targets of semantic results.

                - seg_preds: Segmentation targets.

        Returns:
            dict: Loss of segmentation and part prediction.

                - loss_seg (torch.Tensor): Segmentation prediction loss.
        """
        gt = seg_targets.long()
        losses = F.nll_loss(torch.log(F.softmax(seg_preds, dim = 1).clamp(min=1e-7)), 
                            gt, torch.tensor(self.weight).to(seg_preds.device))

        return {'seg_loss' : losses}

    @torch.no_grad()
    def predict(self, pred_dict):
        prob, label = F.softmax(pred_dict['seg_preds'] * pred_dict['masks'], dim = 1).max(1)
        res_dict = {
            'prob': prob,
            'label': label
        }
        return res_dict
