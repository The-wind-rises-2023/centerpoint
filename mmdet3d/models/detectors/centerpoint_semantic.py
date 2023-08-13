# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector

@DETECTORS.register_module()
class CenterPointSemantic(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_semantic=True,
                 init_cfg=None):
        super(CenterPointSemantic,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head, 
                             train_cfg, test_cfg, pretrained, init_cfg)
        self.use_semantic = use_semantic
        if train_cfg:
            self.cfg = self.train_cfg['pts']
        elif test_cfg:
            self.cfg = self.test_cfg['pts']
        self.in_channels = 1
        self.nx = round((self.cfg['point_cloud_range'][3] - self.cfg['point_cloud_range'][0]) / self.cfg['voxel_size'][0])
        self.ny = round((self.cfg['point_cloud_range'][4] - self.cfg['point_cloud_range'][1]) / self.cfg['voxel_size'][1])
        self.grid_size = [self.nx, self.ny]

    def scatter(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        batch_masks = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)
            mask = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels
            mask[:, indices] = 1
            # Append to a list for later stacking.
            batch_canvas.append(canvas)
            batch_masks.append(mask)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_masks = torch.stack(batch_masks, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.ny,
                                         self.nx)
        batch_masks = batch_masks.view(batch_size, self.ny,
                                         self.nx)

        return batch_canvas, batch_masks

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        seg_pixel_gt = None
        if voxels.shape[-1] == 5:
            # got gt seg label
            self.with_semantic_gt=True

            seg_labels = voxels[:,:,-1].view(voxels.shape[0], voxels.shape[1], 1)
            seg_labels_gt,_ = seg_labels.max(axis=1)
            batch_size = coors[-1, 0] + 1
            # 0 is for Kong
            seg_pixel_gt, masks = self.scatter(seg_labels_gt + 1, coors, batch_size)
            
            voxels = voxels[:,:,:4]

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)
        return {'x': x, 'seg_gt':seg_pixel_gt, 'masks': masks}

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        self.pts_bbox_head(pts_feats)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(pts_feats['seg_preds'], pts_feats['seg_gt'])
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        seg_results = self.pts_bbox_head.predict(outs)

        return seg_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """same as simple test pts
        """
        return self.simple_test_pts(feats, img_metas, rescale)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """ Test function without augmentation. """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        res = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict in res:
                result_dict.update(bbox_pts)
                if 'seg_gt' in pts_feats:
                    result_dict.update({'seg_gt': pts_feats['seg_gt']})
        return res

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
