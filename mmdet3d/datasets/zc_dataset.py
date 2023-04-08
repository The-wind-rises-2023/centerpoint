# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class ZCDataset(Custom3DDataset):
    r"""ZC Dataset.

    This class serves as the API for experiments on the 
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ('Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=None, 
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range

    def convert_to_kitti_format(self, gts, dets, label2cat):
        for gt in gts:
            gt['name'] = [
                'DontCare' if n not in self.CLASSES else n for n in gt['gt_names']]
            gt['occluded'] = np.array([0 for i in range(len(gt['name']))])
            gt['truncated'] = np.array([0 for i in range(len(gt['name']))])
            gt['bbox'] = np.array([[0,0,200,200] for i in range(len(gt['name']))]).reshape(-1,4) #x1y1x2y2 fake image box
            
            # alpha = local_yaw, but no need now
            gt['alpha'] = np.array([0 for i in range(len(gt['name']))]) # do not care and do not use aos

            # TODO check clock wise, rotation_y ?
            #3d box in camera axies
            cam_box =  LiDARInstance3DBoxes(gt['gt_bboxes_3d']).convert_to(Box3DMode.CAM)
            gt['location'] = cam_box.bottom_center.numpy()
            gt['dimensions'] = cam_box.dims.numpy()
            gt['rotation_y'] = cam_box.yaw.numpy()

        for dt in dets:
            if len(dt['labels_3d']) == 0:
                dt['location'] = np.empty((0,3), dtype=np.float32)
                dt['dimensions'] = np.empty((0,3), dtype=np.float32)
                dt['rotation_y'] = np.empty((0), dtype=np.float32)
                dt['score'] = dt['scores_3d'].numpy()
                dt['name'] = dt['labels_3d'].numpy()
                dt['bbox'] = np.empty((0,4), dtype=np.float32)
                dt['alpha'] = np.empty((0), dtype=np.float32)
                continue
            #cam_box =  LiDARInstance3DBoxes(dt['boxes_3d'].numpy()).convert_to(Box3DMode.CAM)
            cam_box =  LiDARInstance3DBoxes(dt['boxes_3d'].tensor).convert_to(Box3DMode.CAM)
            dt['location'] = cam_box.bottom_center.numpy()
            dt['dimensions'] = cam_box.dims.numpy()
            dt['rotation_y'] = cam_box.yaw.numpy()
            dt['score'] = dt['scores_3d'].numpy()
            dt['name'] = [label2cat[n.item()] for n in dt['labels_3d']]
            dt['bbox'] = np.array([[0,0,200,200] for i in range(len(dt['name']))]) #x1y1x2y2 fake image bbox 
            dt['alpha'] = np.array([0 for i in range(len(dt['name']))]) # do not care and do not use aos
        return gts, dets

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)

        # for centerpoint
        if 'pts_bbox' in result_files[0]:
            for r in result_files:
                r.update({k: r['pts_bbox'][k] for k in r['pts_bbox'].keys()}) 

        from mmdet3d.core.evaluation import kitti_eval_zc
        gt_annos = [info['annos'] for info in self.data_infos]

        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}

        gt_annos, result_files = self.convert_to_kitti_format(gt_annos, result_files, label2cat)

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                eval_types = ['bev']
                if 'img' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = kitti_eval_zc(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            if metric == 'img_bbox':
                ap_result_str, ap_dict = kitti_eval_zc(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                ap_result_str, ap_dict = kitti_eval_zc(gt_annos, result_files,
                                                    self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict
