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
import json


@DATASETS.register_module()
class ZCSemanticDataset(Custom3DDataset):
    r"""ZC Semantic Dataset.

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

    SEG_CLASSES = (
            'Kong', 'Unlabel', 'Road', 'Station', 'Fence')
        #    none,      0,        1,      2,         3

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 do_not_eval=False,
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
        self.do_not_eval = do_not_eval


    def name(self):
        return 'ZCSemanticDataset'

    def convert_to_kitti_format(self, gts, dets, label2cat):
        if not self.do_not_eval:
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
        else:
            gts = []

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

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['sample_idx']
        pts_filename = osp.join(self.data_root,
                                info['lidar_points']['lidar_path'])

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)

        return input_dict

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        return example


    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations

    def save_data(self, results, dst, gt_data, with_gt=False):
        """
        save pointcloud + predict result + gt in txt format in dst dir to compare
        torch result and other result(like result in perception onnx)
        """
        # import ipdb;ipdb.set_trace()
        sz = len(results['boxes_3d'])

        json_objects = []
        with open(dst, 'w') as f:
            for idx in range(sz):
                box_lidar = results['boxes_3d'][idx]
                box_json = {'psr': {'position': {'x': float(box_lidar.center[0][0]), 'y': float(box_lidar.center[0][1]),
                                        'z': float(box_lidar.center[0][2])},
                                    'scale': {'x': float(box_lidar.dims[0][0]), 'y': float(box_lidar.dims[0][1]),
                                        'z': float(box_lidar.dims[0][2])},
                                    'rotation': {'x': 0, 'y': 0, 'z': float(box_lidar.yaw[0])}},
                                    'obj_type': self.CLASSES[results['labels_3d'][idx]],
                                    'obj_id' : '1',
                                    'score': float(results['scores_3d'][idx])
                                    }
                json_objects.append(box_json)
            if with_gt:
                # import ipdb;ipdb.set_trace()
                for idx in range(len(gt_data)):
                    gt_lidar = gt_data[idx]
                    box_json = {'psr': {'position': {'x': float(gt_lidar[0]), 'y': float(gt_lidar[1]),
                                        'z': float(gt_lidar[2])},
                                    'scale': {'x': float(gt_lidar[3]), 'y': float(gt_lidar[4]),
                                        'z': float(gt_lidar[5])},
                                    'rotation': {'x': 0, 'y': 0, 'z': float(gt_lidar[6])}},
                                    'obj_type': "GT",
                                    'obj_id' : '1',
                                    'score': 1.0
                                    }
                    json_objects.append(box_json)

            json.dump(json_objects, f)

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

        # for semantic
        dts=[info['label'].cpu() for info in result_files]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.SEG_CLASSES)}
        
        if not self.do_not_eval:
            from mmdet3d.core.evaluation import seg_eval
            gt_labels=[info['seg_gt'].cpu() for info in result_files]       
            ap_dict = seg_eval(gt_labels, dts, label2cat, 0)
        else: 
            ap_dict = {}

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def get_label_color(self, label_map):
        sem_label_color_list = [np.array([0, 0, 0]), 
                        np.array([220, 220, 220]), 
                        np.array([255,0,255]), 
                        np.array([0, 255, 255]),
                        np.array([0, 255, 0]),
                        np.array([128,128,128])]
        color_map = np.zeros((label_map.shape[0], label_map.shape[1], 3))
        for i in range(len(sem_label_color_list)):
            mask = (label_map == i)
            color_map[mask] = sem_label_color_list[i]
        return color_map

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """

        assert out_dir is not None, 'Expect out_dir, got none.'
        import cv2 as cv 
        predict_dir = osp.join(out_dir, "predict")
        mmcv.mkdir_or_exist(predict_dir)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_points']['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            img_path = predict_dir + ('/%s.png' % file_name)

            pred_np = np.array(result['label'][0].to('cpu'))
            pred_img = self.get_label_color(pred_np)
            if not self.do_not_eval:
                gt_np = np.array(result['seg_gt'][0].to('cpu'))
                gt_img = self.get_label_color(gt_np)
                res = np.concatenate((pred_img, gt_img), axis = 0)
            else: 
                res = pred_img
            cv.imwrite(img_path, res)
