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
from .pipelines import LoadPointsFromFile

from mmdet3d.core.bbox import box_np_ops, points_cam2img

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
    
    def name(self):
        return 'ZCDataset'

    def convert_to_kitti_format(self, gts, dets, label2cat):
        for gt in gts:
            gt['name'] = [
                'DontCare' if n not in self.CLASSES else n for n in gt['gt_names']]
            mask = [True if n != 'DontCare' else False for n in gt['name']]
            gt['name'] = [n for n in gt['name'] if n != 'DontCare']
            gt['occluded'] = np.array([0 for i in range(len(gt['name']))])
            gt['truncated'] = np.array([0 for i in range(len(gt['name']))])
            gt['bbox'] = np.array([[0,0,200,200] for i in range(len(gt['name']))]).reshape(-1,4) #x1y1x2y2 fake image box
            
            # alpha = local_yaw, but no need now
            gt['alpha'] = np.array([0 for i in range(len(gt['name']))]) # do not care and do not use aos

            # TODO check clock wise, rotation_y ?
            #3d box in camera axies
            gt['gt_bboxes_3d'] = gt['gt_bboxes_3d'][mask,...]
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

    def get_points3d(self, index):
        input_dict = self.get_data_info(index)

        if getattr(self, "loader", None) is None:
            self.loader = LoadPointsFromFile(coord_type='LIDAR', load_dim=4, use_dim=4, file_client_args=dict(backend='disk')) 
        result = self.loader(input_dict)
        return result['points']


    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 do_not_eval=False,
                 save_badcase_only=False,
                 min_gt_points_dict=None,
                 savedata=False,
                 save_data_with_gt=False
                 ):
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
        # filter boxes by min pts number
        if min_gt_points_dict is not None:
            assert len(results) == len(self), print(f"{len(results)} != {len(self)}", file=sys.stderr, flush=True)
            if not isinstance(min_gt_points_dict, dict):
                min_gt_points_dict = {p.split('=')[0]:int(p.split('=')[1]) for p in min_gt_points_dict}

            label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
            for ridx, r in enumerate(results):
                points = self.get_points3d(ridx)
                boxes = r['pts_bbox']['boxes_3d']
                labels = r['pts_bbox']['labels_3d']
                scores = r['pts_bbox']['scores_3d']

                new_boxes = []
                new_labels = []
                new_scores = []

                box_type = type(r['pts_bbox']['boxes_3d'])

                for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                    label_name = label2cat[label.item()]
                    if label_name not in min_gt_points_dict:
                        new_boxes.append(box)
                        new_labels.append(label)
                        new_scores.append(score)
                        continue
                    indices = box_np_ops.points_in_rbbox(points[:, :3].tensor.numpy(), box.reshape(-1,7).numpy())
                    num_points_in_gt = indices.sum(0)
                    if num_points_in_gt < int(min_gt_points_dict[label_name]):
                        continue
                    new_boxes.append(box)
                    new_labels.append(label)
                    new_scores.append(score)
                
                results[ridx]['pts_bbox']['boxes_3d'] = torch.stack(new_boxes) if len(new_boxes) > 0 else torch.tensor([],dtype=torch.float32)
                results[ridx]['pts_bbox']['boxes_3d'] = box_type(results[ridx]['pts_bbox']['boxes_3d'])
                results[ridx]['pts_bbox']['labels_3d'] = torch.stack(new_labels) if len(new_labels) > 0 else torch.tensor([],dtype=torch.int32)
                results[ridx]['pts_bbox']['scores_3d'] = torch.stack(new_scores) if len(new_scores) > 0 else torch.tensor([],dtype=torch.float32)
                    
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)

        # for centerpoint
        if 'pts_bbox' in result_files[0]:
            for r in result_files:
                r.update({k: r['pts_bbox'][k] for k in r['pts_bbox'].keys()}) 
        
        ap_dict = dict()
        if not do_not_eval:
            from mmdet3d.core.evaluation import kitti_eval_zc
            gt_annos = [info['annos'] for info in self.data_infos]
            label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}

            gt_annos, result_files = self.convert_to_kitti_format(gt_annos, result_files, label2cat)
            if isinstance(result_files, dict):
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
            self.show(results, out_dir, show=show, pipeline=pipeline, do_not_eval=do_not_eval,
                      savedata=savedata, save_data_with_gt=save_data_with_gt)
        return ap_dict

    def show(self, results, out_dir, show=True, pipeline=None, do_not_eval=False,
             savedata=False, save_data_with_gt=False):
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
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_points']['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            
            gt_bboxes, show_gt_bboxes = None,None
            if not do_not_eval:
                gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
                show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            # save data as json
            if savedata:
                predict_dir = osp.join(out_dir, "predict")
                mmcv.mkdir_or_exist(predict_dir)
                json_path = predict_dir + ('/%s.json' % file_name)
                self.save_data(result, json_path, gt_bboxes, save_data_with_gt)

            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    file_name,
                    box_mode='lidar',
                    show=show)          
