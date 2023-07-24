from glob import glob
from os.path import join
from concurrent import futures as futures
import os.path as osp
import mmcv
import numpy as np
import multiprocessing

import pypcd.pypcd as pypcd

ROOTDIR = None


def process_info(info_path):

    filename = osp.splitext(osp.basename(info_path))[0]
    info = mmcv.load(info_path)
    # process info
    gt_bboxes_3d = []
    gt_labels = []

    for idx, obj in enumerate(info):
        gt_labels.append(obj['obj_type'])
        obj = obj['psr']
        one_box = []
        for key in ('position', 'scale'):
            one_box.extend([float(obj[key][v]) for v in ('x', 'y', 'z')])
        #yaw
        one_box.append(obj['rotation']['z'])

        gt_bboxes_3d.append(one_box)

    gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
    result = {
        'sample_idx': filename,
        'annos': {'box_type_3d': 'lidar',
                  'gt_bboxes_3d': gt_bboxes_3d, 'gt_names': gt_labels},
        'calib': {},
        'images': {}
    }
    return result


def process_pcd(pcd_path):
    file_name = osp.splitext(osp.basename(pcd_path))[0]
    pc = pypcd.PointCloud.from_path(pcd_path)
    pts = np.stack([pc.pc_data['x'],
                    pc.pc_data['y'],
                    pc.pc_data['z']],
                   axis=-1)
    pts = np.concatenate([pts.astype(np.float32), np.expand_dims(
        pc.pc_data["intensity"].astype(np.float32), -1)], axis=1)
    global ROOTDIR
    save_path = osp.join(ROOTDIR, f"bin/{file_name}.bin")
    pts.tofile(save_path)
    return save_path


def process_single_data(info_path, pcd_path):
   # process pcd
    save_bin_path = process_pcd(pcd_path)

    if info_path is None:
        # process pcd
        return {
            'sample_idx': osp.splitext(osp.basename(pcd_path))[0],
            'annos': {'box_type_3d': 'lidar',
                      'gt_bboxes_3d': np.empty((0, 7)), 'gt_names': []},
            'calib': {},
            'images': {},
            'lidar_points': {'lidar_path': save_bin_path,
                             }
        }

    # process anno info
    result = process_info(info_path)
    
    # keep empty info
    #if not result['annos']['gt_names']:
    #    return None

    
    result.update(
        {'lidar_points': {'lidar_path': save_bin_path,
                          }})
    return result


def generate_pickle(infos_path, pcds_path, filename, num_workers=8, speed_up=True):
    # for io speed
    if speed_up:
        with futures.ProcessPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_data, infos_path, pcds_path)
    else:
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_data, infos_path, pcds_path)

    infos = filter(None, infos)
    mmcv.dump(list(infos), filename)
