# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools.data_converter import indoor_converter as indoor
from tools.data_converter import kitti_converter as kitti
from tools.data_converter import lyft_converter as lyft_converter
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)


def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)

    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
        return

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    nuscenes_converter.export_2d_annotation(
        root_path, info_train_path, version=version)
    nuscenes_converter.export_2d_annotation(
        root_path, info_val_path, version=version)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{out_dir}/{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers, num_points):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path,
        info_prefix,
        out_dir,
        workers=workers,
        num_points=num_points)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 5. Here we store pose information of these frames
            for later use.
    """
    from tools.data_converter import waymo_converter as waymo

    splits = ['training', 'validation', 'testing']
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'testing'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(
        out_dir, info_prefix, max_sweeps=max_sweeps, workers=workers)
    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()

def zc_data_prep(
            root_path,
            out_dir,
            info_prefix="json",
            pcd_prefix="pcd", 
            training_ratios = 0.9,
            workers=4,
            with_plain=False):
    """
    root_path-> json-> xxx.json...
             -> pcd-> xxx.pcd...

    TODO: deal with road plain
    """
    import mmcv
    import numpy as np
    import math
    import os
    from glob import glob
    from tools.data_converter import zc_converter as zc
    from tqdm import tqdm

    assert osp.exists(root_path)
    assert training_ratios <= 1.0

    info_files = glob(osp.join(root_path, info_prefix, '*.json'))
    ori_pcd_files = glob(osp.join(root_path, pcd_prefix, '*.pcd'))

    pcd_files = []
    # check info files and pcd files to make sure data are matched when list length is not equal
    for info_file in tqdm(info_files):
        name = osp.splitext(osp.basename(info_file))[0]
        pcd_path = osp.join(root_path, pcd_prefix, name + '.pcd')
        assert pcd_path in ori_pcd_files, print(f"{pcd_path} not exist!")
        pcd_files.append(pcd_path)

    assert len(info_files) > 0 and len(pcd_files) > 0, print(
        f"{len(info_files)}{len(pcd_files)}")

    zc.ROOTDIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, 'bin'), exist_ok=True)

    # generate pickles
    idxs = np.arange(len(info_files))
    np.random.shuffle(idxs) 
    splits = ['training', 'testing']

    train_numbers = int(training_ratios * len(info_files))
    # may skip some files
    other_numbers = math.floor(len(info_files) - train_numbers / (len(splits) - 1))

    cur_numbers = 0
    print(f"Start generate {len(splits)} pickle files. {splits}")
    for split in splits:
        if split == 'training':
            numbers = train_numbers
        else:
            numbers = other_numbers

        split_idxs = idxs[cur_numbers:cur_numbers + numbers]

        split_infos = [info_files[idx] for idx in split_idxs]

        split_pcds = [pcd_files[idx] for idx in split_idxs]

        cur_numbers += numbers
        pkl_file_name = osp.join(out_dir, f'{split}_infos.pkl')
        zc.generate_pickle(split_infos, split_pcds, pkl_file_name, num_workers=workers)

    GTDatabaseCreater(
        'ZCDataset',
        out_dir,
        'gt',
        f'{out_dir}/training_infos.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()


def zc_data_dummy(
        root_path,
        out_dir,
        workers=4):
    """
    root_path-> json-> xxx.json...
             -> pcd-> xxx.pcd...

    TODO: deal with road plain
    """
    import mmcv
    import numpy as np
    import math
    import os
    from glob import glob
    from tools.data_converter import zc_converter as zc
    from tqdm import tqdm

    assert osp.exists(root_path)

    pcd_files = glob(osp.join(root_path, '*/*.pcd'))

    assert len(pcd_files) >= 0, print(
        f"{len(pcd_files)}")

    zc.ROOTDIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, 'bin'), exist_ok=True)

    pkl_file_name = osp.join(out_dir, f'dummy_infos.pkl')
    zc.generate_pickle([None for _ in range(len(pcd_files))], pcd_files, pkl_file_name, num_workers=workers)


def zc_semantic_data_prep(
            root_path,
            out_dir,
            info_prefix=None,
            pcd_prefix="pcd", 
            training_ratios = 0.9,
            workers=4,
            with_plain=False):
    """
    root_path-> pcd-> xxx.pcd...

    TODO: deal with road plain
    """
    import mmcv
    import numpy as np
    import math
    import os
    from glob import glob
    from tools.data_converter import zc_semantic_converter as zc_semantic
    from tqdm import tqdm

    assert osp.exists(root_path)
    assert training_ratios <= 1.0
    
    info_files = glob(osp.join(root_path, info_prefix, '*.json'))
    ori_pcd_files = glob(osp.join(root_path, pcd_prefix, '*.pcd'))
    
    pcd_files = []

    if len(info_files) == 0:
        for pcd_file in tqdm(ori_pcd_files):
            pcd_files.append(pcd_file)
            info_files.append(None)
        # zc_semantic.generate_pickle([], pcd_files, 'testing_infos.pkl', num_workers=workers)
        # return
    else:
        # check info files and pcd files to make sure data are matched when list length is not equal
        for info_file in tqdm(info_files):
            name = osp.splitext(osp.basename(info_file))[0]
            pcd_path = osp.join(root_path, pcd_prefix, name + '.pcd')
            assert pcd_path in ori_pcd_files, print(f"{pcd_path} not exist!")
            pcd_files.append(pcd_path)

    # assert len(info_files) > 0 and len(pcd_files) > 0, print(
    #     f"{len(info_files)}{len(pcd_files)}")
    
    assert len(pcd_files) > 0, print(f"{len(pcd_files)}")

    zc_semantic.ROOTDIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, 'bin'), exist_ok=True)

    # generate pickles
    idxs = np.arange(len(pcd_files))
    np.random.shuffle(idxs) 
    splits = ['training', 'testing']

    train_numbers = int(training_ratios * len(pcd_files))
    # may skip some files
    other_numbers = math.floor(len(pcd_files) - train_numbers / (len(splits) - 1))

    cur_numbers = 0
    print(f"Start generate {len(splits)} pickle files...")
    for split in splits:
        if split == 'training':
            numbers = train_numbers
        else:
            numbers = other_numbers

        split_idxs = idxs[cur_numbers:cur_numbers + numbers]

        split_infos = [info_files[idx] for idx in split_idxs]

        split_pcds = [pcd_files[idx] for idx in split_idxs]

        cur_numbers += numbers
        pkl_file_name = osp.join(out_dir, f'{split}_infos.pkl')

        zc_semantic.generate_pickle(split_infos, split_pcds, pkl_file_name, num_workers=workers)

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--num-points',
    type=int,
    default=-1,
    help='Number of points to sample for indoor datasets.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument('--train-ratio', type=float, default=0.9)
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            num_points=args.num_points,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'zc':
        zc_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers,
            training_ratios = args.train_ratio
        )
    elif args.dataset == 'zc_dummy':
        zc_data_dummy(
            root_path=args.root_path,
            out_dir=args.out_dir,
        ) 
    elif args.dataset == 'zc_semantic':
        zc_semantic_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers, 
            training_ratios = args.train_ratio)
    else:
        raise NotImplementedError
