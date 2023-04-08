_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_zc.py',
    '../_base_/datasets/zc.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -20, -4, 120, 20, 4]

# dataset settings
data_root = '/home/jing/Data/data/20221220-lidar-camera/'
#data_root = '/mnt/c/Users/xing/Downloads/test_pcd_json_20221220/'
class_names = [
    'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
]
# PointPillars adopted a different sampling strategies among classes

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/kitti/':
#         's3://openmmlab/datasets/detection3d/kitti/',
#         'data/kitti/':
#         's3://openmmlab/datasets/detection3d/kitti/'
#     }))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'training_db_infos.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=10, Pedestrian=10, Cyclist=10, Truck=10, Train=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15, Truck=15, Train=15),
    #'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    #use_ground_plane=False,
    file_client_args=file_client_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    #dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False), # TODO,should add ground_plane and db_info
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5), # got some bug
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.1, 0.1],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=8,  # 单张 GPU 上的样本数
    workers_per_gpu=8,  # 每张 GPU 上用于读取数据的进程数
    #train=dict(dataset=dict(pipeline=train_pipeline, classes=class_names)),
    train=dict(pipeline=train_pipeline, classes=class_names, data_root=data_root,
               ann_file=data_root + 'training_infos.pkl',),
    val=dict(pipeline=test_pipeline, classes=class_names, data_root=data_root,
             ann_file=data_root + 'testing_infos.pkl',),
    #ann_file=data_root + 'training_infos.pkl',),
    test=dict(pipeline=test_pipeline, classes=class_names, data_root=data_root,
              ann_file=data_root + 'testing_infos.pkl',))
        #ann_file=data_root + 'training_infos.pkl',))

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
runner = dict(max_epochs=80)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=2)

# fp16 settings
fp16 = dict(loss_scale=64.)
