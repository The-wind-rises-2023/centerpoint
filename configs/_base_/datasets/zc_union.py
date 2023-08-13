# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# min x y z -> max x y z
point_cloud_range = [0, -20, -4, 120, 20, 4]

# use class names in zc_dataset
class_names = [
    'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
]
seg_class_names = [
    'Kong', 'Unlabel', 'Road', 'Station', 'Fence']
#    none,      0,        1,      2,         3

dataset_type = 'ZCUnionDataset'
dataset_type_det = 'ZCDataset'
dataset_type_seg = 'ZCSemanticDataset'
data_root = 'data/zc/'
data_root_seg = 'data/zc_seg/'

# this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))

# this for kitti
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'training_infos.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            Car=5, Pedestrian=5, Cyclist=5, Truck=5, Train=5)),
    classes=class_names,
    sample_groups=dict(
        Car=12, Pedestrian=6, Cyclist=6, Truck=12, Train=12),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    file_client_args=file_client_args)


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # for kitti
    #dict(type='ObjectSample', db_sampler=db_sampler),
    #dict(
    #    type='ObjectNoise',
    #    num_try=100,
    #    translation_std=[1.0, 1.0, 0.5],
    #    global_rot_range=[0.0, 0.0],
    #    rot_range=[-0.78539816, 0.78539816]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
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
    # test aug
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
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data_det_train = dict(
                    type=dataset_type_det,
                    data_root=data_root,
                    ann_file=data_root + 'training_infos.pkl',
                    pipeline=train_pipeline,
                    # filter_empty_gt=False,
                    modality=input_modality,
                    classes=class_names,
                    test_mode=False,
                    box_type_3d='LiDAR'
                    )
data_det_val = dict(type=dataset_type_det,
                    data_root=data_root,
                    pipeline=test_pipeline, 
                    classes=class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    ann_file=data_root + 'testing_infos.pkl')
data_det_test = dict(type=dataset_type_det,
                    data_root=data_root,
                    pipeline=test_pipeline, 
                    classes=class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    ann_file=data_root + 'testing_infos.pkl')
data_seg_train = dict(type=dataset_type_seg,
                    data_root=data_root_seg,
                    # 训练数据文件路径
                    ann_file=data_root_seg + 'training_infos.pkl',
                    pipeline=train_pipeline,
                    classes=seg_class_names,
                    modality=input_modality,
                    test_mode=False,
                    box_type_3d='LiDAR'
                    )
data_seg_val = dict(type=dataset_type_seg,
                    data_root=data_root_seg,
                    pipeline=test_pipeline, 
                    classes=seg_class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    ann_file=data_root_seg + 'testing_infos.pkl',
                        )
data_seg_test = dict(type=dataset_type_seg,
                    data_root=data_root_seg,
                    pipeline=test_pipeline, 
                    classes=seg_class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    ann_file=data_root_seg + 'testing_infos.pkl',
                    do_not_eval=False,
                    )

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[data_det_train, data_seg_train],
    val=data_det_val, 
    test=data_det_test, 
    )


evaluation = dict(interval=1, pipeline=eval_pipeline)