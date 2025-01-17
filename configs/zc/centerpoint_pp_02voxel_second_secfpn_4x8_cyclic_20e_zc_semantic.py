_base_ = [
    '../_base_/datasets/zc_semantic.py',
    '../_base_/models/centerpoint_pp_02voxel_second_secfpn_zc_semantic.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
#point_cloud_range = [0, -20.8, -4, 121.6, 20.8, 4]
point_cloud_range = [0, -20, -4, 120, 20, 4]
# For nuScenes we usually do 10-class detection
class_names = [
    'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
]
seg_class_names = [
    'Kong', 'Unlabel', 'Road', 'Station', 'Fence']
#    none,      0,        1,      2,         3


model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    ##TODO: @zxj head
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range, code_weights=[0, 1, 1, 10.0, 10.0])),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], 
                  point_cloud_range=point_cloud_range,
                  code_weights=[0, 1,  1, 1.0, 1.0])))


dataset_type = 'ZCSemanticDataset'
#data_root = '/home/jing/Data/data/seg/all_data/'
data_root = '/home/jing/Data/data/seg/test/'

file_client_args = dict(backend='disk')

#TODO with this
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'gt_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
    filter_by_min_points=dict(Car=10, Pedestrian=10, Cyclist=10, Truck=10, Train=10)),
    classes=class_names,
    sample_groups=dict(Car=5, Pedestrian=5, Cyclist=5, Truck=5, Train=5),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.1, 0.1],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
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
        ]),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# eval_pipeline = train_pipeline.copy()

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
    type='MultiScaleFlipAug3D',
    img_scale=(1333, 800),
    pts_scale_ratio=1,
    flip=False,
    transforms=[
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
    samples_per_gpu=16,  # 单张 GPU 上的样本数
    workers_per_gpu=16,  # 每张 GPU 上用于读取数据的进程数
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'training_infos.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        #use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(pipeline=test_pipeline, classes=class_names,
            ann_file=data_root + 'testing_infos.pkl',
             ),
    test=dict(pipeline=test_pipeline, classes=class_names,
            ann_file=data_root + 'testing_infos.pkl',
            do_not_eval=True
              ))

evaluation = dict(interval=10, pipeline=eval_pipeline)

log_config = dict(
    interval=1)

runner = dict(max_epochs=80)

#lr = 1e-5
# fp16 settings
#fp16 = dict(loss_scale=64.)

