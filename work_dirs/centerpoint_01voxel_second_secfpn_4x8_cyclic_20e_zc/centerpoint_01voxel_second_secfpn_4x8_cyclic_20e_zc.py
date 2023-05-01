point_cloud_range = [0, -20, -4, 120, 20, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train']
dataset_type = 'ZCDataset'
data_root = '/home/jing/Data/data/20221220-det-merge/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
db_sampler = dict(
    data_root='/home/jing/Data/data/20221220-det-merge/',
    info_path='/home/jing/Data/data/20221220-det-merge/gt_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            Car=10, Pedestrian=10, Cyclist=10, Truck=10, Train=10)),
    classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
    sample_groups=dict(Car=5, Pedestrian=5, Cyclist=5, Truck=5, Train=5),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        file_client_args=dict(backend='disk')),
    file_client_args=dict(backend='disk'))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='/home/jing/Data/data/20221220-det-merge/',
            info_path=
            '/home/jing/Data/data/20221220-det-merge/gt_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    Car=10, Pedestrian=10, Cyclist=10, Truck=10, Train=10)),
            classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
            sample_groups=dict(
                Car=5, Pedestrian=5, Cyclist=5, Truck=5, Train=5),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=[0, 1, 2, 3],
                file_client_args=dict(backend='disk')))),
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
    dict(type='PointsRangeFilter', point_cloud_range=[0, -20, -4, 120, 20, 4]),
    dict(type='ObjectRangeFilter', point_cloud_range=[0, -20, -4, 120, 20, 4]),
    dict(
        type='ObjectNameFilter',
        classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train']),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -20, -4, 120, 20, 4]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -20, -4, 120, 20, 4]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='ZCDataset',
        data_root='/home/jing/Data/data/20221220-det-merge/',
        ann_file='/home/jing/Data/data/20221220-det-merge/training_infos.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='ObjectSample',
                db_sampler=dict(
                    data_root='/home/jing/Data/data/20221220-det-merge/',
                    info_path=
                    '/home/jing/Data/data/20221220-det-merge/gt_dbinfos_train.pkl',
                    rate=1.0,
                    prepare=dict(
                        filter_by_difficulty=[-1],
                        filter_by_min_points=dict(
                            Car=10,
                            Pedestrian=10,
                            Cyclist=10,
                            Truck=10,
                            Train=10)),
                    classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
                    sample_groups=dict(
                        Car=5, Pedestrian=5, Cyclist=5, Truck=5, Train=5),
                    points_loader=dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=4,
                        use_dim=[0, 1, 2, 3],
                        file_client_args=dict(backend='disk')))),
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
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -20, -4, 120, 20, 4]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[0, -20, -4, 120, 20, 4]),
            dict(
                type='ObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train']),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist', 'Truck',
                             'Train']),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type='ZCDataset',
        data_root='data/zc/',
        ann_file='/home/jing/Data/data/20221220-det-merge/testing_infos.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -20, -4, 120, 20, 4]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='ZCDataset',
        data_root='data/zc/',
        ann_file='/home/jing/Data/data/20221220-det-merge/testing_infos.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -20, -4, 120, 20, 4]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=4,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -20, -4, 120, 20, 4]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
                    ],
                    with_label=False),
                dict(type='Collect3D', keys=['points'])
            ])
    ])
voxel_size = [0.1, 0.1, 0.2]
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.1, 0.1, 0.2],
        max_voxels=(90000, 120000),
        point_cloud_range=[0, -20, -4, 120, 20, 4]),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=512,
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Truck']),
            dict(num_class=1, class_names=['Train'])
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[0, -20, -4, 120, 20, 4],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=7,
            pc_range=[0, -20]),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0],
            point_cloud_range=[0, -20, -4, 120, 20, 4])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[0, -20, -4, 120, 20, 4],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            pc_range=[0, -20])))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_ids = [0]
