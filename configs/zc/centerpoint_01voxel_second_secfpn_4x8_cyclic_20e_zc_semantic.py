# 指定继承的基础配置，同名参数会覆盖掉基础配置中同名参数内容
_base_ = [
    # 指定基础dataset 配置
    '../_base_/datasets/zc_semantic.py',
    # 指定基础model 配置
    '../_base_/models/centerpoint_01voxel_second_secfpn_zc_semantic.py',
    # 指定基础调度和runtime 配置，学习率、优化器，runner等等
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# 指定点云范围，超出部分不管，xmin, ymin, zmin, xmax, ymax,zmax 
point_cloud_range = [0, -20.8, -4, 121.6, 20.8, 4]

# 指定训练类名, 需要和制作好的dataset数据对应，数量可以少于dataset中的类目数，未包含的将被忽略掉
class_names = [
    'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
]
seg_class_names = [
    'Kong', 'Unlabel', 'Road', 'Station', 'Fence']
#    none,      0,        1,      2,         3


# 在base 的model config基础上额外覆盖一些模型参数
model = dict(
    # 处理的点云范围
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    # model training and testing settings
    # code_weights: 回归各个属性的权重
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range, semantic_code_weights=[0, 1, 1, 10.0, 10.0])),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], 
                  point_cloud_range=point_cloud_range,
                  code_weights=[0, 1,  1, 1.0, 1.0])))


dataset_type = 'ZCSemanticDataset'
data_root = '/home/jing/Data/data/seg/all_data/'
#data_root = '/home/jing/Data/data/seg/test/'
#data_root = '/home/jing/Data/data/20221220-lidar-camera/'
#data_root = '/mnt/c/Users/xing/Downloads/test_pcd_json_20221220/'

file_client_args = dict(backend='disk')

# 训练数据预处理相关
train_pipeline = [
    # 读取数据配置
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    #dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # 旋转和缩放数据增强, rot_range 是旋转角度范围，scale_ratio_range 是scale 的范围, translation_std 是平移的std
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.1, 0.1],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    # 随机进行点云flip, flip_ratio_bev_horizontal: 水平翻转的概率, flip_ratio_bev_vertical: 垂直翻转的概率
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    # 点云范围过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    # 点云point顺序打乱
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
    #dict(
    #    type='LoadPointsFromMultiSweeps',
    #    sweeps_num=9,
    #    use_dim=[0, 1, 2, 3, 4],
    #    file_client_args=file_client_args,
    #    pad_empty_sweeps=True,
    #    remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        # eval 时点云scale 固定, 不进行变化
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
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
    #dict(
    #    type='LoadPointsFromMultiSweeps',
    #    sweeps_num=9,
    #    use_dim=[0, 1, 2, 3, 4],
    #    file_client_args=file_client_args,
    #    pad_empty_sweeps=True,
    #    remove_close=True),
    dict(
    type='MultiScaleFlipAug3D',
    img_scale=(1333, 800),
    pts_scale_ratio=1,
    flip=False,
    transforms=[
        # dict(
        #     type='GlobalRotScaleTrans',
        #     rot_range=[0, 0],
        #     scale_ratio_range=[1., 1.],
        #     translation_std=[0, 0, 0]),
        # dict(type='RandomFlip3D'),
        dict(
            type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])

    # dict(
    #     type='DefaultFormatBundle3D',
    #     class_names=class_names,
    #     with_label=False),
    # dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=4,  # 单张 GPU 上的样本数
    workers_per_gpu=4,  # 每张 GPU 上用于读取数据的进程数
    train=dict(
        #type='CBGSDataset', # TODO add CBGS
        #dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 训练数据文件路径
        ann_file=data_root + 'training_infos.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        #use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    #),
    # val 数据文件路径
    val=dict(pipeline=test_pipeline, classes=class_names,
            ann_file=data_root + 'testing_infos.pkl',
             ),
    # test 数据文件路径
    test=dict(pipeline=test_pipeline, classes=class_names,
            ann_file=data_root + 'testing_infos.pkl',
             # do_not_eval=True, #打开则不进行eval，可对原始数据进行推理
            ))

evaluation = dict(interval=10, pipeline=eval_pipeline)

log_config = dict(
    interval=1)

runner = dict(max_epochs=80)

#lr = 1e-5
# fp16 settings
#fp16 = dict(loss_scale=64.)

