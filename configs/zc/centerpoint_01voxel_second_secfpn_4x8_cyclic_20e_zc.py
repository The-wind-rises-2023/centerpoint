# 指定继承的基础配置，同名参数会覆盖掉基础配置中同名参数内容
_base_ = [
    # 指定基础dataset 配置
    '../_base_/datasets/zc.py',
    # 指定基础model 配置
    '../_base_/models/centerpoint_01voxel_second_secfpn_zc.py',
    # 指定基础调度和runtime 配置，学习率、优化器，runner等等
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# 指定点云范围，超出部分不管，xmin, ymin, zmin, xmax, ymax,zmax 
point_cloud_range = [0, -20, -4, 120, 20, 4]

# 指定训练类名, 需要和制作好的dataset数据对应，数量可以少于dataset中的类目数，未包含的将被忽略掉
class_names = [
    'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
]


# 在base 的model config基础上额外覆盖一些模型参数
model = dict(
    # 处理的点云范围
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    # 指定centerpoint head 的task， 如下指定了5个task, 每个task head 只负责一个类别
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Truck']),
            dict(num_class=1, class_names=['Train'])
        ],
        # 每个task head 对应的loss weight，weight 越大则越希望提高某个类别效果，但可能导致其他类别效果下降
        task_weight=[2.0,1.5,1.0,1.0,1.0],
        # 每个task head 对应去回归或预测哪些属性, 第一位为维度，第二位为conv 层数
        # remove vel
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        # 编码3d box 的相关参数, 
        bbox_coder=dict(pc_range=point_cloud_range[:2],
                        post_center_range=[0, -20, -4, 120, 20, 4],
                        )),
    # model training and testing settings
    # code_weights: 回归head的各个属性的权重, 顺序为reg, height, dim, rot
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range, code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    # post_center_limit_range: 后处理时中心的范围
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], post_center_limit_range=point_cloud_range)))


dataset_type = 'ZCDataset'
data_root = '/home/jing/Data/data/20221220-det-merge/'
#data_root = '/home/jing/Data/data/20221220-lidar-camera/'
#data_root = '/mnt/c/Users/xing/Downloads/test_pcd_json_20221220/'

file_client_args = dict(backend='disk')

# 采样器配置，可以随机采样不同障碍物到当前点云中
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'gt_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
    # 只采样点云数大于一定值的障碍物
    filter_by_min_points=dict(Car=10, Pedestrian=10, Cyclist=10, Truck=10, Train=10)),
    classes=class_names,
    # 每个类别在采样多少个到当前帧点云中
    sample_groups=dict(Car=5, Pedestrian=5, Cyclist=5, Truck=5, Train=5),
    # 采样输入相关参数
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args))

# 训练数据预处理相关
train_pipeline = [
    # 读取数据配置
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
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
    # target 障碍物范围过滤
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # target 类别过滤
    dict(type='ObjectNameFilter', classes=class_names),
    # 点云point顺序打乱
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
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# eval_pipeline = train_pipeline.copy()

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
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
    samples_per_gpu=8,  # 单张 GPU 上的样本数
    workers_per_gpu=8,  # 每张 GPU 上用于读取数据的进程数
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
              ))

# 每隔4个epoch eval 一次
evaluation = dict(interval=4, pipeline=eval_pipeline)

# 打印log 间隔
log_config = dict(
    interval=1)

# 训练120个epoch
runner = dict(max_epochs=120)

#lr = 1e-5
# fp16 settings
#fp16 = dict(loss_scale=64.)

