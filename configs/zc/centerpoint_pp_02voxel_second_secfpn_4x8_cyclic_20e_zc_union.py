# 指定继承的基础配置，同名参数会覆盖掉基础配置中同名参数内容
_base_ = [
    # 指定基础dataset 配置
    '../_base_/datasets/zc_union.py',
    # 指定基础model 配置
    '../_base_/models/centerpoint_pp_02voxel_second_secfpn_zc_union.py',
    # 指定基础调度和runtime 配置，学习率、优化器，runner等等
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# 指定点云范围，超出部分不管，xmin, ymin, zmin, xmax, ymax,zmax 
point_cloud_range = [0, -20, -4, 120, 20, 4]
# For nuScenes we usually do 10-class detection
# 指定训练类名, 需要和制作好的dataset数据对应，数量可以少于dataset中的类目数，未包含的将被忽略掉
# 检测使用的类型
class_names = [
    'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Train'
]
# 分割使用的类型
seg_class_names = [
    'Kong', 'Unlabel', 'Road', 'Station', 'Fence']
#    none,      0,        1,      2,         3

# 在base 的model config基础上额外覆盖一些模型参数
model = dict(
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
    # semantic_code_weights： 回归semantic各个类别的权重，顺序为类型输入顺序，见seg_class_names
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range, 
                   code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                   semantic_code_weights=[0, 1, 1, 1.0, 2.0])),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], 
                  point_cloud_range=point_cloud_range, 
                  post_center_limit_range=point_cloud_range,
                  task='det')
    )
)

dataset_type = 'ZCUnionDataset'
data_root = '/home/jing/Data/data/20221220-det/' #'/home/jing/Data/data/20230617-det-merge/'
data_root_seg = '/home/jing/Data/data/seg/tony_all_data/' #'/home/jing/Data/data/seg/all_data/'
dataset_type_det = 'ZCDataset'
dataset_type_seg = 'ZCSemanticDataset'

samples_per_gpu=4,  # 单张 GPU 上的样本数
workers_per_gpu=4,  # 每张 GPU 上用于读取数据的进程数

# this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
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
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectSample', db_sampler=db_sampler),
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
    #dict(
    #    type='LoadPointsFromMultiSweeps',
    #    sweeps_num=9,
    #    use_dim=[0, 1, 2, 3, 4],
    #    file_client_args=file_client_args,
    #    pad_empty_sweeps=True,
    #    remove_close=True),
    dict(
        type='ObjectNumPtFilter',
        min_pt_num_dict=dict(Car=100, Pedestrian=100, Cyclist=100, Truck=100, Train=100)
    ),
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

# 分割训练数据预处理pipeline
train_pipeline_seg = [
    # 读取数据配置
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
    # 随机进行点云flip, flip_ratio_bev_horizontal: 水平翻转的概率, flip_ratio_bev_vertical: 垂直翻转的概率
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    # 点云范围过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # 点云point顺序打乱
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points'])
]
# 分割训练数据推理预处理pipeline
test_pipeline_seg = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
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
# 分割训练数据评测预处理pipeline
eval_pipeline_seg = [
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
]
# 检测训练数据训练集
data_det_train = dict(
                    type=dataset_type_det,
                    data_root=data_root,
                    ann_file=data_root + 'training_infos.pkl',
                    pipeline=train_pipeline,
                    # filter_empty_gt=False,
                    modality=input_modality,
                    classes=class_names,
                    test_mode=False,
                    box_type_3d='LiDAR',
                    # samples_per_gpu=samples_per_gpu,  # 单张 GPU 上的样本数
                    # workers_per_gpu=samples_per_gpu,  # 每张 GPU 上用于读取数据的进程数
                    )
# 检测训练数据验证集
data_det_val = dict(type=dataset_type_det,
                    data_root=data_root,
                    pipeline=test_pipeline, 
                    classes=class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    ann_file=data_root + 'testing_infos.pkl',
                    samples_per_gpu=4,  # 单张 GPU 上的样本数
                    )
# 检测训练数据测试集
data_det_test = dict(type=dataset_type_det,
                    data_root=data_root,
                    pipeline=test_pipeline, 
                    classes=class_names,
                    modality=input_modality,
                    filter_empty_gt=False,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    ann_file=data_root + 'testing_infos.pkl',
                    # samples_per_gpu=samples_per_gpu,  # 单张 GPU 上的样本数
                    # workers_per_gpu=samples_per_gpu,  # 每张 GPU 上用于读取数据的进程数
                    )
# 分割训练数据训练集
data_seg_train = dict(type=dataset_type_seg,
                    data_root=data_root_seg,
                    # 训练数据文件路径
                    ann_file=data_root_seg + 'training_infos.pkl',
                    pipeline=train_pipeline_seg,
                    classes=seg_class_names,
                    filter_empty_gt=False,
                    modality=input_modality,
                    test_mode=False,
                    box_type_3d='LiDAR',
                    # samples_per_gpu=samples_per_gpu,  # 单张 GPU 上的样本数
                    # workers_per_gpu=samples_per_gpu,  # 每张 GPU 上用于读取数据的进程数
                    )
# 分割训练数据验证集
data_seg_val = dict(type=dataset_type_seg,
                    data_root=data_root_seg,
                    pipeline=test_pipeline_seg,
                    classes=seg_class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    filter_empty_gt=False,
                    ann_file=data_root_seg + 'testing_infos.pkl',
                    # samples_per_gpu=samples_per_gpu,  # 单张 GPU 上的样本数
                    # workers_per_gpu=workers_per_gpu,  # 每张 GPU 上用于读取数据的进程数
                        )
# 分割训练数据测试集
data_seg_test = dict(type=dataset_type_seg,
                    data_root=data_root_seg,
                    pipeline=test_pipeline_seg, 
                    classes=seg_class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR',
                    ann_file=data_root_seg + 'testing_infos.pkl',
                    do_not_eval=False,
                    # samples_per_gpu=samples_per_gpu,  # 单张 GPU 上的样本数
                    # workers_per_gpu=workers_per_gpu,  # 每张 GPU 上用于读取数据的进程数
                    )

data = dict(
    samples_per_gpu=4,  # 单张 GPU 上的样本数
    workers_per_gpu=4,  # 每张 GPU 上用于读取数据的进程数
    type=dataset_type,
    train=[data_det_train, data_seg_train], 
    val=data_det_val,           # 目前只支持验证集为检测
    test=data_seg_test       
    )

evaluation = dict(interval=10, pipeline=eval_pipeline)

log_config = dict(
    interval=10)

runner = dict(max_epochs=120)

#lr = 1e-5
# fp16 settings
#fp16 = dict(loss_scale=64.)

