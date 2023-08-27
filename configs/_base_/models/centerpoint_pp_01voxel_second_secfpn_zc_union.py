voxel_size = [0.1, 0.1, 8]
model = dict(
    type='CenterPointUnion',
    # max_num_points : voxel 中最多point数, voxel_size 的尺寸（x、y、z), train 和 test 最多的voxel数
    # pillar 特征提取参数, max_num_points 代表每个pillar柱子中最多多少个点, max_voxels 代表训练和测试时最多选择多少pillar个数, 计算后得到b * n * 4 的数据
    pts_voxel_layer=dict(
        max_num_points=32, voxel_size=voxel_size, max_voxels=(60000, 80000)),

    # 计算pillar 特征, in_channels 代表每个点输入数据维度,feat_channels 代表每个pillar 计算完后的特征维度, 计算后得到 b * n * 64 的数据
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -20, -4, 120, 20, 4]),

    # 将计算好的pillar重新投影到bev视角下的grid中，output_shape (y, x) 需要根据 point_cloud_range / voxel_size 计算， 
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 1200]),

    # backbone. 经过backbone 后包含两层featurempa， 尺寸分别为原始featuremap 1/8 和 1/16
    # out_channels: 输出2层，通道数分别为128，256
    # layer_nums: 输出的2层featuremap 分别使用了5层基础层
    # layer_strides: 输出的2层的conv stride
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    # fpn neck, 经过fpn后是concat 后的 1/8 原始featuremap 大小的featuremap
    # 输入的2层feautremap 在 fpn中的输出通道数
    # upsample_strides: 2层featuremap 的上采样倍数，1表示保持原始尺寸，2 表示上采样2倍,
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # head 参数, in_channels 对应上面的out_channels. tasks 是针对不同的任务的输出头，每个task 会输出heatmap 和 回归值, 对应是否有障碍物以及障碍物的位置、尺寸等
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        #common_heads=dict(
        #    reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    pts_semantic_head=dict(
        type='SemanticHead',
        in_channels=sum([128, 128, 128]),
        out_channels=[96, 24],
        num_classes=5,
        seg_score_thr=0.3,
        loss_seg=dict(type='FocalLoss', loss_weight=1)
        ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[1200, 400, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            semantic_code_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
            task='det')),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            task='det')))
