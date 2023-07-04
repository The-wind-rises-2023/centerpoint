voxel_size = [0.1, 0.1, 0.2]
model = dict(
    type='CenterPoint',
    # max_num_points : voxel 中最多point数, voxel_size 的尺寸（x、y、z), train 和 test 最多的voxel数
    pts_voxel_layer=dict(
        max_num_points=32, voxel_size=voxel_size, max_voxels=(90000, 120000)),
    # num_features: 点的特征维度
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    # sparse conv 编码器, sparse_shape : z y x 顺序，point_cloud_range / voxel_size 计算
    # output_channels : 输出通道数.
    # encoder_channels : 多层encoder对应的通道conv 通道数, 同时会设置stride为2，featuremap 最终变为原来 1 / 8
    # 多层encoder 对应的padding 尺寸
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 400, 1200],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    # backbone. 经过backbone 后包含两层featurempa， 尺寸分别为原始featuremap 1/8 和 1/16
    # out_channels: 输出2层，通道数分别为128，256
    # layer_nums: 输出的2层featuremap 分别使用了5层基础层
    # layer_strides: 输出的2层的conv stride
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    # fpn neck, 经过fpn后是concat 后的 1/8 原始featuremap 大小的featuremap
    # 输入的2层feautremap 在 fpn中的输出通道数
    # upsample_strides: 2层featuremap 的上采样倍数，1表示保持原始尺寸，2 表示上采样2倍,
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # head 参数, in_channels 对应上面的out_channels. tasks 是针对不同的任务的输出头，每个task 会输出heatmap 和 回归值, 对应是否有障碍物以及障碍物的位置、尺寸等
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
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
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[1200, 400, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
