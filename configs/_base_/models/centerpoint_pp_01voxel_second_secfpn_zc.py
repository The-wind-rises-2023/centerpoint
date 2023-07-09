# xyz方向上划分格子的尺寸,通常尺寸越小精度越高，计算复杂度也更高
voxel_size = [0.1, 0.1, 8]
model = dict(
    type='CenterPoint',
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

    # backbone 参数， 64 对应pillar 中每个pillar特征维度, out_channels 表示输出3层featuremap，每个featuremap 的channel数填在out_channels 中, layer_nums 是输出3层featuremap 时经过的基础层数(基础层每个网络可能不一样，可以是单conv bn rulu层，也可以是basic resbolock层， layer_strides 表示输出的3层featuremap下采样倍数，2，2，2 意味着3层featuremap 的h、w 分别为原始H、W的 1/2 、1/4、1/8
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    # fpn 层，融合上面backbone 的多层featuremap,in_channels 和上面backbone的out_channels 对应，out_channels 表示当前fpn中3层经过处理后的channel数, upsample_strides 表示输入的3层上采样的倍数， 0.5 1 2 意味着上采样0.5倍（等于下采样2倍），上采样1倍（大小不变），上采样2倍， 输入的3层本来为 1/2，1/4，1/8，净多上采样后变为 1/4， 1/4， 1/4， 然后会将3层featuremap concat 到一起， 计算后为 n * c * H/4 * W/4 （H、W为原始featuremap大小）
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

        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)), share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            # 后处理的中心范围
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            # heatmap输出的阈值
            score_threshold=0.1,
            # 输入到head 的featuremap 下采样倍数，敌营secondfpn 中的 1/4 的下采样featuremap， 因此是4
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        # 回归和分类loss设置
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            #point_cloud_range / voxel_size 计算，顺序为x、y、z 
            grid_size=[1200, 400, 1],
            voxel_size=voxel_size,
            # 同上面的 out_size_factor 
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # head 中每个属性对应loss weight
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            # 同上面的 out_size_factor 
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            # 旋转nms
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
