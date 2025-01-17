_base_ = ['./voxel-detection_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 32, fp16_mode=True),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    min_shape=[5000, 32, 4],
                    opt_shape=[60000, 32, 4],
                    max_shape=[80000, 32, 4]),
                num_points=dict(
                    min_shape=[5000], opt_shape=[60000], max_shape=[80000]),
                coors=dict(
                    min_shape=[5000, 4],
                    opt_shape=[60000, 4],
                    max_shape=[80000, 4]),
            ))
    ])
