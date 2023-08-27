export PATH_TO_MMDET=../../zc/
export TENSORRT_DIR=/home/jing/Workspace/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jing/anaconda3/lib/python3.9/site-packages/torch/lib/:${LD_LIBRARY_PATH}


# pp
#echo "################################## pp fp32 #######################################"
#${TENSORRT_DIR}/bin/trtexec --onnx=./work_dir/end2end.onnx \
#    --minShapes=voxels:5000x32x4,num_points:5000,coors:5000x4 \
#    --optShapes=voxels:40000x32x4,num_points:40000,coors:40000x4  \
#    --maxShapes=voxels:80000x32x4,num_points:80000,coors:80000x4  \
#    --plugins=/home/jing/anaconda3/envs/base-bak/lib/python3.9/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so 2>&1 | tee pp_fp32.log

# pp fp16
echo "################################## pp fp16 #######################################"
${TENSORRT_DIR}/bin/trtexec --onnx=./work_dir/end2end.onnx \
    --plugins=/home/jing/anaconda3/envs/base-bak/lib/python3.9/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so --fp16 \
    --minShapes=voxels:5000x32x4,num_points:5000,coors:5000x4 \
    --optShapes=voxels:40000x32x4,num_points:40000,coors:40000x4  \
    --maxShapes=voxels:80000x32x4,num_points:80000,coors:80000x4  \
    2>&1 | tee pp_fp16.log 


## union
#echo "################################## union fp32 #######################################"
#${TENSORRT_DIR}/bin/trtexec --onnx=./work_dir/end2end_union.onnx \
#    --plugins=/home/jing/anaconda3/envs/base-bak/lib/python3.9/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so \
#    --minShapes=voxels:5000x32x5,num_points:5000,coors:5000x4 \
#    --optShapes=voxels:40000x32x5,num_points:40000,coors:40000x4  \
#    --maxShapes=voxels:80000x32x5,num_points:80000,coors:80000x4  \
#    1>union_fp32.log 2>&1
#
## union
#echo "################################## union fp16 #######################################"
#${TENSORRT_DIR}/bin/trtexec --onnx=./work_dir/end2end_union.onnx \
#    --plugins=/home/jing/anaconda3/envs/base-bak/lib/python3.9/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so --fp16 \
#    --minShapes=voxels:5000x32x5,num_points:5000,coors:5000x4 \
#    --optShapes=voxels:40000x32x5,num_points:40000,coors:40000x4  \
#    --maxShapes=voxels:80000x32x5,num_points:80000,coors:80000x4  \
#    1>union_fp16.log 2>&1
#
