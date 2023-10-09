set -x

TENSORRT_DIR=/data/TensorRT-8.4.3.1/
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

cp ./../mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so ./centerpoint/src/centerpoint/3rd_party/

${TENSORRT_DIR}/bin/trtexec --onnx=./centerpoint/src/centerpoint/model/end2end_union.onnx --fp16 \
        --minShapes=voxels:5000x32x4,num_points:5000,coors:5000x4 \
        --optShapes=voxels:20000x32x4,num_points:20000,coors:20000x4  \
        --maxShapes=voxels:40000x32x4,num_points:40000,coors:40000x4  \
        --plugins=./../mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so \
        --saveEngine=./centerpoint/src/centerpoint/model/end2end.plan \
        1>fp16_15w.txt 2>&1

