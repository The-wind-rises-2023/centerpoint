cuda_path='/usr/local/cuda'

export TENSORRT_DIR=/home/jing/Workspace/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=${TENSORRT_DIR}:${LD_LIBRARY_PATH}

mkdir build
cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=trt -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_TOOLKIT_ROOT_DIR=${cuda_path} \
        ..

make -j16

