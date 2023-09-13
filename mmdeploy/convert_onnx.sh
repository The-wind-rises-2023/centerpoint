export PATH_TO_MMDET=../../zc/
export TENSORRT_DIR=/home/jing/Workspace/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

python ./tools/deploy.py \
   configs/mmdet3d/voxel-detection/centerpoint_union_pp_01voxel_second_4x8.py \
   $PATH_TO_MMDET/work_dirs/centerpoint_pp_01voxel_second_secfpn_4x8_cyclic_20e_zc_union/centerpoint_pp_01voxel_second_secfpn_4x8_cyclic_20e_zc_union.py \
   $PATH_TO_MMDET/work_dirs/centerpoint_pp_01voxel_second_secfpn_4x8_cyclic_20e_zc_union/epoch_1.pth \
   1671124327075438.bin \
   --work-dir work_dir \
   --device cuda:0


   #1671550233137085_dim5.bin \
