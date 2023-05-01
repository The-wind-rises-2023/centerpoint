## 数据准备
1. 将对应数据按照预先约定的形式进行放置在数据目录下（例如$data_dir），子目录结构如下：
```
├── jpg  # 图片数据
├── json # 标注数据
└── pcd  # 原始点云，pcd格式
```
2. 在工程目录下，执行如下脚本
```
cd ${WorkDir}/
python tools/create_data.py zc --root-path $data_dir --out-dir $data_dir --extra-tag json
# --root-path 指原始数据的地址
# --out-dir 输出路径地址
# --extra-tag 真值存放的相对路径
```
3. 检查成功生成后的路径
```
├── bin                      # 生成的二进制点云数据，用于训练时加速读取
├── gt_dbinfos_train.pkl     # 训练时增强用的obj样本, pkl用于训练时加速读取
├── gt_gt_database           # 训练时增强用的obj样本
├── testing_infos.pkl        # 测试集数据, pkl用于加速读取
├── training_infos.pkl       # 训练集数据, pkl用于加速读取
├── jpg
├── json
└── pcd
```
## 训练
0. 修改数据地址
```
# 修改配置中的data_root, for example: centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc.py
data_root = $data_dir
```
1. 训练指令
```
python tools/train.py xxx_config
# for example
python tools/train.py configs/zc/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc.py
```
输出的log、epoch默认存储在work_dirs/xxx_config目录下，可通过传参 --work-dir $dir 修改

## 测试
0. 修改数据地址：同训练
1. 测试指令
```
python tools/test.py xxx_config xxx_epoch.pth --eval zc
# for example
python tools/test.py configs/zc/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc.py work_dirs/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc/epoch_80.pth --eval zc
# 若需要可视化或者输出对应推理结果，加入下述参数
python tools/test.py configs/zc/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc.py work_dirs/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc/epoch_80.pth --eval zc --eval-options 'show=True' 'out_dir=./work_dirs/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_zc/eval/'
```
1.1 eval 后会在终端打印评测结果，如下：
```
Car bev_mAP40 : nan, Precision: 0.9543973941368078, Recall: 0.7855227882037533                                   │··················
Pedestrian bev_mAP40 : nan, Precision: 0.8305084745762712, Recall: 0.7903225806451613                            │··················
Cyclist bev_mAP40 : nan, Precision: 0.9153439153439153, Recall: 0.8564356435643564                               │··················
Truck bev_mAP40 : nan, Precision: 0.9411764705882353, Recall: 0.8205128205128205                                 │··················
Train bev_mAP40 : nan, Precision: 1.0, Recall: 0.9230769230769231     
```
1.2 打开show=True后，会弹出可视化窗口，如下图：
<div align="center">
<img src="docs/3rd/sample-vis.png" />
</div>
其中：绿色框为prediction，蓝色框为groundtruth

鼠标左键+拖动：翻转视角
滚轮+拖动：平移
滚轮上下：缩放
下一张：q / esc