# pytorch-superpoint

这是论文[SuperPoint: Self-Supervised Interest Point Detection and Description]( https://arxiv.org/abs/1712.07629 )的Paddle实现。


## 参考github repo与原始论文之间的差异
- *Descriptor loss*: 参考的github repo使用不同的方法描述loss，包括密集方法（如论文所示，但略有不同）和稀疏方法。参考的github repo的作者注意到稀疏损失可以更有效地收敛到类似的性能，因此这里默认设置是稀疏方法.

## HPatches上的结果
| 任务                                       | Homography estimation |      |      | Detector metric |      | Descriptor metric |                |
|-------------------------------------------|-----------------------|------|------|-----------------|------|-------------------|----------------|
|                                           | Epsilon = 1           | 3    | 5    | Repeatability   | MLE  | NN mAP            | Matching Score |
| 原始论文提供的superpoint模型                 | 0.310                  | 0.684 | 0.829 | 0.581          | 1.158 | 0.821              | 0.470           |
| 参考的[pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint) | 0.46                  | 0.75 | 0.81 | 0.63            | 1.07 | 0.78              | 0.42           |
| 根据pytorch-superpoint实现的superpoint_paddle模型 | 0.46                  | 0.75 | 0.81 | 0.63            | 1.07 | 0.78              | 0.36           |



## 安装
### 要求
- python == 3.8
- paddlepaddle >= 2.0 (tested in 2.1.2)
- cuda (tested in cuda11.0)


### 路径设置
- 数据集的路径 ($DATA_DIR), logs在 `setting.py` 中设置

### 数据集
数据集应下载到 $DATA_DIR 目录中. 合成shapes数据集也将在那里生成。文件夹结构应如下所示：

```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # 将自动生成
```

- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)



## 运行命令

### 1) 关于synthetic_shapes的MagicPoint训练

- 不需要下载synthetic data，将在第一次运行它时生成它.
- synthetic data以 `./datasets` 的形式导出. 您可以在 `settings.py` 中更改设置.

#### 运行命令

```
python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```

### 2) 在 MS-COCO 上导出 detectiion

这是 homography adaptation(HA) 的步骤，用于输出ground truth以进行联合训练.
- 确保配置文件中的预训练模型正确
- 确保COCO数据集唯一 '$DATA_DIR' (在setting.py中定义)
<!-- - 您可以通过编辑配置文件中的'task'来导出 hpatches 或 coco 数据集  -->

#### 运行命令:

#### 导出coco - 在训练集上
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### 导出coco - 在验证集上
- 在 'magicpoint_coco_export.yaml' 中 设置 'export_folder' 为 'val' 
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```


### 3) 在 MS-COCO/ KITTI 上训练 superpoint

需要fake ground truth标签来训练检测器detectors，标签可以从步骤 2) 导出，也可以从 [link](https://drive.google.com/drive/folders/1nnn0UbNMFF45nov90PJNnubDyinm2f26?usp=sharing) 下载. 然后，像往常一样，您需要在训练之前设置配置文件.

- config file配置文件
  - root: 指定您的root标签
  - root_split_txt: 放置 train.txt/ val.txt 分割文件的位置 (COCO不需要, KITTI需要)
  - labels: 从 homography adaptation 导出的标签
  - pretrained: 指定预训练模型 (可以从头开始训练)
  
- 'eval': 在训练期间打开验证/评估

#### 运行命令
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```


### 4) 导出/ 评估 HPatches 上的指标
- 使用预训练模型或在配置文件中指定模型
#### 导出
- 下载 HPatches 数据集 (上面的链接). 输入 $DATA_DIR .
  
- 导出 keypoints, descriptors, matching
##### 运行命令
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
#### 验证/评估

- 评估单应性估计 homography estimation/ 重复性 repeatability/ 匹配分数 matching scores ...
##### 运行命令
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```


## 预训练模型
### 来自 原始论文 的模型
```pretrained/superpoint_v1.pdparams```
### 目前paddle训练得到的模型
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pdparams```

## 训练及验证流程
我们在MS-COCO图像上训练Superpoint，在Hpatches数据集进行验证。

- 下载MS-COCO 2014数据集， 将train2014和val2014放在`./datasets/COCO/`目录下
  
- 下载HPatches数据集， 放入`./datasets/HPatches/`目录下
  --Hpatches包含116个场景，每个场景共6张图片，一个场景包含一张原图和5张对原图进行视角变换、明亮度变化的图片，一共580个图像对
  
- 下载ground truth标签（也可以由步骤1）生成和步骤2）导出），放入`./logs/magicpoint_synth20_homoAdapt100_coco_f1/`目录下
  
- 运行训练命令，将生成`./logs/superpoint_coco/`目录
  -- 训练结束后，将在`./logs/superpoint_coco/checkoints/`目录下保存训练好的权重文件
  
- 运行导出命令，导出 keypoints, descriptors, matching，将生成`superpoint_hpatches_test`
  -- 其中`./logs/superpoint_hpatches_test/predictions/`目录下为581个图像对的`.npz`文件
  
- 运行验证命令，将在`log/superpoint_hpatches_test`目录生成结果文件`result.npz`和`result.txt`
  -- 其中`result.txt`包含每个图像对的Repeatability、NN mAP、Matching Score结果以及580个图像对的平均Repeatability、NN mAP、Matching Score等评估指标

## 结果对比

### 加载paddle训练完的权重文件进行验证的结果`result.txt`
```
path: logs/superpoint_hpatches_test/predictions
output Images: False
repeatability threshold: 3
repeatability: 0.6318583662981305
localization error: 1.0711925585271342
Homography estimation: 
Homography threshold: [1, 3, 5, 10, 20, 50]
Average correctness: [0.46206897 0.75172414 0.8137931  0.86206897 0.89310345 0.93448276]
nn mean AP: 0.7839061867540921
matching score: 0.36575061417680643
```

### 加载pytorch训练完的权重文件进行验证的结果`result.txt`
```
path: logs/superpoint_hpatches_test/predictions
output Images: False
repeatability threshold: 3
repeatability: 0.6318583662981305
localization error: 1.0711925585271342
Homography estimation: 
Homography threshold: [1, 3, 5, 10, 20, 50]
Average correctness: [0.46206897 0.75172414 0.8137931  0.86206897 0.89310345 0.93448276]
nn mean AP: 0.7839061867540921
matching score: 0.4244876249370818
```

## 总结
- 对比`result.txt`的结果可知，除了指标matching score与pytorch-superpoint差距较大外（暂时没找到原因），其它指标与pytorch-superpoint一样。
- 原论文中，e=1时Homography estimation为0.310，由于参考的github repo在原始论文的基础上进行了改进，因为目标是与参考github repo的指标对齐，根据结果，e=1时paddle实现superpoint的Homography estimation为0.462。



