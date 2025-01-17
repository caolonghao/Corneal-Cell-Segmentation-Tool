# 细胞图像边缘检测与分割

这个项目使用深度学习方法（Dense U-Net）对细胞显微图像进行边缘检测和细胞分割。

## 功能特点

- 使用Dense U-Net进行细胞边缘检测
- 支持批量处理图像
- 自动进行图像预处理和归一化
- 提供后处理功能（骨架化、膨胀等形态学操作）
- 支持细胞区域标记和连通域分析

## 环境要求

- Python 3.x
- TensorFlow
- OpenCV
- scikit-image
- numpy
- imageio

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备模型权重文件
   - 将训练好的权重文件放在 `weights` 文件夹下
   - 需要的文件：
     - `weights_edges_epoch150.h5`
     - `weights_ROI_epoch150.h5`

2. 运行程序
```bash
python main.py -i <输入文件夹> -o <输出文件夹> -t <阈值>
```

参数说明：
- `-i` 或 `--input_folder`：输入图像文件夹路径（默认：./data/new_crop）
- `-o` 或 `--output_folder`：输出结果保存路径（默认：./data/new_crop_edge）
- `-t` 或 `--threshold`：边缘检测阈值（默认：0.5）

## 输出文件

程序会在输出文件夹中生成两种类型的结果：
1. `*_edge_binary.png`：边缘检测的二值化结果
2. `*_cell_label.png`：细胞区域标记结果

## 模型架构

- 网络类型：Dense U-Net
- 参数量：约7.1M
- 图像输入尺寸：512 x 512 x 1

## 注意事项

- 输入图像会被自动归一化到0-255范围
- 程序会自动创建输出文件夹（如果不存在）
- 建议根据实际图像情况调整阈值参数
