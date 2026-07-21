# EEGNet 脑电信号分类实验报告

## 1. 任务说明

本实验复现 EEGNet 论文（Lawhern et al., 2018）中的模型结构，并在公开脑电数据集上进行训练与评估。

```text
输入：多通道脑电信号 (1 × C × T)
输出：脑电状态/任务类别预测
评估指标：分类准确率（Accuracy）、Cohen's Kappa、混淆矩阵
```

## 2. 数据集

- 数据集名称：
- 数据集来源：
- 脑电范式：（运动想象 / P300 / ERN / 情绪识别）
- 通道数（C）：
- 时间采样点数（T）：
- 采样率：
- 类别数：
- 被试数：
- 训练集样本数：
- 验证集样本数（划分比例）：
- 测试集样本数：
- 使用设备：CPU / GPU
- 总训练耗时：

## 3. 数据预处理

### 3.1 数据集初始状态

请说明所使用的数据集在上手时的状态（是否已经过预处理，已经包含了哪些处理）：

```text
（例如：BCI Competition IV 2a 数据集已经过 0.5-100 Hz 带通滤波和 50 Hz 陷波滤波，
本实验在此基础上进一步做了……）
```

### 3.2 预处理流程

请在下表中逐项记录你所做的预处理步骤及参数：

| 步骤 | 预处理操作 | 参数设置 | 选用理由/说明 |
|---|---|---|---|
| 1 | 重参考（Re-referencing） | 参考方式： | |
| 2 | 带通滤波（Bandpass Filtering） | 高通= Hz, 低通= Hz, 滤波器类型= | |
| 3 | 陷波滤波（Notch Filtering） | 频率= Hz（国内 50 Hz） | |
| 4 | 坏导处理（Bad Channel） | 检测方法/处理方式： | |
| 5 | 伪迹去除（Artifact Removal） | 方法（ICA/AutoReject/阈值拒绝/无）： | |
| 6 | 分段（Epoching） | tmin= s, tmax= s, 时间点数 T= | |
| 7 | 基线校正（Baseline Correction） | 基线窗口： | |
| 8 | 归一化/标准化 | 方式=, 统计量来源：仅训练集 / 全部数据 | |
| 9 | 验证集划分 | 划分比例=, 划分方式（随机/按被试）： | |

### 3.3 与论文预处理方式的对比

请说明你的预处理流程与 EEGNet 论文中预处理方式的异同：

```text
（在这里填写）
```

## 4. 模型结构

请画出你实现的 EEGNet 结构（标注各层具体参数）：

```text
Input (1 × C × T)

--- Block 1 ---
  -> Conv2d (in=1, out=F1= , kernel=(1, F2= ), padding= )
  -> BatchNorm2d
  -> DepthwiseConv2d (in= , out= , kernel=(C, 1), groups= )
  -> BatchNorm2d
  -> ELU
  -> AveragePool2d (kernel=(1, ))
  -> Dropout (p= )

--- Block 2 ---
  -> SeparableConv2d:
      DepthwiseConv2d (in= , out= , kernel=(1, ), groups= )
      PointwiseConv2d (in= , out= , kernel=1)
  -> BatchNorm2d
  -> ELU
  -> AveragePool2d (kernel=(1, ))
  -> Dropout (p= )

--- Classifier ---
  -> Flatten
  -> Linear (in= , out= )
```

模型总参数量：

请说明各组件在 EEGNet 中的作用：

| 组件 | 在 EEGNet 中的作用 |
|---|---|
| Conv2d (temporal) | |
| DepthwiseConv2d (spatial) | |
| SeparableConv2d | |
| BatchNorm2d | |
| ELU | |
| AveragePool2d | |
| Dropout | |
| Linear | |

## 5. 训练设置

| 配置 | 数值 |
|---|---|
| epochs | |
| batch size | |
| optimizer | Adam |
| learning rate | |
| weight decay | |
| loss function | CrossEntropyLoss |
| learning rate scheduler | CosineAnnealing / ReduceLROnPlateau / 无 |
| F1 | |
| D | |
| F2 | |
| dropout rate | |
| device | CPU / GPU |

## 6. 训练过程

### 6.1 Loss 与 Accuracy 记录

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| ... | | | | |

请粘贴 loss 与 accuracy 曲线图。

请简要描述训练过程是否正常（loss 是否下降、accuracy 是否上升、是否有过拟合迹象）：

```text
（在这里填写）
```

## 7. 测试结果

### 7.1 整体性能

| 指标 | 结果 |
|---|---|
| Test Accuracy | |
| Test Loss | |
| Cohen's Kappa | |
| Random Baseline | （100 / 类别数）% |

### 7.2 各类别性能

| 类别 | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 类别 1 | | | | |
| 类别 2 | | | | |
| ... | | | | |

### 7.3 混淆矩阵

请粘贴混淆矩阵图片。

请分析结果是否达到预期，并与论文中报告的结果进行对比：

```text
（在这里填写）
```

## 8. 预测结果展示

至少展示 5 个测试样本的预测结果。

| 样本编号 | 真实标签 | 预测标签 | 是否正确 | 置信度 |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

请粘贴预测可视化的图片。

## 9. 特征可视化

可选用 t-SNE、PCA 等方法对模型提取的特征进行降维可视化：

```text
（在这里粘贴 t-SNE / PCA 可视化图片并简要分析）
```

## 10. 超参数对比实验

尝试调整至少两项超参数，并记录结果对比：

| 实验 | 超参数变更 | Test Accuracy | Kappa |
|---|---|---|---|
| Baseline | F1=8, D=2 | | |
| 实验 A | | | |
| 实验 B | | | |

请简要分析哪个超参数对结果影响最大，以及可能的原因：

```text
（在这里填写）
```

## 11. 问题与改进

请简要说明：

- 遇到了哪些问题（数据维度不匹配、深度可分离卷积中的 groups 参数设置、loss 不下降、过拟合等）；
- 如何定位并解决这些问题；
- 如果继续改进，可以从哪些方面入手，例如调整 F1/D/F2 参数、使用不同的滤波频段、增加被试间验证、使用数据增强（如滑动窗口裁剪）、尝试 EEGSym 等更新架构。

```text
（在这里填写）
```

## 12. Git 提交记录

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```

## 13. 参考资料

- Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*, 15(5), 056013.
- EEGNet 官方实现：https://github.com/vlawhern/arl-eegmodels
- Braindecode：https://github.com/braindecode/braindecode
- MOABB：https://neurotechx.github.io/moabb/
