# EEGNet 脑电信号分类 —— 论文复现作业

## 1. 作业背景

本次作业是**脑电信号深度学习**系列的核心实践，目标是复现经典论文 **EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces**（Lawhern et al., 2018）。

EEGNet 是一种专为脑电信号设计的轻量级卷积神经网络，它利用深度可分离卷积（Depthwise Separable Convolution）大幅减少参数量，同时保持优异的分类性能。该网络已在运动想象（Motor Imagery）、P300 诱发电位、错误相关负波（ERN）等多种脑电范式上得到验证。

评价重点是**复现的准确性**、**对 EEGNet 各组件作用的理解**以及**实验结果的合理性分析**，不以绝对准确率作为唯一评价标准。

## 2. 任务目标

复现 EEGNet 论文中的模型结构，并在公开脑电数据集上进行训练与评估。

任务形式：

```text
输入：多通道脑电信号 (1 × Channels × TimePoints)
输出：脑电状态/任务类别预测
评估：分类准确率（Accuracy）、Kappa 值、混淆矩阵等
```

核心目标：

1. 理解脑电信号的数据格式和含义（通道数 × 时间点）；
2. 掌握 EEGNet 的核心组件：二维卷积、深度卷积（DepthwiseConv2d）、可分离卷积（SeparableConv2d）；
3. 以类的形式手动搭建 EEGNet 模型（不直接调用第三方 EEGNet 实现）；
4. 理解 ELU 激活函数、BatchNorm、Dropout、AveragePool2d 在脑电解码中的作用；
5. 实现训练循环（train loop）和评估循环（eval loop）；
6. 计算并记录 loss、accuracy、Kappa 值等指标；
7. 可视化训练过程、混淆矩阵及模型预测结果。

## 3. 数据集要求

使用公开脑电数据集，可以在以下数据集中选择，推荐BCI Competition IV 2a：

**推荐数据集：**

| 数据集 | 范式 | 通道数 | 类别数 | 被试数 |
|---|---|---|---|---|
| BCI Competition IV 2a | 运动想象 | 22 | 4 | 9 |
| BCI Competition IV 2b | 运动想象 | 3 | 2 | 9 |
| PhysioNet EEG Motor Imagery | 运动想象 | 64 | 4 | 109 |
| SEED | 情绪识别 | 62 | 3 | 15 |

- BCI Competition IV 数据集：https://www.bbci.de/competition/iv/
- PhysioNet 运动想象数据集：https://physionet.org/content/eegmmidb/
- Moabb 库可便捷加载多种脑电数据集：https://moabb.neurotechx.com/

要求：

- **必须划分验证集**：从训练集中划分至少 15%~20% 作为 validation set，用于监控过拟合；
- 对被试内（within-subject）和被试间（cross-subject）两种评估方式进行说明；
- 训练和验证阶段不使用测试集，测试集仅在最终评估时使用；
- 数据集文件不要提交到 Git 仓库（加入 `.gitignore`）。

**数据预处理要求**

脑电信号与图像数据不同，原始脑电包含大量生理伪迹和环境噪声，**预处理是复现实验的重要环节**，需要在报告中对所采用的预处理方式以及选用理由进行说明。

要求至少完成以下预处理步骤，并在实验报告中记录具体参数：

1. **重参考（Re-referencing）**：从记录参考转换到分析参考（如公共平均参考 CAR、双耳乳突参考等），报告中需说明所选参考方式及理由；
2. **带通滤波（Bandpass Filtering）**：设定高通和低通截止频率，保留与任务相关的频段，报告中需说明所选频段范围和滤波器参数；
3. **陷波滤波（Notch Filtering）**：去除工频干扰（国内为 50 Hz），报告中需注明是否使用及参数；
4. **伪迹去除（Artifact Removal）**：处理眼电、肌电等生理伪迹，可以采用 ICA、AutoReject、阈值拒绝等方式，报告中需说明所选方法；
5. **分段（Epoching）**：将连续数据按事件标记切分为固定长度 trial，报告中需记录时间窗口范围及对应的采样点数 T；
6. **基线校正（Baseline Correction）**：以事件前一段时间窗口为基线对各 trial 进行校正，报告中需记录基线窗口范围；
7. **归一化/标准化（Normalization/Standardization）**：对数据进行缩放以使模型训练稳定，报告中需说明归一化方式（如 Z-score），并明确统计量是仅基于训练集计算的。

额外注意事项（报告中需确认）：

- 检查所使用数据集的原始状态：部分公开数据集已做过部分预处理，需在报告中说明数据集初始状态以及你在此基础上额外做了哪些处理；
- 标准化时必须仅用训练集的均值和标准差去标准化验证集和测试集，避免数据泄露；
- 做公共平均参考（CAR）前应先处理坏导，避免坏导噪声污染所有通道。

## 4. 模型结构要求

需要以类的形式自行搭建 EEGNet 模型，参考论文 Table I 和 Table II 的结构。

### 4.1 参考 EEGNet 结构

```text
Input (1 × C × T)

--- Block 1 ---
  -> Conv2d (in=1, out=F1, kernel=(1, F2), padding='same')
  -> BatchNorm2d
  -> DepthwiseConv2d (in=F1, out=D*F1, kernel=(C, 1), groups=F1, bias=False)
  -> BatchNorm2d
  -> ELU Activation
  -> AveragePool2d (kernel=(1, 4))
  -> Dropout (p=0.25)

--- Block 2 ---
  -> SeparableConv2d (in=D*F1, out=F2, kernel=(1, 16), padding='same')
    (DepthwiseConv2d + PointwiseConv2d)
  -> BatchNorm2d
  -> ELU Activation
  -> AveragePool2d (kernel=(1, 8))
  -> Dropout (p=0.25)

--- Classifier ---
  -> Flatten
  -> Linear (hidden -> N_classes)
```

### 4.2 关键参数说明

| 参数 | 含义 | 典型值 |
|---|---|---|
| F1 | 时间滤波器数量 | 8 |
| D | 空间滤波器深度乘数 | 2 |
| F2 | 可分离卷积输出通道数 | 16 (F1*D) |
| C | EEG 通道数 | 由数据集决定 |
| T | 时间采样点数 | 由数据集决定 |
| kernel (1, F2) | 时间卷积核大小 | F2 = 采样率的一半（如 64 for 128Hz） |

### 4.3 可自行调整的部分

- 时间卷积核长度 F2；
- F1、D、F2 等超参数；
- Dropout 比率；
- 是否加入 L1/L2 正则化；
- 学习率调度策略。

**鼓励尝试不同配置，并在报告中对比效果。**

## 5. 训练要求

### 5.1 基本训练设置

```text
dataset: 上述推荐数据集之一
epochs: at least 100
batch size: 16 or 32
optimizer: Adam
learning rate: 1e-3 (Adam)
loss: CrossEntropyLoss
```

### 5.2 训练循环要求

每个 epoch 需要记录：

- 训练集 average loss
- 训练集 accuracy
- 验证集 average loss
- 验证集 accuracy
- 可选：Kappa 值、F1 分数等

### 5.3 评估要求

训练结束后在测试集上报告：

- Test Accuracy
- Kappa 值（Cohen's Kappa）
- 混淆矩阵
- 各类别的 Precision / Recall / F1

最低结果要求：

- 测试集 accuracy 应明显优于随机基线；
- 必须输出混淆矩阵和至少 5 个测试样本的预测可视化；
- 鼓励加入 t-SNE 等特征可视化，展示模型学到的表征。

## 6. 最低完成标准

1. 能够加载公开脑电数据集，正确划分 train / val / test；
2. 能够完成脑电数据预处理全流程（重参考、滤波、伪迹去除、分段、基线校正、标准化），并记录参数；
3. 能够自行搭建 EEGNet 模型（不使用第三方 EEGNet 实现）；
4. 能够完成训练循环并打印每个 epoch 的 loss 与 accuracy；
5. 能够在测试集上评估并报告 accuracy、Kappa 值、混淆矩阵；
6. 能够可视化至少 5 个测试样本的预测结果；
7. 能够对比至少两组不同超参数的实验结果。

## 7. 最终提交内容

1. **实现代码**

   需要包含数据加载、模型定义、训练脚本、评估脚本，代码应有适当注释。

2. **实验报告**

   使用 `report-template.md` 模板撰写，包含任务说明、数据集描述、模型结构、训练设置、实验结果和分析。

3. **训练过程记录**

   提供 loss / accuracy 曲线或关键日志截图，说明模型确实完成了训练并收敛。

4. **预测结果展示**

   展示混淆矩阵、特征可视化（如 t-SNE）以及代表性样本的预测结果。

## 8. 过程记录与防作业要求

为了确认作业是本人逐步完成，而不是一次性生成成品代码，本次作业必须满足以下两条过程性要求。

### 9 Git 小步提交

要求每完成一个小模块提交一次 commit，禁止一次性把整个项目 push 上去。

合格 commit 粒度示例：

```text
feat: 加载 EEG 数据集并完成预处理（重参考、滤波、分段等）
feat: 搭建 EEGNet 模型（Block 1 + Block 2 + Classifier）
feat: 实现训练循环和 loss 计算
feat: 实现评估循环和指标计算
feat: 添加混淆矩阵和特征可视化
fix: 修复 DepthwiseConv 维度不匹配问题
docs: 补充实验报告中的 loss/accuracy 曲线
```

提交时一并附上：

```text
仓库地址：GitHub / Gitee 均可
git log --oneline 的文本输出或截图
```

## 10. 参考资源

- EEGNet 论文：Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces", Journal of Neural Engineering, 2018
- EEGNet 官方实现：https://github.com/vlawhern/arl-eegmodels
- Braindecode 库（含 EEGNet 实现）：https://github.com/braindecode/braindecode
- MOABB（脑电基准测试框架）：https://moabb.neurotechx.com/
- BCI Competition IV：https://www.bbci.de/competition/iv/
- MNE-Python（脑电数据处理）：https://mne.tools/
