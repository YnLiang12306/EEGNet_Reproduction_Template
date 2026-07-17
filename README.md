# MNIST 手写数字分类 —— 基础 CNN 复现作业

## 1. 作业背景

本次作业是 **脑电信号深度学习** 系列的前置练习，目标是让每位同学**亲手搭一个 CNN、跑通训练、看懂 loss/accuracy 曲线、独立排查问题**。

后续任务将在此基础上引入 EEGNet 等更复杂的模型来解码脑电信号。如果基础 CNN 的数据加载、前向传播、反向传播、评估指标这些环节不熟，直接上手 EEGNet 会非常吃力。因此本次作业聚焦于最经典的 MNIST 手写数字分类任务，结构简单，数据量小。

评价重点是 **流程完整性** 和 **对 CNN 各组件作用的理解**，不以极限 accuracy 作为硬性要求。

## 2. 任务目标

实现一个 CNN 模型，对 MNIST 手写数字图像进行分类。

任务形式：

```text
输入：28×28 灰度手写数字图像
输出：0-9 的类别预测
评估：在 MNIST 测试集上报告分类准确率（accuracy），还可尝试补充其他指标和呈现形式，如召回率、F1分数、精确率、混淆矩阵、AUC曲线、ROC曲线等。
```

核心目标：

1. 理解 MNIST 数据的形状和含义（28×28 单通道）；
2. 掌握 `DataLoader`、`Dataset` 的基本用法；
3. 手动搭建 CNN：卷积层（Conv2d）、池化层（MaxPool2d）、全连接层（Linear），以类的形式实现；
4. 理解激活函数（ReLU）和 Dropout 的作用；
5. 实现训练循环（train loop）和评估循环（eval loop）；
6. 计算并记录 loss 与 accuracy；
7. 使用训练好的模型预测新样本并可视化结果。

## 3. 数据集要求

使用公开数据集：**MNIST**

- 官方数据集主页：http://yann.lecun.com/exdb/mnist/
- PyTorch 可通过 `torchvision.datasets.MNIST` 自动下载。
- 若都不行可尝试国内镜像源，如：https://pypi.tuna.tsinghua.edu.cn/simple/mnist/

数据规模：

| 用途 | 图像数 |
|---|---:|
| 训练集 | 60,000 |
| 测试集 | 10,000 |

要求：

- **必须划分验证集**：从训练集中划分至少 5,000 张作为 validation set，用于监控过拟合；
- 训练和验证阶段不使用测试集，测试集仅在最终评估时使用；
- 数据集文件不要提交到 Git 仓库（加入 `.gitignore`）。

## 4. 模型结构要求

需要以类的形式自行搭建一个 CNN 模型，不能直接调用 `torchvision.models` 中的预定义模型。

### 4.1 最低要求结构

至少包含以下组件：

```text
Input (1×28×28)
  -> Conv2d (in=1, out=?, kernel=3 or 5)
  -> ReLU
  -> MaxPool2d (kernel=2)
  -> Conv2d (out=?, kernel=3 or 5)
  -> ReLU
  -> MaxPool2d (kernel=2)
  -> Flatten
  -> Linear (hidden_dim -> 10)
```

### 4.2 可尝试结构

```text
Input (1×28×28)
  -> Conv2d (1 -> 16, kernel=3, padding=1)
  -> BatchNorm2d (16)
  -> ReLU
  -> MaxPool2d (2)
  -> Conv2d (16 -> 32, kernel=3, padding=1)
  -> BatchNorm2d (32)
  -> ReLU
  -> MaxPool2d (2)
  -> Flatten
  -> Linear (32*7*7 -> 128)
  -> ReLU
  -> Dropout (0.3)
  -> Linear (128 -> 10)
```

### 4.3 可自行调整的部分

- 卷积核大小（3×3 或 5×5）；
- 各层通道数；
- 是否使用 BatchNorm / Dropout；
- 全连接层 hidden dim 大小；
- 激活函数类型（ReLU / LeakyReLU / GELU 等）。

**鼓励尝试不同配置，并在报告中对比效果。**

## 5. 训练要求

### 5.1 基本训练设置

```text
dataset: MNIST
epochs: at least 50
batch size: 32 or 64
optimizer: Adam or SGD
learning rate: 1e-3 for Adam, or 0.01 for SGD
loss: CrossEntropyLoss or other
```

### 5.2 训练循环要求

每个 epoch 需要记录：

- 训练集 average loss
- 训练集 accuracy
- 验证集 average loss
- 验证集 accuracy
- 以及一些其他自行挑选的指标

### 5.3 评估要求

训练结束后在测试集上报告最终 accuracy 以及一些其他自行挑选的指标。

最低结果要求：

- 测试集 accuracy 应 ≥ 95%，若不到需在报告中分析原因；
- 必须输出至少 5 个测试样本的预测可视化（图片 + 真实标签 + 预测标签）。
- 可以尝试加入更多可视化方式，如t-SNE，以后都是要用到的

## 6. 最低完成标准

1. 能够加载 MNIST 数据集，正确划分 train / val / test；
2. 能够自行搭建 CNN 模型（不使用 torchvision 预定义模型）；
3. 能够完成训练循环并打印每个 epoch 的 loss 与 accuracy；
4. 能够在测试集上评估并报告 accuracy；
5. 能够可视化至少 5 个测试样本的预测结果。

## 7. 最终提交内容

1. **实现代码**

   需要包含数据加载、模型定义、训练脚本、评估脚本，代码应有适当注释。

2. **实验报告**

   使用 `report-template.md` 模板撰写，包含任务说明、模型结构、训练设置、实验结果和分析。

3. **训练过程记录**

   提供 loss / accuracy 曲线或关键日志截图，说明模型确实完成了训练并收敛。

4. **预测结果展示**

   展示至少 5 张 MNIST 测试图片，给出真实类别和模型预测类别。

## 8. 过程记录与防作业要求

为了确认作业是本人逐步完成，而不是一次性生成成品代码，本次作业必须满足以下两条过程性要求。


### 9 Git 小步提交

要求每完成一个小模块提交一次 commit，禁止一次性把整个项目 push 上去。

合格 commit 粒度示例：

```text
feat: 加载 MNIST 数据集并划分 train/val/test
feat: 搭建基础 CNN 模型
feat: 实现训练循环和 loss 计算
feat: 实现评估循环和 accuracy 计算
feat: 添加测试集预测和可视化
fix: 修复数据维度不匹配问题
docs: 补充实验报告中的 loss/accuracy 曲线
```

提交时一并附上：

```text
仓库地址：GitHub / Gitee 均可
git log --oneline 的文本输出或截图
```

## 10. 参考资源

- PyTorch 官方 CNN 教程：https://pytorch.org/tutorials/
- MNIST 数据集：http://yann.lecun.com/exdb/mnist/
- CS231n Convolutional Neural Networks：https://cs231n.github.io/convolutional-networks/
