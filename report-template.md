# MNIST 手写数字分类实验报告

## 1. 任务说明

本实验使用 CNN（卷积神经网络）对 MNIST 手写数字进行分类。

```text
输入：28×28 单通道灰度手写数字图像
输出：0-9 类别预测
评估指标：测试集 classification accuracy
```

## 2. 数据集

- 数据集名称：MNIST
- 数据集地址：http://yann.lecun.com/exdb/mnist/
- 训练集图像数：60,000
- 验证集图像数：（从训练集中划分的数量）
- 测试集图像数：10,000
- 图像尺寸：28×28 单通道
- 类别数：10
- 使用设备：CPU / GPU
- 总训练耗时：

## 3. 数据预处理

请说明数据预处理方式：

| 预处理 | 参数设置 |
|---|---|
| 归一化（Normalize） | mean= , std= |
| 数据增强（如有） |  |
| 验证集划分比例 |  |

可自行补充其他，并请说明为什么选择这些预处理方式：

```text
（在这里填写）
```

## 4. 模型结构

请画出你的 CNN 结构：

```text
Input (1×28×28)
  -> Conv2d (in= , out= , kernel= , stride= , padding= )
  -> BatchNorm2d ( ) / 不使用
  -> ReLU / LeakyReLU / ...
  -> MaxPool2d (kernel= , stride= )
  -> Conv2d (in= , out= , kernel= , stride= , padding= )
  -> BatchNorm2d ( ) / 不使用
  -> ReLU / LeakyReLU / ...
  -> MaxPool2d (kernel= , stride= )
  -> Flatten
  -> Linear (in= , out= )
  -> ReLU / ...
  -> Dropout (p= ) / 不使用
  -> Linear (in= , out= )
```

模型总参数量：

请说明各组件的作用：

| 组件 | 作用 |
|---|---|
| Conv2d |  |
| BatchNorm2d |  |
| ReLU |  |
| MaxPool2d |  |
| Dropout |  |
| Linear |  |

## 5. 训练设置

| 配置 | 数值 |
|---|---:|
| epochs |  |
| batch size |  |
| optimizer | Adam / SGD |
| learning rate |  |
| loss function | CrossEntropyLoss |
| device | CPU / GPU |

## 6. 训练过程

### 6.1 Loss 与 Accuracy 记录

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---:|---:|---:|---:|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| ... |  |  |  |  |

请粘贴 loss 与 accuracy 曲线图。

请简要描述训练过程是否正常（loss 是否下降、accuracy 是否上升、是否有过拟合迹象）：

```text
（在这里填写）
```

## 7. 测试结果

| 指标 | 结果 |
|---|---:|
| test accuracy |  |
| test loss |  |
| random baseline（10 类） | 10% |

可自行补充其他指标，并请分析结果是否达到预期：

```text
（在这里填写）
```

## 8. 预测结果展示

至少展示 5 个测试样本的预测结果。

| 样本编号 | 真实标签 | 预测标签 | 是否正确 |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |

请粘贴预测可视化的图片。

## 9. 超参数对比实验

尝试调整至少一项超参数，并记录结果对比：

| 实验 | 超参数变更 | Test Accuracy |
|---|---|---|
| baseline | （默认配置） |  |
| 实验 A |  |  |
| 实验 B |  |  |

请简要分析哪个超参数对结果影响最大：

```text
（在这里填写）
```

## 10. 问题与改进

请简要说明：

- 遇到了哪些问题（维度不匹配、loss 不下降、过拟合等）；
- 如何定位并解决这些问题；
- 如果继续改进，可以从哪些方面入手，例如加深模型、调整学习率策略、使用数据增强、尝试更复杂的结构等。

```text
（在这里填写）
```

## 11. Git 提交记录

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```
