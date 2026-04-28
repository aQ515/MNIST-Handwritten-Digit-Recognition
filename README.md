# MNIST 手写数字识别

基于 PyTorch 的全连接神经网络，在 MNIST 数据集上实现 **98.4%** 测试准确率。

## 项目结构

```
├── model.py              # 神经网络模型定义
├── train.py              # 训练脚本（含 Loss/Accuracy 可视化）
├── inference.py          # 推理脚本（支持本地图片输入）
├── mnist_model.pth       # 训练好的模型权重
├── training_curves.png   # 训练过程可视化图表
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install torch torchvision matplotlib pillow numpy

# 训练模型
python train.py

# 推理测试
python inference.py your_image.png
```

## 模型结构

| 层   | 输入维度 | 输出维度 | 激活函数 |
|------|---------|---------|---------|
| fc1  | 784     | 256     | ReLU    |
| fc2  | 256     | 128     | ReLU    |
| fc3  | 128     | 64      | ReLU    |
| fc4  | 64      | 10      | —       |

- **Dropout**: 0.2（防止过拟合）
- **优化器**: Adam (lr=0.001)
- **损失函数**: CrossEntropyLoss
- **参数量**: ~242,762

## 性能

| 指标           | 数值    |
|---------------|--------|
| 测试集准确率    | 98.4%  |
| 训练轮数 (Epochs) | 20  |
| Batch Size     | 64     |

## 优化过程

1. 初始3层网络 (128→64→10)，测试准确率约 92%
2. 扩展为4层 (256→128→64→10)，加入 Dropout(0.3)
3. 降低学习率至 0.001，Dropout 降至 0.2，训练 20 轮 → 98.4%

## 技术栈

Python · PyTorch · torchvision · Matplotlib · Pillow

---

广东药科大学 计算机科学与技术 课程设计项目
