# MNIST 手写数字识别

用 PyTorch 实现的全连接神经网络，在 MNIST 数据集上做手写数字分类。

测试准确率: 98.4%

## 运行

```bash
pip install torch torchvision matplotlib pillow

python train.py        # 训练
python inference.py    # 推理
```

## 踩坑

- 一开始只用了3层网络，准确率卡在92%左右，加到4层后明显提升
- 最开始 Dropout 设了 0.3，结果训练收敛太慢，降到 0.2 好很多
- 学习率设 0.01 时 loss 来回跳，调成 0.001 就稳定了
- 如果把 torch.save 放在 model.eval() 之后会有点问题，要注意顺序
