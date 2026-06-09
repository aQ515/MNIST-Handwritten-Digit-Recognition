"""MNIST手写数字识别模型"""

import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    """4层全连接网络: 784->256->128->64->10"""

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 把图片展平成一维向量
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    model = MNISTClassifier()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")
