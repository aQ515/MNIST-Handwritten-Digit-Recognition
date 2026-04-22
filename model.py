"""MNIST 手写数字识别 - 神经网络模型定义"""

import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    """
    全连接神经网络分类器
    网络结构: Input(784) -> FC1(256) -> FC2(128) -> FC3(64) -> Output(10)
    """

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
