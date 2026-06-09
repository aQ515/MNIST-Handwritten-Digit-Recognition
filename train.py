"""MNIST手写数字识别 - 训练脚本"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MNISTClassifier

# 超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")


def get_data_loaders():
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集: {len(train_dataset)}张, 测试集: {len(test_dataset)}张")
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  batch {batch_idx+1}/{len(train_loader)}, loss={loss.item():.4f}")

    return total_loss / len(train_loader), 100. * correct / len(train_loader.dataset)


def evaluate(model, test_loader):
    """测试集评估"""
    model.eval()
    total_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()

    return total_loss / len(test_loader), 100. * correct / len(test_loader.dataset)


def plot_curves(history):
    """画loss和accuracy曲线"""
    x = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, history["train_loss"], "b-", label="train", linewidth=1.5)
    axes[0].plot(x, history["test_loss"], "r-", label="test", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, history["train_acc"], "b-", label="train", linewidth=1.5)
    axes[1].plot(x, history["test_acc"], "r-", label="test", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 标注最佳结果
    best_acc = max(history["test_acc"])
    best_epoch = history["test_acc"].index(best_acc) + 1
    axes[1].annotate(f"Best: {best_acc:.2f}%",
                     xy=(best_epoch, best_acc),
                     xytext=(best_epoch + 2, best_acc - 0.5),
                     arrowprops=dict(arrowstyle="->", color="green"),
                     fontsize=11, color="green")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("曲线已保存")


def main():
    print(f"超参数: batch={BATCH_SIZE}, lr={LEARNING_RATE}, epochs={EPOCHS}")

    # 加载数据
    train_loader, test_loader = get_data_loaders()

    # 初始化模型
    model = MNISTClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_acc = 0.0

    print("\n开始训练...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"  train loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  test  loss={test_loss:.4f}, acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "mnist_model.pth")
            print(f"  -> 保存模型 (acc={best_acc:.2f}%)")

    print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")
    plot_curves(history)


if __name__ == "__main__":
    main()
