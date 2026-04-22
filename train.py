"""MNIST 手写数字识别 - 训练脚本（优化版本）"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MNISTClassifier

BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Epochs: {EPOCHS}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {len(train_dataset)} 张, 测试集: {len(test_dataset)} 张")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")
    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return test_loss / len(loader), 100. * correct / total


model     = MNISTClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

history  = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss,  test_acc  = evaluate(model, test_loader)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "mnist_model.pth")
        print(f">>> 保存最佳模型! 测试准确率: {best_acc:.2f}%")

print(f"\n{'='*50}")
print(f"最佳测试准确率: {best_acc:.2f}%")
print(f"{'='*50}")

# 可视化训练曲线
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
axes[0].plot(epochs_range, history["test_loss"],  "r-", label="Test Loss",  linewidth=1.5)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Test Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], "b-", label="Train Accuracy", linewidth=1.5)
axes[1].plot(epochs_range, history["test_acc"],  "r-", label="Test Accuracy",  linewidth=1.5)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Training & Test Accuracy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

best_epoch_val = history["test_acc"].index(best_acc) + 1
axes[1].annotate(f"Best: {best_acc:.2f}% (Epoch {best_epoch_val})",
                 xy=(best_epoch_val, best_acc),
                 xytext=(best_epoch_val + 2, best_acc - 0.5),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=11, color="green", fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("训练曲线已保存至 training_curves.png")
