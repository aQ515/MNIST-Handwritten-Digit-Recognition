"""MNIST 手写数字识别 - 推理脚本
使用方法:
    python inference.py <图片路径>
    python inference.py              # 交互模式
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from model import MNISTClassifier

MODEL_PATH = "mnist_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 '{MODEL_PATH}'")
        print("请先运行 train.py 训练模型")
        sys.exit(1)
    model = MNISTClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)


def predict(image_tensor, model):
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0][pred_class].item() * 100
    return pred_class, confidence


def main():
    model = load_model()
    print(f"使用设备: {DEVICE}")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("请输入图片路径: ").strip().strip('"').strip("'")

    if not image_path:
        print("未输入图片路径，退出。")
        return

    try:
        image_tensor = preprocess_image(image_path)
        print(f"图片已加载: {image_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    digit, confidence = predict(image_tensor, model)

    print(f"\n{'='*30}")
    print(f"  预测结果: {digit}")
    print(f"  置信度:   {confidence:.2f}%")
    print(f"{'='*30}")


if __name__ == "__main__":
    main()
