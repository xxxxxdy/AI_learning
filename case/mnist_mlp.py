"""
MNIST 手写数字识别 —— 两隐藏全连接层 MLP
结构：输入(784) → FC1(256,ReLU) → FC2(128,ReLU) → 输出(10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── 超参数 ────────────────────────────────────────────
BATCH_SIZE    = 256
EPOCHS        = 5
LR            = 0.001
DATA_ROOT     = "./data"
MODEL_PATH    = "./mnist_mlp.pth"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 数据加载 ──────────────────────────────────────────
def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST 均值/标准差
    ])
    train_set = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"训练集: {len(train_set):,}  测试集: {len(test_set):,}")
    return train_loader, test_loader


# ── 网络定义（两隐藏全连接层）────────────────────────
class MnistMLP(nn.Module):
    """
    输入  784（28×28 展平）
      └─ FC1  784 → 256,  ReLU
      └─ FC2  256 → 128,  ReLU
      └─ FC3  128 → 10    （输出 logits）
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # 第一隐藏层
        self.fc2 = nn.Linear(256, 128)   # 第二隐藏层
        self.fc3 = nn.Linear(128,  10)   # 输出层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)             # 展平 28×28 → 784
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)              # 返回原始 logits


# ── 单轮训练 ──────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += model(images).argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


# ── 评估 ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        correct += model(images).argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ── 主流程 ────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  MNIST 手写数字识别 — 双隐层全连接网络")
    print("=" * 50)
    print(f"设备: {DEVICE}\n")

    train_loader, test_loader = get_loaders()

    model     = MnistMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 打印网络结构
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {total_params:,}\n")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        te_acc = evaluate(model, test_loader)
        print(f"Epoch [{epoch}/{EPOCHS}]  "
              f"Loss: {tr_loss:.4f}  "
              f"Train: {tr_acc:.2f}%  "
              f"Test: {te_acc:.2f}%")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → 模型已保存（最佳准确率: {best_acc:.2f}%）")

    print(f"\n训练结束！最佳测试准确率: {best_acc:.2f}%")


# ── 推理演示 ──────────────────────────────────────────
def demo_inference():
    """随机抽取 10 张测试图片，打印真实标签、预测结果和置信度"""
    import numpy as np
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)

    model = MnistMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    indices = np.random.choice(len(test_set), 10, replace=False)
    print("\n随机抽取 10 张测试样本：")
    print(f"{'#':>3}  {'真实':>4}  {'预测':>4}  {'置信度':>8}  {'✓/✗':>4}")
    print("-" * 32)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, label = test_set[idx]
            probs = F.softmax(model(img.unsqueeze(0).to(DEVICE)), dim=1)
            pred  = probs.argmax(1).item()
            conf  = probs.max(1).values.item() * 100
            mark  = "✓" if pred == label else "✗"
            print(f"{i+1:>3}  {label:>4}  {pred:>4}  {conf:>7.2f}%  {mark:>4}")


if __name__ == "__main__":
    main()
    demo_inference()
