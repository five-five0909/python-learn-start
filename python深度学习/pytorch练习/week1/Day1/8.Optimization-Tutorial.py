import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ============================================================================
# 1. 数据准备（准备训练集、测试集和数据加载器）
# ============================================================================
# FashionMNIST 包含 60,000 张训练图像和 10,000 张测试图像，均为 28x28 灰度图，共 10 个类别。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()          # 将图像转换为张量，并归一化到 [0, 1]
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# DataLoader 负责将数据集封装为可迭代的批次对象
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# ============================================================================
# 2. 模型定义
# ============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()   # 将 28x28 的图像展平为 784 维向量
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),    # 输入 784 维 -> 隐藏层 512 维
            nn.ReLU(),                # ReLU 非线性激活
            nn.Linear(512, 512),      # 第二隐藏层 512 -> 512
            nn.ReLU(),
            nn.Linear(512, 10),       # 输出层：512 -> 10（对应 10 个类别）
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 初始化模型
model = NeuralNetwork()

# ============================================================================
# 3. 超参数、损失函数和优化器定义
# ============================================================================
learning_rate = 1e-3                # 学习率
loss_fn = nn.CrossEntropyLoss()     # 交叉熵损失函数（内部已包含 Softmax）
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器

# ============================================================================
# 4. 训练循环（逐批次优化参数）
# ============================================================================
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 数据集总样本数
    model.train()                   # 将模型设置为训练模式（影响 BatchNorm、Dropout 等层）

    for batch, (X, y) in enumerate(dataloader):
        # --- 前向传播 ---
        pred = model(X)             # 计算模型输出（logits）
        loss = loss_fn(pred, y)     # 计算损失

        # --- 反向传播与参数更新 ---
        optimizer.zero_grad()       # 1. 梯度清零（默认梯度会累加，因此必须手动清零）
        loss.backward()             # 2. 反向传播，计算每个参数的梯度
        optimizer.step()            # 3. 根据梯度更新参数

        # 每 100 个批次打印一次损失和进度
        if batch % 100 == 0:
            loss = loss.item()
            current = min((batch + 1) * len(X), size)  # 防止超过总样本数（可选）
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
# ============================================================================
# 5. 测试循环（评估模型在测试集上的表现）
# ============================================================================
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)      # 测试集样本数
    num_batches = len(dataloader)
    model.eval()                        # 将模型设置为评估模式（冻结 Dropout 等）

    test_loss, correct = 0, 0

    with torch.no_grad():               # 禁用梯度计算，减少内存占用，加速推理
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()                     # 累积损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 累积正确预测数

    test_loss /= num_batches            # 平均损失
    correct /= size                     # 准确率

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# ============================================================================
# 6. 执行训练（迭代多个 epoch）
# ============================================================================
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")