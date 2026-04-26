import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

# ============================================================================
# 0. 早停机制类
# ============================================================================
class EarlyStopping:
    """
    早停机制：当验证损失在连续 patience 个 epoch 内没有改善时，停止训练，
    并自动将最佳模型权重保存到本地文件。

    参数:
        patience (int): 允许验证损失不下降的最大 epoch 数。默认 10。
        verbose (bool): 是否打印详情。默认 False。
        path (str): 最佳模型权重的保存路径。默认 'checkpoint.pt'。
    """
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0                # 计数器，记录未改善的 epoch 数
        self.best_loss = float('inf')   # 记录当前最佳损失
        self.early_stop = False         # 触发早停的标志

    def __call__(self, val_loss, model):
        """
        每个 epoch 结束时调用，传入当前验证损失和模型。

        参数:
            val_loss (float): 当前 epoch 的验证损失。
            model (nn.Module): 训练的模型。
        """
        if val_loss < self.best_loss:
            # 损失下降：重置计数器，保存最佳模型
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f'[早停] 验证损失下降至 {val_loss:.6f}，保存模型。')
        else:
            # 损失未下降：计数器 +1
            self.counter += 1
            if self.verbose:
                print(f'[早停] 验证损失未下降，连续 {self.counter}/{self.patience} 次。')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """保存模型权重（state_dict）到文件"""
        torch.save(model.state_dict(), self.path)

# ============================================================================
# 1. 数据准备
# ============================================================================
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# ============================================================================
# 2. 模型定义
# ============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# 设备选择（你之前的代码风格保留了）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ============================================================================
# 3. 损失函数与优化器
# ============================================================================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# ============================================================================
# 4. 训练与测试函数（加入设备移动）
# ============================================================================
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)  # 乘以 batch 大小，方便求平均

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

# ============================================================================
# 5. 集成早停的训练主循环
# ============================================================================
# 初始化早停，耐心值 = 10，并指定模型保存路径
early_stopping = EarlyStopping(patience=10, verbose=True, path='best_model.pt')

epochs = 100  # 设置一个较大的值，早停会自动终止
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    # 训练一个 epoch
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    # 验证一个 epoch
    val_loss = test_loop(test_dataloader, model, loss_fn)

    print(f"训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")

    # 传入验证损失，调用早停检查
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print(f"早停触发！在第 {epoch+1} 个 epoch 停止训练。")
        break

print("训练完成。")

# ============================================================================
# 6. 加载最佳模型进行最终评估（可选）
# ============================================================================
print("\n加载最佳模型权重...")
best_model = NeuralNetwork().to(device)
best_model.load_state_dict(torch.load('best_model.pt'))
best_model.eval()

# 计算测试集准确率
correct = 0
total = 0
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = best_model(X)
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

print(f"最佳模型在测试集上的准确率: {100 * correct / total:.2f}%")