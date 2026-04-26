import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================================
# 1. 获取训练设备
# ============================================================================
# 检测可用的加速器（CUDA、MPS、XPU 等），优先使用 GPU，否则回退到 CPU。
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

# ============================================================================
# 2. 定义神经网络类
# ============================================================================
# 通过继承 nn.Module 来构建神经网络。
# 在 __init__ 中定义网络层，在 forward 中定义数据的前向传播流程。
class NeuralNetwork(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的初始化方法（必须执行）
        super().__init__()
        # Flatten 层：将输入图像 (batch, 28, 28) 展平为 (batch, 784)
        self.flatten = nn.Flatten()
        # 使用 Sequential 容器顺序堆叠多个层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),   # 全连接层：784 个输入 → 512 个输出
            nn.ReLU(),               # ReLU 激活函数：引入非线性
            nn.Linear(512, 512),     # 第二隐藏层：512 → 512
            nn.ReLU(),
            nn.Linear(512, 10),      # 输出层：512 → 10（对应 10 个类别）
        )

    def forward(self, x):
        # 1. 将图像张量拉平
        x = self.flatten(x)
        # 2. 通过线性层 + ReLU 堆栈，得到原始的类别预测分数 (logits)
        logits = self.linear_relu_stack(x)
        return logits

# 实例化模型并将其移动到设备上
model = NeuralNetwork().to(device)
print(model)

# ============================================================================
# 3. 使用模型进行预测
# ============================================================================
# 创建一个模拟的小批量数据：3 张 28x28 的随机图像
X = torch.rand(3, 28, 28, device=device)     # 形状 (3, 28, 28)
logits = model(X)                              # 调用模型，得到 (3, 10) 的 logits

# 将 logits 传递给 Softmax 层，获得预测概率 (0~1，且每行求和为 1)
pred_probab = nn.Softmax(dim=1)(logits)
# 取每行概率最大的索引作为预测类别
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# ============================================================================
# 4. 分解模型层：观察数据在每一层之间的形状变化
# ============================================================================
print(f"\n--- 模型层分解 ---")
print(f"原始输入图像形状: {X.shape}")           # torch.Size([3, 28, 28])

# 4.1 nn.Flatten
# 将 28x28 图像展平为 784 个连续像素值（保持 batch 维度不变）
flatten = nn.Flatten()
flat_img = flatten(X)
print(f"经过 Flatten 后: {flat_img.shape}")   # torch.Size([3, 784])

# 4.2 nn.Linear
# 线性变换：out = in @ weight.T + bias
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_img)
print(f"经过第 1 个 Linear(784→20) 后: {hidden1.shape}")  # torch.Size([3, 20])

# 4.3 nn.ReLU
# 非线性激活函数：将负数置零，正数保持不变
print(f"Before ReLU: {hidden1}\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}\n")

# 4.4 nn.Sequential
# 将多个层串联在一起，数据按顺序依次通过每一层
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
)
# 模拟数据
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(f"经过 Sequential 后的输出形状: {logits.shape}")  # torch.Size([3, 10])

# 4.5 nn.Softmax
# 将 logits 转换为概率分布（dim=1 表示在类别维度上做归一化）
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"Softmax 概率形状: {pred_probab.shape}")        # torch.Size([3, 10])

# ============================================================================
# 5. 查看模型参数
# ============================================================================
print(f"\n--- 模型参数 ---")
print(f"模型结构:\n{model}\n")

# 遍历每一层的名称及其参数
for name, param in model.named_parameters():
    print(f"层: {name} | 大小: {param.size()} | 数值样例: {param[:2]}")