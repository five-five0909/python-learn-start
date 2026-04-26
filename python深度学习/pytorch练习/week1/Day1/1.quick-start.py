import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# -------------------- 1. 处理数据 --------------------
# 下载并加载 FashionMNIST 数据集（衣服、鞋包等10类灰度图，28x28）
# root="data"：数据会保存在当前目录的 data 文件夹下
# train=True：加载训练集（60000张）
# train=False：加载测试集（10000张）
# download=True：如果本地没有数据，就自动从网上下载
# transform=ToTensor()：将 PIL 图片或 numpy 数组转为 torch 张量，并把像素值从 [0,255] 缩放到 [0,1]
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# DataLoader 负责将数据集分批次、打乱等，方便训练时按 batch 取数据
# 一批训练64个数据
batch_size = 64
# shuffle=True 会在每个 epoch 开始时打乱训练数据，防止模型记住顺序
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# 测试数据通常不需要打乱
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 查看一个 batch 的数据形状：X 是图像，y 是标签
# X.shape: (batch_size, 通道数, 高, 宽) 这里通道数为1（灰度图），高宽均为28
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # 例如 torch.Size([64, 1, 28, 28])
    print(f"Shape of y: {y.shape} {y.dtype}")     # torch.Size([64]) 64个整数标签
    break


# -------------------- 2. 创建模型 --------------------
# 定义一个简单的全连接神经网络（也叫多层感知机 MLP）
# 这是一个继承的写法 继承了nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Flatten() 将 28x28 的图像拉平成一个长度为 784 的向量
        self.flatten = nn.Flatten()
        # nn.Sequential 按顺序堆叠多个网络层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),   # 全连接层：输入784个像素，输出512个神经元
            nn.ReLU(),               # 非线性激活函数，增加模型表达能力
            nn.Linear(512, 512),     # 第二隐藏层：512入，512出
            nn.ReLU(),
            nn.Linear(512, 10)       # 输出层：512入，10出（对应10个类别）
        )

    def forward(self, x):
        # 定义前向传播过程：输入 x → 拉平 → 经过多层网络 → 输出 logits（未归一化的预测值）
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 如果有 GPU 就用 GPU（cuda），否则用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# 实例化模型并将其移动到对应设备上
model = NeuralNetwork().to(device)


# -------------------- 3. 优化模型参数 --------------------
# 损失函数：交叉熵损失，内部会自动做 softmax，所以模型直接输出 logits 即可
loss_fn = nn.CrossEntropyLoss()
# 优化器：随机梯度下降（SGD），lr=0.001 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 训练函数：遍历一次整个训练集
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 数据集的总样本数
    model.train()  # 将模型设置为训练模式（会启用 dropout、batch norm 的训练行为等）
    # enumerate 迭代的时候 直接输出：图片的一维Tensor数据以及标签
    for batch, (X, y) in enumerate(dataloader):
        # X的维度：(64,1,28,28) 也就是一批次的灰度图片的数据
        # Y的维度：(64,) 也就是一维数组，就是对应这个批次64哥图片的标签值
        X, y = X.to(device), y.to(device)  # 把数据和标签移到 GPU/CPU

        # 前向传播：计算预测值
        pred = model(X)
        # 计算损失
        loss = loss_fn(pred, y)

        # 反向传播：清空上一次的梯度 → 计算新梯度 → 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 100 个 batch 打印一次当前损失和已处理样本数（教程中没打印，这里保留干净版）
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 测试函数：在测试集上评估模型性能
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)        # 测试集总样本数
    num_batches = len(dataloader)         # 总批次数
    model.eval()  # 设置为评估模式（冻结 dropout 等）
    test_loss, correct = 0, 0

    # torch.no_grad() 表示不计算梯度，节省内存和加速
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # 累加损失（乘以 batch 大小以便最终平均）
            test_loss += loss_fn(pred, y).item()
            # pred.argmax(1) 返回每行最大值的索引，即预测类别
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches   # 平均损失
    correct /= size            # 准确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 开始训练和评估多个 epoch
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# -------------------- 4. 保存和加载模型 --------------------
# 保存模型的状态字典（state_dict），里面包含了所有可学习参数（权重、偏置等）
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 重新创建一个同样结构的模型实例，并加载之前保存的参数
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

# 类别名称列表，与 FashionMNIST 的标签顺序对应
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 从测试集中取第一个样本进行预测，看加载后的模型表现
model.eval()
x, y = test_data[0][0], test_data[0][1]   # 第一张图片和它的真实标签
with torch.no_grad():
    x = x.to(device)
    pred = model(x)                         # 预测结果是一个长度为10的向量
    # pred.argmax(1) 找出概率最大的那个类的索引
    predicted_idx = pred.argmax(1).item()   # 转为 Python 数字
    predicted = classes[predicted_idx]
    actual = classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')