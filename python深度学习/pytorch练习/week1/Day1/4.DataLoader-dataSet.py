import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# ============================================================================
# 1. 加载 FashionMNIST 数据集（PyTorch 内置）
# ============================================================================
# FashionMNIST 包含 60,000 张训练图像和 10,000 张测试图像，
# 每张是 28x28 的灰度图，共 10 个类别（T恤、裤子、套头衫、连衣裙等）。
training_data = datasets.FashionMNIST(
    root="data",                # 数据存放目录（若 local 没有，将自动下载）
    train=True,                 # True 表示加载训练集（60,000 张）
    download=True,              # 如果 root 下没有数据，则从互联网下载
    transform=ToTensor()        # 将 PIL 图像转换为 torch.Tensor，并归一化到 [0, 1]
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,                # False 表示加载测试集（10,000 张）
    download=True,
    transform=ToTensor()
)

# ============================================================================
# 2. 可视化数据集中的样本
# ============================================================================
# 建立类别索引与名称的映射字典
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 创建一个 8x8 英寸的画布，并用 3x3 的子图展示随机样本
# 这是matplotlib的绘图
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # 从训练集中随机抽取一个索引
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]      # 获取图像张量和标签索引

    # 在 3x3 网格中添加子图
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])                # 子图标题显示类别名称
    plt.axis("off")                             # 隐藏坐标轴
    # img.squeeze() 将形状 (1, 28, 28) 压缩为 (28, 28)，以便 imshow 显示
    plt.imshow(img.squeeze(), cmap="gray")      # 灰度显示

plt.show()

# ============================================================================
# 3. 自定义 Dataset 类（用于你自己的数据）
# ============================================================================
# 自定义 Dataset 必须实现三个方法：__init__, __len__, __getitem__。
# 此示例假设图像文件存放在一个目录中，标签通过 CSV 文件给出。
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        annotations_file: CSV 文件路径，格式为 [文件名, 标签]
        img_dir: 图像文件所在的目录
        transform: 对图像进行的预处理/增强（例如 ToTensor, Normalize）
        target_transform: 对标签进行的转换（例如 one-hot 编码）
        """
        self.img_labels = pd.read_csv(annotations_file)    # 读取 CSV
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回 (image, label) 元组。
        - idx 是 Dataset 对外的索引。
        - 内部从磁盘加载图像，并查找对应标签。
        - 若设置了 transform，则先对图像和标签做变换再返回。
        """
        # 组合图像文件的完整路径
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # 使用 torchvision.io.read_image 读取图像（返回 Tensor）
        image = read_image(img_path)
        # 获取标签（第 1 列）
        label = self.img_labels.iloc[idx, 1]

        # 应用变换（如果提供了）
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# 注意：上面的 CustomImageDataset 仅为演示，不会直接运行，因为需要实际文件。

# ============================================================================
# 4. 使用 DataLoader 将 Dataset 封装成可迭代的批次
# ============================================================================
# DataLoader 负责将 Dataset 按 batch 组织，并提供：
#   - 自动分批（batch_size）
#   - 随机打乱（shuffle）
#   - 多进程加速（num_workers）
batch_size = 64

# 训练集 DataLoader：每个 epoch 开始时打乱数据
train_dataloader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True          # 训练时打乱，防止模型记忆数据顺序
)

# 测试集 DataLoader：一般不需要打乱，只需按批次加载
test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

# ============================================================================
# 5. 迭代 DataLoader 并观察输出形状
# ============================================================================
# 迭代一次 train_dataloader，查看一个批次的数据
# 由于设置了 shuffle=True，每次迭代完所有批次后，数据会被重新洗牌
train_features, train_labels = next(iter(train_dataloader))
print(f"特征批次形状 (batch_size, channels, height, width): {train_features.shape}")
print(f"标签批次形状 (batch_size): {train_labels.shape}")

# 显示第一个样本的真实标签
print(f"第一个样本的标签: {labels_map[train_labels[0].item()]}")

# 如果需要可视化这个批次中的某个图像：
# plt.imshow(train_features[0].squeeze(), cmap="gray")
# plt.title(labels_map[train_labels[0].item()])
# plt.show()