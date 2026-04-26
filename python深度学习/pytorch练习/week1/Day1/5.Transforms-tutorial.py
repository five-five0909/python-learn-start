import torch
from torchvision import transforms, datasets

# ============================================================================
# 日常高频使用的图像转换
# ============================================================================
# Compose 用于将多个变换按顺序串联起来
train_transforms = transforms.Compose([
    # ---------- 空间变换 ----------
    transforms.Resize(256),                     # 1. 将短边缩放到256，长边按比例缩放
    transforms.RandomResizedCrop(224),          # 2. 随机裁剪并缩放至224x224（训练时常用）
    transforms.RandomHorizontalFlip(p=0.5),     # 3. 以50%概率水平翻转（左右镜像）
    transforms.RandomVerticalFlip(p=0.1),       # 4. 以10%概率垂直翻转（视任务适用性）
    transforms.RandomRotation(degrees=15),      # 5. 在[-15°, +15°]内随机旋转

    # ---------- 颜色 / 像素变换 ----------
    transforms.ColorJitter(
        brightness=0.2,   # 亮度在 [0.8, 1.2] 范围内随机变化
        contrast=0.2,     # 对比度
        saturation=0.2,   # 饱和度
        hue=0.1           # 色相 (-0.1~0.1)
    ),
    transforms.GaussianBlur(kernel_size=3),     # 6. 使用3x3高斯核随机模糊（可选）

    # ---------- 转为Tensor ----------
    transforms.ToTensor(),                      # 7. 转为张量，并自动将像素值缩放到 [0,1]

    # ---------- 归一化 ----------
    # 使用 ImageNet 的均值和标准差（适用于 RGB 三通道图像）
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],             # 8. 对各通道减均值
        std=[0.229, 0.224, 0.225]               # 9. 除以标准差，使分布接近N(0,1)
    ),
])

# 测试时通常只需固定的大小和归一化，不能做随机增强
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),                 # 从中心裁剪224x224，而非随机
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================================
# 应用到数据集（以 FashionMNIST 为例，但它只有1通道，需调整）
# ============================================================================
# 若你的数据是灰度图，应修改 Normalize 参数（单通道均值/标准差）
# 这里使用 FashionMNIST 的统计值
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(224),                  # 缩放至224可适应预训练模型
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])  # FashionMNIST 单通道
    ])
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ])
)