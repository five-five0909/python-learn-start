# ============================================================================
# config.py — 全局配置模块
# ============================================================================
#
# 职责：把所有超参数和路径集中管理，其他模块从这里导入
#
# 为什么要单独一个配置文件？
#   1. 集中管理：所有超参数在一个地方修改，不用到处找
#   2. 避免硬编码：其他模块不直接写数字，而是导入常量
#   3. 方便实验：想改学习率？只改这一个文件就行
#
# 包含内容：
#   - 超参数：学习率、batch 大小、epoch 数等
#   - 路径配置：数据文件、模型文件、输出文件的位置
#   - 设备选择：自动检测 GPU/CPU
#   - 随机种子：固定所有随机数源，保证实验可复现
#
# ============================================================================

import os
# ↑ 操作系统相关功能：创建文件夹、拼接文件路径
#   例：os.path.join("data", "train.csv") → "data/train.csv"

import random
# ↑ Python 内置随机数模块
#   用于固定随机种子，保证实验可复现

import numpy as np
# ↑ NumPy：科学计算库，处理多维数组
#   这里用于 np.random.seed() 固定 NumPy 的随机种子

import torch
# ↑ PyTorch：深度学习框架
#   这里用于 torch.manual_seed() 和设备检测


# ============================================================================
# 超参数
# ============================================================================
# 超参数 = 人为设定的参数（不是模型从数据中学到的）
# 这些值需要根据实验效果手动调整

SEED = 42
# ↑ 随机种子，固定这个数字后，每次运行的随机结果都一样
#   42 是经典选择（《银河系漫游指南》里的"生命、宇宙以及一切的答案"）
#   影响范围：权重初始化、数据打乱顺序、Dropout、数据集划分等

DATA_DIR = "data/house-prices-advanced-regression-techniques"
# ↑ 数据文件夹路径（相对于项目根目录）

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
# ↑ 训练集文件路径
#   os.path.join 会自动用正确的分隔符拼接路径
#   Windows：data\house-prices...\train.csv
#   Linux：  data/house-prices.../train.csv

TEST_PATH = os.path.join(DATA_DIR, "test.csv")
# ↑ 测试集文件路径

OUT_DIR = os.path.join(DATA_DIR, "out")
# ↑ 输出文件夹路径（存放模型和提交文件）

MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")
# ↑ 最优模型保存路径
#   .pt 是 PyTorch 模型文件的标准后缀

SUBMISSION_PATH = os.path.join(OUT_DIR, "submission.csv")
# ↑ Kaggle 提交文件路径

BATCH_SIZE = 32
# ↑ 每批次送入模型的样本数
#   32 是较小的 batch，适合小数据集（1460 条）
#   batch 越小 → 梯度噪声越大 → 正则化效果越好
#   batch 越大 → 梯度越稳定 → 训练越快

EPOCHS = 500
# ↑ 最大训练轮数
#   1 个 epoch = 把全部训练数据看一遍
#   设大一点没关系，有早停机制保底

PATIENCE = 20
# ↑ 早停耐心值：连续 20 个 epoch 验证损失没改善就停
#   太小 → 可能还没学够就停了
#   太大 → 浪费时间，过拟合了还在训练

LEARNING_RATE = 1e-3
# ↑ 学习率：每次参数更新的步长
#   1e-3 = 0.001，Adam 优化器的经典默认值
#   太大 → 训练不稳定，loss 震荡
#   太小 → 训练太慢，可能卡在局部最优

VAL_RATIO = 0.2
# ↑ 验证集比例：从训练集中拿出 20% 作为验证集
#   训练集 1460 条 × 0.2 = 292 条验证集，1168 条训练集


# ============================================================================
# 设备选择
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
# ↑ 自动选择计算设备
#   torch.cuda.is_available() → 检查有没有可用的 GPU
#   有 GPU → device = "cuda"（用显卡计算，快 10~100 倍）
#   没 GPU → device = "cpu"（用 CPU 计算，慢但能跑）
#
#   后续代码中 .to(device) 就是把数据/模型搬到对应设备上


# ============================================================================
# 创建输出文件夹
# ============================================================================

os.makedirs(OUT_DIR, exist_ok=True)
# ↑ 创建输出文件夹（如果不存在的话）
#   exist_ok=True → 文件夹已存在也不报错


# ============================================================================
# 固定随机种子：让每次运行结果一模一样
# ============================================================================
#
# 为什么要固定随机种子？
#   神经网络的权重初始化、数据打乱顺序、Dropout 等都涉及随机数
#   不固定种子 → 每次运行结果不同 → 无法复现 → 无法对比实验
#
# ============================================================================

def set_seed(seed=SEED):
    """固定所有随机数源，保证实验可复现。

    Args:
        seed: 随机种子数字，默认使用全局 SEED

    固定的随机数源：
        1. Python 内置 random 模块
        2. NumPy 的随机数生成器
        3. PyTorch CPU 端的随机数生成器
        4. PyTorch GPU 端的随机数生成器（如果有的话）
        5. cuDNN 的确定性算法和自动调优（如果有的话）
    """

    random.seed(seed)
    # ↑ 固定 Python 内置 random 模块的种子
    #   影响：random.shuffle()、random.choice() 等

    np.random.seed(seed)
    # ↑ 固定 NumPy 的随机种子
    #   影响：np.random.randn()、np.random.permutation() 等

    torch.manual_seed(seed)
    # ↑ 固定 PyTorch CPU 端的随机种子
    #   影响：权重初始化、random_split 等

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # ↑ 固定 PyTorch GPU 端的随机种子（所有 GPU 都固定）
        #   manual_seed_all → 对所有可用 GPU 生效
        #   manual_seed     → 只对当前 GPU 生效

        torch.backends.cudnn.deterministic = True
        # ↑ 强制 cuDNN 使用确定性算法
        #   cuDNN 是 NVIDIA 的 GPU 加速库
        #   默认情况下 cuDNN 会自动选最快的算法，但不同次运行可能选不同算法
        #   设为 True → 每次选同样的算法 → 结果可复现

        torch.backends.cudnn.benchmark = False
        # ↑ 关闭 cuDNN 的自动调优
        #   benchmark=True 时，cuDNN 会在第一次运行时测试多种算法，选最快的
        #   但这个"测试"过程本身有随机性
        #   设为 False → 关闭自动调优 → 保证确定性


# 包被导入时自动执行 set_seed()，确保后续所有操作都在固定种子下运行
set_seed()
