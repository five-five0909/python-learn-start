# ============================================================================
# 【项目说明】
# Kaggle House Prices Advanced Regression Techniques 竞赛
# 目标：预测房屋售价 SalePrice
# 评分指标：log RMSE（对数空间的均方根误差）
#
# 本代码特点：
# 1. 使用自定义 Dataset 类封装数据处理逻辑（代码更模块化）
# 2. 完整的早停机制 + 学习率调度器
# 3. 固定随机种子保证实验可复现
# 4. 丰富的评估指标：logRMSE、RMSE、MAE、R²
# ============================================================================

# ============================================================================
# 第一步：引入依赖库
# ============================================================================

import os
# ↑ 操作系统相关功能：创建文件夹、拼接文件路径
#   例：os.path.join("data", "train.csv") → "data/train.csv"

import random
# ↑ Python 内置随机数模块
#   用于固定随机种子，保证实验可复现

import numpy as np
# ↑ NumPy：科学计算库，处理多维数组
#   例：np.log1p()、np.sqrt()、np.expm1() 等数学函数
#   为什么叫 np？约定俗成的缩写，少打字

import pandas as pd
# ↑ Pandas：数据分析库，处理表格数据（类似 Excel）
#   核心数据结构：DataFrame（表格）和 Series（一列）
#   例：pd.read_csv() 读取 CSV 文件为 DataFrame

import torch
# ↑ PyTorch：深度学习框架
#   提供张量（Tensor）、自动求导（Autograd）、神经网络模块等

from torch import nn
# ↑ nn 是 PyTorch 的神经网络模块
#   包含：nn.Linear（全连接层）、nn.ReLU（激活函数）、nn.MSELoss（损失函数）等
#   from torch import nn = 从 torch 包中导入 nn 子模块

from torch.utils.data import Dataset, DataLoader, random_split
# ↑ PyTorch 的数据加载工具
#   Dataset     → 自定义数据集的基类（你要继承它来写自己的数据集）
#   DataLoader  → 自动把数据分成小批次（batch），还能打乱顺序
#   random_split → 按比例随机划分数据集（如 80% 训练，20% 验证）

from sklearn.preprocessing import StandardScaler
# ↑ scikit-learn 的标准化工具
#   StandardScaler：把数据变成均值=0、标准差=1 的分布
#   公式：(x - 均值) / 标准差


# ============================================================================
# 【全局配置区】
# ============================================================================
# ↑ 把所有超参数集中放在最前面，方便修改和管理
#   超参数 = 人为设定的参数（不是模型学到的）

SEED            = 42
# ↑ 随机种子，固定这个数字后，每次运行的随机结果都一样
#   42 是经典选择（《银河系漫游指南》里的"生命、宇宙以及一切的答案"）

DATA_DIR        = "data/house-prices-advanced-regression-techniques"
# ↑ 数据文件夹路径

TRAIN_PATH      = os.path.join(DATA_DIR, "train.csv")
# ↑ 训练集文件路径
#   os.path.join 会自动用正确的分隔符拼接路径
#   Windows：data\house-prices...\train.csv
#   Linux：  data/house-prices.../train.csv

TEST_PATH       = os.path.join(DATA_DIR, "test.csv")
# ↑ 测试集文件路径

OUT_DIR         = os.path.join(DATA_DIR, "out")
# ↑ 输出文件夹路径（存放模型和提交文件）

MODEL_PATH      = os.path.join(OUT_DIR, "best_model.pt")
# ↑ 最优模型保存路径
#   .pt 是 PyTorch 模型文件的标准后缀

SUBMISSION_PATH = os.path.join(OUT_DIR, "submission.csv")
# ↑ Kaggle 提交文件路径

BATCH_SIZE      = 64
# ↑ 每批次送入模型的样本数
#   每次取 64 条数据一起训练，而不是 1 条 1 条来
#   太小（如 1）→ 训练慢，梯度噪声大
#   太大（如 1024）→ 内存不够，梯度太"稳"可能学不好
#   64 是常用默认值

EPOCHS          = 500
# ↑ 最大训练轮数
#   1 个 epoch = 把全部训练数据看一遍
#   500 个 epoch = 看 500 遍（配合早停机制，不用真的跑满 500 轮）

PATIENCE        = 40
# ↑ 早停耐心值：连续 40 个 epoch 验证损失没改善就停
#   防止模型过度训练（过拟合）

LEARNING_RATE   = 1e-3
# ↑ 学习率：每次参数更新的步长
#   1e-3 = 0.001
#   太大 → 模型震荡，loss 不收敛
#   太小 → 训练太慢，卡在局部最优
#   0.001 是 Adam 优化器的经典默认值

WEIGHT_DECAY    = 1e-4
# ↑ 权重衰减（L2 正则化系数）
#   1e-4 = 0.0001
#   作用：惩罚过大的权重，防止过拟合
#   原理：每次更新时，把权重乘以 (1 - 0.0001)，让它慢慢变小

VAL_RATIO       = 0.2
# ↑ 验证集比例：从训练集中拿出 20% 作为验证集
#   训练集 1460 条 × 0.2 = 292 条验证集，1168 条训练集

device = "cuda" if torch.cuda.is_available() else "cpu"
# ↑ 自动选择计算设备
#   torch.cuda.is_available() → 检查有没有可用的 GPU
#   有 GPU → device = "cuda"（用显卡计算，快 10~100 倍）
#   没 GPU → device = "cpu"（用 CPU 计算，慢但能跑）
#
#   后续代码中 .to(device) 就是把数据/模型搬到对应设备上

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
# 分级策略：
#   日常开发 / 快速实验 → 只用基础三件套 + GPU 种子（够用了）
#   调参 / 做对比实验   → 基础三件套 + GPU + cuDNN（保证公平对比）
#   发论文 / 提交竞赛   → 全部固定（别人必须能复现你的结果）
#
# ============================================================================

random.seed(SEED)
# ↑ 固定 Python 内置 random 模块的种子
#   影响：random.shuffle()、random.choice() 等

np.random.seed(SEED)
# ↑ 固定 NumPy 的随机种子
#   影响：np.random.randn()、np.random.permutation() 等

torch.manual_seed(SEED)
# ↑ 固定 PyTorch CPU 端的随机种子
#   影响：权重初始化、random_split 等

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # ↑ 固定 PyTorch GPU 端的随机种子（所有 GPU 都固定）
    #   manual_seed_all → 对所有可用 GPU 生效
    #   manual_seed     → 只对当前 GPU 生效

    torch.backends.cudnn.deterministic = True
    # ↑ 强制 cuDNN 使用确定性算法
    #   cuDNN 是 NVIDIA 的 GPU 加速库
    #   默认情况下 cuDNN 会自动选最快的算法，但不同次运行可能选不同算法
    #   设为 True → 每次选同样的算法 → 结果可复现
    #   代价：可能略微变慢（因为不能选最快的算法了）

    torch.backends.cudnn.benchmark = False
    # ↑ 关闭 cuDNN 的自动调优
    #   benchmark=True 时，cuDNN 会在第一次运行时测试多种算法，选最快的
    #   但这个"测试"过程本身有随机性
    #   设为 False → 关闭自动调优 → 保证确定性


# ============================================================================
# 【BestCheckpoint 类】最佳模型保存器
# ============================================================================
#
# 这个类解决一个核心问题：训练 500 轮，哪一轮的模型最好？
#
# 训练过程中，模型不是越训练越好：
#   前 100 轮：模型在学习，验证集 loss 逐渐下降（欠拟合阶段）
#   100~200 轮：验证集 loss 降到最低（最佳状态）
#   200~500 轮：验证集 loss 开始上升（过拟合了，模型"背答案"而不是"学规律"）
#
# BestCheckpoint 的职责：
#   每轮训练结束后，检查当前模型是否是"历史最佳"
#   如果是 → 保存到文件（覆盖之前的）
#   如果不是 → 跳过
#   训练结束后 → 加载那个最好的模型
#
# 与 EarlyStopping 的分工：
#   BestCheckpoint → 管"哪个最好"和"怎么保存/加载"
#   EarlyStopping  → 管"什么时候该停"
#
# checkpoint 文件里保存了什么？
#   - model_state_dict:      模型的所有权重参数（一个大字典）
#   - optimizer_state_dict:  优化器的状态（学习率、动量等内部变量）
#   - scheduler_state_dict:  学习率调度器的状态
#   - epoch:                 最佳 epoch 编号
#   - best_loss:             最佳验证损失
#   - best_metrics:          最佳 epoch 的完整指标（logRMSE、RMSE、MAE、R²）
#
# ============================================================================

class BestCheckpoint:
    """
    最佳模型 Checkpoint 管理器

    用法：
        best_ckpt = BestCheckpoint("model.pt")

        # 训练循环中每轮调用：
        is_best = best_ckpt.check(val_loss, model, optimizer, scheduler, epoch, metrics)

        # 训练结束后加载最优模型：
        best_ckpt.load(model)
    """

    def __init__(self, path=MODEL_PATH, min_delta=1e-6):
        # ↑ 构造函数：初始化管理器
        #
        #   path = MODEL_PATH → checkpoint 保存到哪里
        #   min_delta = 1e-6  → 最小改善阈值
        #
        #   为什么需要 min_delta？
        #     假设上一轮 loss = 0.010000，本轮 loss = 0.009999
        #     改善了 0.000001，微乎其微，可能是噪声
        #     min_delta 过滤掉这种"假改善"，避免频繁保存
        #     只有 loss 下降超过 1e-6 才算"真的改善了"

        self.path       = path
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        # ↑ 历史最佳验证损失，初始化为正无穷大
        #   第一轮的 loss 一定比 inf 小，所以第一轮一定会保存

        self.best_epoch = 0
        # ↑ 最佳 epoch 编号，初始化为 0

        self.best_metrics = {}
        # ↑ 最佳 epoch 的完整指标字典，初始化为空

        self.has_saved  = False
        # ↑ 是否保存过至少一次 checkpoint
        #   训练结束后如果没有保存过，说明出问题了

    def check(self, val_loss, model, optimizer, scheduler, epoch, val_metrics):
        # ↑ 检查当前模型是否是最佳的，如果是则保存 checkpoint
        #
        #   参数说明：
        #     val_loss    → 当前验证集的 MSE 损失（越小越好）
        #     model       → 当前模型（PyTorch nn.Module）
        #     optimizer   → 当前优化器（AdamW）
        #     scheduler   → 当前学习率调度器（ReduceLROnPlateau）
        #     epoch       → 当前是第几轮（从 1 开始）
        #     val_metrics → 当前验证集的完整指标字典
        #
        #   返回值：
        #     True  → 发现了新的最佳模型（已保存到文件）
        #     False → 当前模型不是最佳（没保存）

        if val_loss < self.best_loss - self.min_delta:
            # ↑ 判断条件：当前 loss 比历史最佳 loss 小（且超过 min_delta 阈值）
            #
            #   例：
            #     第 1 轮：best_loss=inf, val_loss=0.05 → 0.05 < inf - 1e-6 ✓ 保存
            #     第 2 轮：best_loss=0.05, val_loss=0.04 → 0.04 < 0.05 - 1e-6 ✓ 保存
            #     第 3 轮：best_loss=0.04, val_loss=0.041 → 0.041 < 0.04 - 1e-6 ✗ 跳过
            #     第 4 轮：best_loss=0.04, val_loss=0.0399999 → 差值 < 1e-6 ✗ 跳过（改善太小）

            # ---- 更新最佳记录 ----
            self.best_loss    = val_loss
            self.best_epoch   = epoch
            self.best_metrics = val_metrics.copy()
            # ↑ .copy() 复制一份，防止后续修改影响保存的值

            self.has_saved    = True

            # ---- 保存完整 checkpoint ----
            checkpoint = {
                "epoch":              epoch,
                # ↑ 保存 epoch 编号，方便加载时知道这是第几轮的模型

                "model_state_dict":   model.state_dict(),
                # ↑ 保存模型的所有权重参数
                #   model.state_dict() 返回一个字典：
                #   {
                #     "net.0.weight": tensor(...),   # 第一层 Linear 的权重
                #     "net.0.bias": tensor(...),     # 第一层 Linear 的偏置
                #     "net.1.weight": tensor(...),   # BatchNorm 的缩放因子
                #     "net.1.bias": tensor(...),     # BatchNorm 的偏移量
                #     ...
                #   }

                "optimizer_state_dict": optimizer.state_dict(),
                # ↑ 保存优化器的内部状态
                #   Adam 优化器会记录：
                #   - 每个参数的"一阶矩"（动量，类似速度）
                #   - 每个参数的"二阶矩"（自适应学习率）
                #   保存后可以断点续训（从上次停的地方继续）

                "scheduler_state_dict": scheduler.state_dict(),
                # ↑ 保存学习率调度器的状态
                #   ReduceLROnPlateau 会记录当前学习率和内部计数器

                "best_loss":          val_loss,
                # ↑ 保存最佳 loss 值

                "best_metrics":       val_metrics.copy(),
                # ↑ 保存最佳 epoch 的完整指标
            }

            torch.save(checkpoint, self.path)
            # ↑ 把 checkpoint 字典保存到文件
            #   torch.save 用 Python 的 pickle 序列化
            #   文件大小通常几十 MB（主要是模型权重）

            return True    # 发现了新的最佳模型

        return False       # 当前模型不是最佳

    def load(self, model, optimizer=None, scheduler=None):
        # ↑ 加载最佳 checkpoint，恢复模型（可选：恢复优化器和调度器）
        #
        #   参数说明：
        #     model     → 需要加载权重的模型（必须和保存时结构一样！）
        #     optimizer → 可选，恢复优化器状态（继续训练时需要）
        #     scheduler → 可选，恢复调度器状态（继续训练时需要）
        #
        #   返回值：
        #     checkpoint 字典（包含 epoch、metrics 等信息）

        checkpoint = torch.load(self.path, map_location=device, weights_only=False)
        # ↑ 从文件加载 checkpoint 字典
        #   map_location=device → 把张量加载到指定设备（CPU 或 GPU）
        #     如果模型是在 GPU 上保存的，但当前没有 GPU，这行确保不会报错
        #   weights_only=False → 允许加载任意 Python 对象（不仅仅是权重）

        model.load_state_dict(checkpoint["model_state_dict"])
        # ↑ 把模型权重加载到模型中
        #   load_state_dict → 从字典恢复权重
        #   模型结构必须和保存时完全一致，否则报错

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # ↑ 如果传了 optimizer 且 checkpoint 里有优化器状态，就恢复
            #   恢复后可以断点续训（学习率、动量等都回到保存时的状态）

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # ↑ 如果传了 scheduler 且 checkpoint 里有调度器状态，就恢复

        return checkpoint

    @property
    def best_log_rmse(self):
        # ↑ @property 让这个方法可以像属性一样访问：best_ckpt.best_log_rmse
        #   而不需要写 best_ckpt.best_log_rmse()
        return self.best_loss ** 0.5
        # ↑ best_loss 是 MSE（均方误差），logRMSE = sqrt(MSE)
        #   ** 0.5 就是开平方根


# ============================================================================
# 【EarlyStopping 类】早停机制
# ============================================================================
#
# 解决的问题：训练多少轮才够？
#
# 如果训练太少轮 → 欠拟合（模型没学会）
# 如果训练太多轮 → 过拟合（模型"背答案"，遇到新数据就不行了）
#
# EarlyStopping 的逻辑：
#   每轮训练后，检查验证集 loss 有没有下降
#   如果下降了 → 计数器归零，继续训练
#   如果没下降 → 计数器 +1
#   如果连续 PATIENCE（40）轮都没下降 → 停止训练
#
# 就像耐心一样：给模型 40 次机会，如果一直不改善，就不再等了
#
# ============================================================================

class EarlyStopping:
    """
    早停机制

    用法：
        early_stop = EarlyStopping(patience=40)

        # 训练循环中每轮调用：
        if early_stop.step(val_loss):
            print("训练终止！")
            break
    """

    def __init__(self, patience=PATIENCE, min_delta=1e-6):
        # ↑ 构造函数
        #
        #   patience = 40   → 最多容忍 40 轮不改善
        #   min_delta = 1e-6 → 最小改善阈值（和 BestCheckpoint 一样）

        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        # ↑ 计数器：记录连续多少轮没有改善
        #   每次有改善 → 归零
        #   每次没改善 → +1
        #   达到 patience → 触发早停

        self.best_loss  = float("inf")
        # ↑ 历史最佳验证损失

        self.early_stop = False
        # ↑ 是否触发了早停

    def step(self, val_loss):
        # ↑ 每轮训练结束后调用，更新内部状态
        #
        #   参数：val_loss → 当前验证损失
        #   返回：True → 触发早停，应该终止训练
        #         False → 还没到停止的时候

        if val_loss < self.best_loss - self.min_delta:
            # ↑ 有改善：当前 loss 比历史最佳小（且超过阈值）
            self.best_loss = val_loss    # 更新最佳 loss
            self.counter   = 0           # 计数器归零（重新开始计数）
        else:
            # ↑ 没改善
            self.counter += 1            # 计数器 +1
            if self.counter >= self.patience:
                # ↑ 连续不改善的次数达到耐心上限
                self.early_stop = True   # 触发早停

        return self.early_stop


# ============================================================================
# 【指标计算函数】evaluate_metrics
# ============================================================================
#
# 这个函数计算 4 个评估指标，全面衡量模型的好坏：
#
#   1. logRMSE（对数均方根误差）→ Kaggle 评分指标，越小越好
#   2. RMSE（均方根误差）→ 原始价格空间的误差，单位是美元
#   3. MAE（平均绝对误差）→ 平均偏差多少美元，更直观
#   4. R²（决定系数）→ 模型解释了多少变异，1.0 = 完美
#
# 为什么需要这么多指标？
#   logRMSE → Kaggle 排名用的，必须算
#   RMSE    → 对大误差敏感（贵房子预测错了惩罚大）
#   MAE     → 对所有误差一视同仁，更直观
#   R²      → 0~1 之间，0.85 意味着模型解释了 85% 的房价变异
#
# ============================================================================

def evaluate_metrics(dataloader, model, loss_fn):
    # ↑ 参数说明：
    #   dataloader → 要评估的数据（验证集或训练集的 DataLoader）
    #   model      → 要评估的模型
    #   loss_fn    → 损失函数（MSELoss）

    model.eval()
    # ↑ 把模型切换到"评估模式"
    #   训练模式 model.train() → Dropout 随机丢弃神经元，BatchNorm 用当前批次的统计量
    #   评估模式 model.eval()  → Dropout 不丢弃，BatchNorm 用全局统计量
    #   必须切换！否则 Dropout 会让每次预测结果不一样

    all_preds, all_labels = [], []
    # ↑ 存放所有预测值和真实值的列表
    #   每个 batch 的结果先 append 到列表，最后 concatenate 合并

    total_mse = 0.0
    # ↑ 累计的 MSE（均方误差）总和

    n_samples = 0
    # ↑ 累计的样本数

    with torch.no_grad():
        # ↑ 关闭梯度计算（评估时不需要反向传播）
        #   不关闭也行，但会白白浪费内存和计算资源
        #   torch.no_grad() 上下文管理器，里面的代码不会计算梯度

        for X, y in dataloader:
            # ↑ 从 DataLoader 取一个 batch 的数据
            #   X → 特征张量，形状 (batch_size, num_features)
            #   y → 标签张量，形状 (batch_size, 1)

            X, y = X.to(device), y.to(device)
            # ↑ 把数据搬到 GPU（如果可用）
            #   模型在 GPU 上，数据也必须在 GPU 上，否则报错

            pred = model(X)
            # ↑ 前向传播：输入特征，得到预测值
            #   pred 形状：(batch_size, 1)

            total_mse += loss_fn(pred, y).item() * X.size(0)
            # ↑ 计算当前 batch 的 MSE，并累加
            #
            #   loss_fn(pred, y) → 计算 MSE，结果是一个标量张量
            #   .item()          → 从张量中取出 Python 数字
            #   * X.size(0)      → 乘以 batch 中的样本数
            #
            #   为什么要乘以样本数？
            #     loss_fn 返回的是"平均 MSE"（每个样本的平均）
            #     乘以样本数 = 这个 batch 的"总 MSE"
            #     最后除以总样本数 = 整个数据集的平均 MSE
            #
            #   例：batch_size=64，loss=0.05
            #     total_mse += 0.05 * 64 = 3.2

            n_samples += X.size(0)
            # ↑ 累加样本数
            #   X.size(0) = batch 中的样本数（通常是 64，最后一个 batch 可能更少）

            all_preds.append(pred.squeeze(1).cpu().numpy())
            # ↑ 保存当前 batch 的预测值
            #
            #   pred.squeeze(1) → 把形状从 (64, 1) 压缩成 (64,)
            #     squeeze 移除维度为 1 的维度
            #     因为我们要的是一维数组，不是二维
            #
            #   .cpu() → 从 GPU 搬回 CPU（numpy 只能在 CPU 上用）
            #
            #   .numpy() → 从 PyTorch 张量转为 NumPy 数组
            #
            #   .append() → 添加到列表末尾

            all_labels.append(y.squeeze(1).cpu().numpy())
            # ↑ 保存当前 batch 的真实标签（处理方式同上）

    preds_log  = np.concatenate(all_preds)
    # ↑ 把所有 batch 的预测值拼接成一个大数组
    #   例：[array([12.1, 12.3]), array([12.5, 12.0])] → array([12.1, 12.3, 12.5, 12.0])

    labels_log = np.concatenate(all_labels)
    # ↑ 把所有 batch 的真实标签拼接成一个大数组

    preds  = np.expm1(preds_log)
    # ↑ 把 log 空间的预测值还原回原始价格
    #   expm1(x) = e^x - 1，是 log1p 的逆运算
    #   例：log1p(208500) = 12.247 → expm1(12.247) = 208500

    labels = np.expm1(labels_log)
    # ↑ 把 log 空间的真实标签还原回原始价格

    log_rmse = np.sqrt(total_mse / n_samples)
    # ↑ 计算 logRMSE（Kaggle 评分指标）
    #
    #   total_mse / n_samples = 平均 MSE（在 log 空间）
    #   np.sqrt(...) = 开平方根 = RMSE
    #
    #   这个值越小越好，Kaggle 排名就看这个
    #   典型的好成绩：logRMSE 在 0.10 ~ 0.15 之间

    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    # ↑ 计算原始价格空间的 RMSE
    #
    #   preds - labels → 每个样本的预测误差
    #   (...) ** 2     → 误差的平方（消除正负号）
    #   np.mean(...)   → 取平均
    #   np.sqrt(...)   → 开平方根
    #
    #   单位是美元，例：RMSE = $25,000 意味着平均偏差 2.5 万美元

    mae = np.mean(np.abs(preds - labels))
    # ↑ 计算 MAE（平均绝对误差）
    #
    #   np.abs(...) → 取绝对值（消除正负号）
    #   np.mean(...) → 取平均
    #
    #   和 RMSE 的区别：
    #     RMSE 对大误差更敏感（因为平方会放大大误差）
    #     MAE  对所有误差一视同仁
    #
    #   例：预测 20 万，实际 25 万 → MAE = 5 万
    #       预测 20 万，实际 20.1 万 → MAE = 0.1 万

    ss_res = np.sum((labels - preds) ** 2)
    # ↑ 残差平方和：模型的预测误差总和

    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    # ↑ 总平方和：真实值的变异程度
    #   如果不用模型，直接用均值预测，误差就是这个

    r2 = 1.0 - ss_res / ss_tot
    # ↑ R² 决定系数：模型解释了多少变异
    #
    #   R² = 1 - (模型误差 / 均值误差)
    #
    #   R² = 1.0 → 完美预测（模型误差 = 0）
    #   R² = 0.85 → 模型解释了 85% 的房价变异，还有 15% 没解释
    #   R² = 0.0 → 模型和直接用均值一样差
    #   R² < 0 → 模型比均值还差（可能数据有问题或模型太烂）

    return {"log_rmse": log_rmse, "rmse": rmse, "mae": mae, "r2": r2}
    # ↑ 返回一个字典，包含所有 4 个指标


# ============================================================================
# 【数据集类】HousePricesDataset
# ============================================================================
#
# 为什么要自定义 Dataset？
#   PyTorch 的 DataLoader 需要一个 Dataset 对象来获取数据
#   Dataset 就像一个"数据仓库"，告诉 DataLoader：
#   - 总共有多少条数据？（__len__）
#   - 给我第 i 条数据长什么样？（__getitem__）
#
# 自定义 Dataset 必须实现 3 个方法：
#   __init__  → 初始化：把原始数据转成 PyTorch 张量
#   __len__   → 返回数据总数
#   __getitem__ → 返回第 idx 条数据（特征 + 标签）
#
# 为什么要把 numpy 数组转成 torch 张量？
#   numpy 是 CPU 上的数组，不能参与 GPU 计算和自动求导
#   torch.Tensor 是 PyTorch 的数据格式，支持 GPU 和梯度计算
#
# ============================================================================

class HousePricesDataset(Dataset):
    # ↑ 继承 Dataset 基类，这样 DataLoader 才能用

    def __init__(self, X, y=None):
        # ↑ 构造函数
        #
        #   X → 特征矩阵，numpy 数组，形状 (样本数, 特征数)
        #   y → 标签向量，numpy 数组，形状 (样本数,)
        #       测试集没有标签，所以 y 默认为 None

        self.X = torch.tensor(X, dtype=torch.float32)
        # ↑ 把特征矩阵从 numpy 转为 torch 张量
        #   dtype=torch.float32 → 32 位浮点数（PyTorch 默认格式）
        #   转换后 self.X 形状仍然是 (样本数, 特征数)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            # ↑ 把标签从 numpy 转为 torch 张量，并 reshape
            #
            #   原始 y 形状：(1460,) → 一维数组
            #   .view(-1, 1) → 变成 (1460, 1) → 二维列向量
            #
            #   为什么要 view(-1, 1)？
            #     模型输出形状是 (batch_size, 1)
            #     标签也必须是 (batch_size, 1) 才能计算 loss
            #     -1 表示"自动计算这一维的大小"
        else:
            self.y = None
            # ↑ 测试集没有标签

        print(f"数据集创建完成：{len(self.X)} 条样本，{self.X.shape[1]} 个特征")

    def __len__(self):
        # ↑ 返回数据集的总样本数
        #   DataLoader 需要知道总共有多少数据，才能分 batch
        return len(self.X)

    def __getitem__(self, idx):
        # ↑ 返回第 idx 条数据
        #
        #   DataLoader 会随机选一些索引（如 [3, 17, 42, ...]）
        #   然后调用 __getitem__ 获取对应的数据
        #   最后把这些数据堆叠成一个 batch
        #
        #   参数：idx → 索引（可以是单个数字，也可以是列表）
        #   返回：(特征, 标签) 元组

        if self.y is not None:
            return self.X[idx], self.y[idx]
            # ↑ 有标签时返回 (特征, 标签) 元组
            #   训练集和验证集走这个分支
        else:
            return self.X[idx]
            # ↑ 没标签时只返回特征
            #   测试集走这个分支


# ============================================================================
# 【主程序】
# ============================================================================

# ============================================================================
# 1. 读取数据
# ============================================================================

train_df = pd.read_csv(TRAIN_PATH)
# ↑ 读取训练集 CSV 文件，得到一个 DataFrame（表格）
#   DataFrame 就像 Excel 表格：有行（样本）和列（特征）
#   train_df 形状：(1460, 81)
#   1460 行 = 1460 套房子
#   81 列 = Id + SalePrice + 79 个特征

test_df  = pd.read_csv(TEST_PATH)
# ↑ 读取测试集 CSV 文件
#   test_df 形状：(1459, 80)
#   1459 行 = 1459 套房子
#   80 列 = Id + 79 个特征（没有 SalePrice，因为这就是我们要预测的）

print("=" * 60)
print("【数据预览】")
print("=" * 60)
print(train_df.head())
# ↑ 打印前 5 行数据，快速预览
#   head() 默认显示前 5 行

print(f"\n训练集形状：{train_df.shape}")
# ↑ 打印训练集形状，如 (1460, 81)

print(f"测试集形状：{test_df.shape}")
# ↑ 打印测试集形状，如 (1459, 80)

# ----------------------------------------------------------------------------
# 2. 特征工程
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("【特征工程】")
print("=" * 60)

# ============================================================================
# 【特征工程】逐行详解
# ============================================================================

# ============================================================================
# 第一步：保存测试集 Id 和提取训练标签
# ============================================================================

test_ids = test_df["Id"].copy()
# ↑ 保存测试集的 Id 列，生成提交文件时需要用到
#   test_df["Id"] → 取出 Id 这一列
#   .copy()       → 复制一份，避免后续修改影响原始数据
#   结果：test_ids = [1461, 1462, 1463, ..., 2919]

y_train = np.log1p(train_df["SalePrice"].values).astype(np.float32)
# ↑ 提取训练标签，并做 log 变换
#
#   train_df["SalePrice"]      → 取出房价这一列
#   .values                    → 从 pandas Series 转为 numpy 数组
#                                [208500, 181500, 223500, ...]
#
#   np.log1p(...)              → 对每个值做 log(1 + x) 变换
#                                log1p(208500) = 12.247
#                                log1p(181500) = 12.109
#
#                                为什么用 log1p 不用 log？
#                                因为 log(0) = 负无穷，会出问题
#                                log1p(0) = log(1) = 0，安全
#
#   .astype(np.float32)        → 转为 32 位浮点数（PyTorch 的标准格式）
#
#   结果：y_train = [12.247, 12.109, 12.317, ...]
#   含义：log 空间的房价，作为模型的训练目标


# ============================================================================
# 第二步：删除无用列，分离出纯特征矩阵
# ============================================================================

train_features = train_df.drop(columns=["SalePrice", "Id"])
# ↑ 从训练集中删除 SalePrice（标签）和 Id（编号）
#
#   drop(columns=["SalePrice", "Id"]) → 删除这两列，保留剩下的 79 列特征
#
#   删除 SalePrice → 因为它是标签，不是特征，不能喂给模型
#   删除 Id        → 因为 Id 只是行编号（1, 2, 3, ...），跟房价无关
#                     如果保留，模型会错误地认为"编号越大房价越高"
#
#   结果：train_features 形状 (1460, 79)
#         包含 MSSubClass, MSZoning, LotArea, ... 等 79 个特征

test_features = test_df.drop(columns=["Id"])
# ↑ 从测试集中删除 Id（测试集没有 SalePrice，所以只删 Id）
#
#   结果：test_features 形状 (1459, 79)
#         和训练集有完全相同的 79 列特征


# ============================================================================
# 第三步：自动识别数值列和类别列
# ============================================================================

num_cols = train_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
# ↑ 找出所有数值类型的列名
#
#   select_dtypes(include=["int64", "float64"]) → 筛选数据类型为整数或浮点数的列
#   .columns                                    → 取出这些列的列名
#   .tolist()                                   → 转为 Python 列表
#
#   结果：num_cols = ["MSSubClass", "LotFrontage", "LotArea", "OverallQual",
#                     "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea",
#                     "BsmtFinSF1", "BsmtFinSF2", ...]
#
#   这些列的值本身就是数字，如：
#     LotArea = 8450（面积）
#     YearBuilt = 2003（建造年份）
#     FullBath = 2（卫生间数）

cat_cols = train_features.select_dtypes(include=["object"]).columns.tolist()
# ↑ 找出所有文字类型的列名
#
#   select_dtypes(include=["object"]) → 筛选数据类型为字符串的列
#
#   结果：cat_cols = ["MSZoning", "Street", "Alley", "LotShape", "LandContour",
#                     "Utilities", "LotConfig", "LandSlope", "Neighborhood",
#                     "HouseStyle", "RoofStyle", ...]
#
#   这些列的值是文字，如：
#     MSZoning = "RL"（区域类型）
#     HouseStyle = "2Story"（房屋风格）
#     Neighborhood = "CollgCr"（社区名）

print(f"数值列：{len(num_cols)} 个，类别列：{len(cat_cols)} 个")
# ↑ 打印：数值列：36 个，类别列：43 个


# ============================================================================
# 第四步：填充缺失值
# ============================================================================

medians = train_features[num_cols].median().fillna(0)
# ↑ 计算训练集每个数值列的中位数
#
#   train_features[num_cols] → 取出所有数值列（36 列）
#   .median()                → 对每一列求中位数（返回一个 Series）
#
#   结果长这样：
#     LotFrontage     69.0
#     LotArea       9478.5
#     YearBuilt     1973.0
#     FullBath         2.0
#     ...
#
#   .fillna(0) → 如果某列全是缺失值，中位数也是 NaN，填成 0 保底
#
#   为什么用中位数不用均值？
#     假设 100 套房的面积：99 套是 5000~15000，1 套是 500000（异常值）
#     均值 = (99×10000 + 500000) / 100 = 14900  ← 被异常值拉高
#     中位数 = 10000                           ← 不受异常值影响

train_features[num_cols] = train_features[num_cols].fillna(medians)
# ↑ 用训练集的中位数填充训练集的缺失值
#
#   例：LotFrontage 列有 259 个缺失值
#   这 259 个 NaN 全部被替换成 69.0（该列的中位数）
#
#   fillna(medians) 的工作方式：
#     medians = {"LotFrontage": 69.0, "LotArea": 9478.5, ...}
#     LotFrontage 列的 NaN → 填 69.0
#     LotArea 列的 NaN     → 填 9478.5
#     每列用自己的中位数填充

test_features[num_cols] = test_features[num_cols].fillna(medians)
# ↑ 用训练集的中位数填充测试集的缺失值
#
#   【关键】测试集用的是训练集的 medians，不是自己的中位数
#
#   为什么？
#     这是防止"数据泄漏"的铁律
#     测试集模拟的是"未来的新数据"
#     你在训练时不能以任何方式"偷看"测试集的信息
#     如果用测试集自己的中位数 → 间接利用了测试集的分布信息 → 不公平

train_features[cat_cols] = train_features[cat_cols].fillna("Missing")
# ↑ 训练集的类别列缺失值填成字符串 "Missing"
#
#   例：Alley 列（巷子类型）有大量缺失
#   原始值：NaN, NaN, "Pave", "Grvl", NaN, ...
#   填充后："Missing", "Missing", "Pave", "Grvl", "Missing", ...
#
#   为什么填 "Missing" 不填众数（出现最多的值）？
#     因为有些缺失本身就是信息——"没有巷子"和"有巷子"是不同的含义
#     填 "Missing" 让模型能学到"缺失"这个模式

test_features[cat_cols] = test_features[cat_cols].fillna("Missing")
# ↑ 测试集的类别列也填 "Missing"
#   训练集和测试集用相同的填充值，保证一致性


# ============================================================================
# 第五步：合并训练集和测试集，统一做 One-Hot 编码
# ============================================================================

combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
# ↑ 把训练集和测试集上下拼接成一个大表
#
#   pd.concat([...], axis=0) → axis=0 表示上下拼接（行数增加）
#   ignore_index=True        → 重新编号 0, 1, 2, ...（不保留原来的行号）
#
#   拼接前：
#     train_features 形状 (1460, 79)
#     test_features  形状 (1459, 79)
#
#   拼接后：
#     combined 形状 (2919, 79)
#     行 0~1459   是训练集
#     行 1460~2918 是测试集
#
#   为什么要合并？
#     如果分开做 One-Hot 编码，两边的列可能对不上
#
#     训练集 MSZoning 有：RL, RM, FV          → 编码后 3 列
#     测试集 MSZoning 有：RL, RM, C (all)     → 编码后 3 列
#                                               但列名不同！
#     合并后编码：RL, RM, FV, C (all), RH     → 统一 5 列
#     训练集和测试集的列完全一致，不会出错

combined = pd.get_dummies(combined, columns=cat_cols).astype(np.float32)
# ↑ 对所有类别列做 One-Hot 编码
#
#   pd.get_dummies(combined, columns=cat_cols)
#     → 把 cat_cols 指定的每个类别列展开成多个 0/1 列
#
#   编码前（MSZoning 列）：
#     "RL"
#     "RM"
#     "FV"
#     "RL"
#
#   编码后（MSZoning 列被删掉，变成 5 个新列）：
#     MSZoning_RL  MSZoning_RM  MSZoning_FV  MSZoning_C(all)  MSZoning_RH
#        1             0            0              0              0
#        0             1            0              0              0
#        0             0            1              0              0
#        1             0            0              0              0
#
#   为什么叫 "One-Hot"？每行只有一个 1（hot），其余全是 0
#
#   .astype(np.float32) → 转为 float32 类型（PyTorch 需要）
#
#   编码后 combined 形状：(2919, ~286)
#     79 列 → 约 286 列（43 个类别列展开成了约 207 个 0/1 列）


# ============================================================================
# 第六步：拆回训练集和测试集
# ============================================================================

X_train_raw = combined.iloc[:len(train_df)].values
# ↑ 取前 1460 行作为训练集特征
#
#   combined.iloc[:1460] → 取行号 0 到 1459（就是原来的训练集）
#   .values              → 从 pandas DataFrame 转为 numpy 数组
#
#   结果：X_train_raw 形状 (1460, ~286)
#   每一行是一个房子的特征，全是数字

X_test_raw = combined.iloc[len(train_df):].values
# ↑ 取后 1459 行作为测试集特征
#
#   combined.iloc[1460:] → 取行号 1460 到 2918（就是原来的测试集）
#
#   结果：X_test_raw 形状 (1459, ~286)
#   和训练集有完全相同的列数和列顺序


# ============================================================================
# 第七步：标准化（均值归零，方差归一）
# ============================================================================

scaler = StandardScaler()
# ↑ 创建标准化器
#   StandardScaler 的公式：(x - 均值) / 标准差
#   处理后每列均值 ≈ 0，标准差 ≈ 1

X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
# ↑ 对训练集做标准化：fit + transform（两步合成一步）
#
#   fit_transform = fit + transform
#   fit     → 计算每列的均值和标准差（只算，不改数据）
#   transform → 用算出来的均值和标准差，对数据做标准化
#
#   例：LotArea 列
#     原始值：8450, 9600, 11250, 14000, ...
#     fit 算出：均值 = 10000，标准差 = 3000
#     transform：
#       (8450 - 10000) / 3000 = -0.52
#       (9600 - 10000) / 3000 = -0.13
#       (11250 - 10000) / 3000 = 0.42
#       (14000 - 10000) / 3000 = 1.33
#
#   .astype(np.float32) → 转为 float32
#
#   结果：X_train 形状 (1460, ~286)，值域大约在 -3 到 +3 之间
#
#   为什么要做标准化？
#     LotArea 列：数值范围 1000 ~ 200000（差距 200 倍）
#     FullBath 列：数值范围 0 ~ 4（差距 4 倍）
#     不标准化 → 模型认为 LotArea 更重要（因为数字大）
#     标准化后 → 所有列在同一个量级，模型公平对待每个特征

X_test = scaler.transform(X_test_raw).astype(np.float32)
# ↑ 对测试集做标准化：只用 transform（不用 fit）
#
#   transform → 直接用训练集的均值和标准差来标准化测试集
#
#   例：测试集 LotArea 的一个值是 12000
#     用训练集的参数：(12000 - 10000) / 3000 = 0.67
#
#   为什么不用 fit_transform？
#     fit_transform = 先算测试集自己的均值标准差，再标准化
#     这等于"偷看"了测试集的信息
#     正确做法：只用训练集的参数，测试集直接套用
#
#   结果：X_test 形状 (1459, ~286)

print(f"特征工程完成：训练集 {X_train.shape}，测试集 {X_test.shape}")
# ↑ 打印最终形状
#   输出：特征工程完成：训练集 (1460, 286)，测试集 (1459, 286)

# ============================================================================
# 3. 创建 Dataset & 划分训练集/验证集
# ============================================================================

train_set = HousePricesDataset(X_train, y_train)
# ↑ 创建训练集的 Dataset 对象
#   X_train → 特征矩阵 (1460, ~286)
#   y_train → 标签向量 (1460,)
#   Dataset 内部会把它们转成 torch 张量

test_set  = HousePricesDataset(X_test)
# ↑ 创建测试集的 Dataset 对象
#   X_test → 特征矩阵 (1459, ~286)
#   没有标签（y=None），因为测试集的房价是我们要预测的

train_size = int(len(train_set) * (1 - VAL_RATIO))
# ↑ 计算训练子集的大小
#   len(train_set) = 1460
#   1 - VAL_RATIO = 1 - 0.2 = 0.8
#   int(1460 * 0.8) = int(1168.0) = 1168

val_size   = len(train_set) - train_size
# ↑ 计算验证子集的大小
#   1460 - 1168 = 292

train_subset, val_subset = random_split(
    train_set, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
# ↑ 把 train_set 随机分成两部分：训练子集和验证子集
#
#   random_split(数据集, [大小1, 大小2], generator=随机数生成器)
#
#   train_subset → 1168 条，用于训练模型
#   val_subset   → 292 条，用于验证模型好坏
#
#   generator=torch.Generator().manual_seed(SEED)
#     → 固定随机种子，保证每次划分的结果一样
#     → 如果不固定，每次运行训练集和验证集的分法不同，结果无法复现
#
#   为什么要分验证集？
#     训练集 → 模型用来学习（看答案学习）
#     验证集 → 模型用来自测（模拟考试）
#     如果只用训练集评估，模型可能"背答案"（过拟合），看起来很好但实际不行
#     验证集模拟"没见过的新数据"，能真实反映模型水平

print(f"\n数据划分：训练集 {train_size} 条，验证集 {val_size} 条")


# ============================================================================
# 4. DataLoader：自动分 batch + 打乱顺序
# ============================================================================

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
# ↑ 训练集的 DataLoader
#
#   train_subset → 数据源（1168 条）
#   batch_size=64 → 每次取 64 条数据组成一个 batch
#   shuffle=True  → 每个 epoch 开始时打乱数据顺序
#
#   为什么要 shuffle？
#     如果数据顺序固定（如按 Id 排列），模型可能会学到"顺序"而不是"规律"
#     打乱后，每个 epoch 的 batch 组合都不同，训练更稳定
#
#   1168 条数据 / 64 = 18.25 → 每个 epoch 有 18 个完整 batch + 1 个 16 条的小 batch

val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)
# ↑ 验证集的 DataLoader
#   shuffle=False → 验证时不需要打乱，顺序无所谓
#   而且固定顺序方便调试和对比

test_loader  = DataLoader(test_set,     batch_size=BATCH_SIZE, shuffle=False)
# ↑ 测试集的 DataLoader
#   shuffle=False → 测试时不需要打乱


# ============================================================================
# 【模型定义】NeuralNetwork
# ============================================================================
#
# 这是一个 5 层全连接神经网络（也叫 MLP，多层感知机）
#
# 网络结构：
#   输入层：~286 个特征
#   隐藏层1：512 个神经元 → BatchNorm → ReLU → Dropout(30%)
#   隐藏层2：256 个神经元 → BatchNorm → ReLU → Dropout(30%)
#   隐藏层3：128 个神经元 → BatchNorm → ReLU → Dropout(20%)
#   隐藏层4：64 个神经元  → ReLU（没有 Dropout）
#   输出层：1 个神经元（预测房价）
#
# 为什么用这种"漏斗形"结构（512→256→128→64→1）？
#   逐步压缩信息，从高维特征提取到低维预测
#   就像漏斗：宽口进，窄口出
#
# 各组件的作用：
#   Linear      → 全连接层：y = Wx + b，学习特征的线性组合
#   BatchNorm   → 批归一化：加速训练，稳定梯度
#   ReLU        → 激活函数：引入非线性，让网络能学复杂模式
#   Dropout     → 随机丢弃：防止过拟合（训练时随机关闭一些神经元）
#
# ============================================================================

class NeuralNetwork(nn.Module):
    # ↑ 继承 nn.Module，这是所有 PyTorch 模型的基类

    def __init__(self, input_dim):
        # ↑ 构造函数
        #   input_dim → 输入特征的数量（约 286）

        super().__init__()
        # ↑ 调用父类 nn.Module 的构造函数（必须写！）

        self.net = nn.Sequential(
            # ↑ nn.Sequential：按顺序串联各层，数据从第一层流到最后一层

            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.30),
            # ↑ 第一层：input_dim → 512
            #
            #   nn.Linear(input_dim, 512)
            #     → 全连接层：把 input_dim 维输入映射到 512 维
            #     → 内部参数：权重矩阵 W (512, input_dim) + 偏置向量 b (512,)
            #     → 计算：output = input @ W.T + b
            #
            #   nn.BatchNorm1d(512)
            #     → 批归一化：对每个 batch 的 512 维输出做标准化
            #     → 公式：(x - batch_mean) / sqrt(batch_var + eps)
            #     → 作用：加速训练，减少对初始化的敏感度
            #
            #   nn.ReLU()
            #     → 激活函数：ReLU(x) = max(0, x)
            #     → 负数变 0，正数不变
            #     → 引入非线性，让网络能学复杂模式
            #     → 为什么叫 ReLU？Rectified Linear Unit（修正线性单元）
            #
            #   nn.Dropout(0.30)
            #     → 训练时随机把 30% 的神经元输出设为 0
            #     → 作用：防止过拟合（强迫网络不依赖某些特定神经元）
            #     → 类似"考试时随机去掉一些知识点，逼你全面学习"
            #     → 评估时不用 Dropout（所有神经元都参与）

            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            # ↑ 第二层：512 → 256，结构同上

            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            # ↑ 第三层：256 → 128
            #   Dropout 降到 20%（越往后越少丢弃，保留更多信息）

            nn.Linear(128, 64),        nn.ReLU(),
            # ↑ 第四层：128 → 64
            #   没有 BatchNorm 和 Dropout（快到输出了，不需要太多正则化）

            nn.Linear(64, 1)
            # ↑ 输出层：64 → 1
            #   最终输出一个数字：预测的 log 房价
            #   没有激活函数（因为我们要预测连续值，不需要压缩到某个范围）
        )

    def forward(self, x):
        # ↑ 前向传播：定义数据如何流过网络
        #   x → 输入张量，形状 (batch_size, input_dim)
        #   返回 → 预测值，形状 (batch_size, 1)
        return self.net(x)
        # ↑ 数据依次流过 self.net 中的每一层


input_dim = X_train.shape[1]
# ↑ 获取特征数量
#   X_train.shape = (1460, 286)
#   X_train.shape[1] = 286（第二个维度 = 特征数）

model     = NeuralNetwork(input_dim=input_dim).to(device)
# ↑ 创建模型，并搬到 GPU（如果可用）
#
#   NeuralNetwork(input_dim=286) → 创建一个输入维度为 286 的神经网络
#   .to(device) → 把模型的所有参数搬到 GPU 或 CPU
#
#   模型创建后，内部的权重是随机初始化的
#   训练过程就是不断调整这些权重，让预测越来越准

print(f"\n模型初始化完成：输入维度={input_dim}，设备={device}")


# ============================================================================
# 【损失函数、优化器、调度器】
# ============================================================================
#
# 这三个组件是训练神经网络的核心：
#
#   损失函数（Loss）→ 告诉模型"你错了多少"
#   优化器（Optimizer）→ 告诉模型"怎么改才能更准"
#   调度器（Scheduler）→ 告诉模型"学太快了就慢下来"
#
# 训练过程就像爬山找最低点：
#   Loss 是"你现在在多高的位置"
#   Optimizer 是"往哪个方向走一步"
#   Scheduler 是"走太快了就缩小步幅"
#
# ============================================================================

loss_fn    = nn.MSELoss()
# ↑ 均方误差损失函数（Mean Squared Error）
#
#   MSE = mean((预测值 - 真实值)²)
#
#   例：预测 [12.3, 12.1]，真实 [12.2, 12.0]
#     误差 = [(12.3-12.2)², (12.1-12.0)²] = [0.01, 0.01]
#     MSE = mean([0.01, 0.01]) = 0.01
#
#   为什么用 MSE 不用 MAE？
#     MSE 对大误差更敏感（平方会放大大误差）
#     梯度更好算（MSE 的导数是线性的，MAE 的导数在 0 处不连续）
#     MSE 是回归问题的标准选择

optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# ↑ AdamW 优化器（Adam + 权重衰减修正版）
#
#   model.parameters() → 获取模型的所有可训练参数（权重和偏置）
#   lr=LEARNING_RATE   → 学习率 = 0.001（步长）
#   weight_decay=WEIGHT_DECAY → 权重衰减 = 0.0001（L2 正则化）
#
#   Adam 是什么？
#     Adaptive Moment Estimation（自适应矩估计）
#     结合了 SGD 的动量（Momentum）和 RMSprop 的自适应学习率
#     对每个参数自动调整学习率（参数更新幅度大→降低学习率，反之提高）
#     是目前最常用的优化器
#
#   AdamW 和 Adam 的区别？
#     Adam 的权重衰减实现有 bug（衰减和梯度更新耦合了）
#     AdamW 把权重衰减从梯度更新中分离出来，更正确
#     实际效果：正则化更强，泛化更好

scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
# ↑ 学习率调度器：当验证 loss 停滞时自动降低学习率
#
#   ReduceLROnPlateau = "在平台期降低学习率"
#
#   optimizer → 要调度的优化器
#   mode='min' → 监控指标越小越好（loss 越小越好）
#   factor=0.5 → 每次降低时，学习率乘以 0.5（减半）
#   patience=10 → 连续 10 个 epoch 没改善才降低
#
#   学习率变化过程示例：
#     epoch 1~30:   lr = 0.001    （正常学习）
#     epoch 31~40:  loss 停滞 10 轮 → lr = 0.0005   （减半）
#     epoch 41~50:  loss 又停滞 10 轮 → lr = 0.00025  （再减半）
#     ...
#
#   为什么要降低学习率？
#     训练初期：需要大步快走，快速接近最优解
#     训练后期：需要小步微调，在最优解附近精细搜索
#     如果一直用大学习率，会在最优解附近来回震荡


# ============================================================================
# 【训练函数 & 验证函数】
# ============================================================================
#
# PyTorch 训练循环的标准写法（必须背下来！）：
#
#   训练模式：
#     1. model.train()     → 切换到训练模式
#     2. 前向传播          → 得到预测值
#     3. 计算 loss         → 得到误差
#     4. optimizer.zero_grad() → 清空之前的梯度
#     5. loss.backward()   → 反向传播，计算梯度
#     6. optimizer.step()  → 用梯度更新参数
#
#   验证模式：
#     1. model.eval()      → 切换到评估模式
#     2. torch.no_grad()   → 关闭梯度计算
#     3. 前向传播          → 得到预测值
#     4. 计算 loss         → 得到误差（只记录，不更新参数）
#
# ============================================================================

def train_loop(dataloader, model, loss_fn, optimizer):
    # ↑ 训练一个 epoch（把全部训练数据看一遍）
    #
    #   dataloader → 训练集的 DataLoader
    #   model      → 要训练的模型
    #   loss_fn    → 损失函数
    #   optimizer  → 优化器
    #
    #   返回：这个 epoch 的平均训练 loss

    model.train()
    # ↑ 切换到训练模式
    #   Dropout 会随机丢弃神经元
    #   BatchNorm 会用当前 batch 的均值和方差

    total_loss = 0.0
    # ↑ 累计 loss

    for X, y in dataloader:
        # ↑ 从 DataLoader 取一个 batch
        #   X → 特征，形状 (64, 286)
        #   y → 标签，形状 (64, 1)

        X, y = X.to(device), y.to(device)
        # ↑ 搬到 GPU（如果可用）

        pred = model(X)
        # ↑ 前向传播：输入特征，得到预测值
        #   pred 形状：(64, 1)

        loss = loss_fn(pred, y)
        # ↑ 计算损失：预测值和真实值的差距
        #   loss 是一个标量张量（单个数字）

        optimizer.zero_grad()
        # ↑ 清空之前的梯度！
        #
        #   为什么要清空？
        #     PyTorch 默认会累加梯度（方便某些场景）
        #     如果不清空，梯度会越来越大，训练会爆炸
        #     所以每次更新前必须清零

        loss.backward()
        # ↑ 反向传播：计算每个参数的梯度
        #
        #   梯度 = "loss 对参数的偏导数"
        #   梯度告诉每个参数："你应该变大还是变小，变多少"
        #
        #   这一步是 PyTorch 的核心魔法：
        #     自动求导（Autograd）自动计算所有梯度
        #     你只需要定义前向传播，梯度自动算

        optimizer.step()
        # ↑ 用梯度更新参数
        #
        #   Adam 更新公式（简化版）：
        #     参数 = 参数 - 学习率 × 梯度 / sqrt(梯度的二阶矩)
        #
        #   每个参数都被微调一点，让 loss 变小

        total_loss += loss.item() * X.size(0)
        # ↑ 累加当前 batch 的 loss
        #   loss.item() → 从张量取出 Python 数字
        #   * X.size(0) → 乘以 batch 大小（因为 loss 是平均的）

    return total_loss / len(dataloader.dataset)
    # ↑ 返回整个 epoch 的平均 loss
    #   total_loss / 总样本数 = 平均每条数据的 loss


def val_loop(dataloader, model, loss_fn):
    # ↑ 验证一个 epoch
    #
    #   和 train_loop 的区别：
    #     1. 用 model.eval() 而不是 model.train()
    #     2. 用 torch.no_grad() 关闭梯度计算
    #     3. 不调用 optimizer.step()（不更新参数）
    #     4. 不调用 optimizer.zero_grad()（不需要清空梯度）

    model.eval()
    # ↑ 切换到评估模式
    #   Dropout 不丢弃（所有神经元参与）
    #   BatchNorm 用全局统计量

    total_loss = 0.0
    with torch.no_grad():
        # ↑ 关闭梯度计算（节省内存和计算）
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)


# ============================================================================
# 【训练主循环】
# ============================================================================
#
# 这是整个程序的核心！把前面所有组件串起来：
#
#   for epoch in range(500):
#       1. 训练模型（train_loop）
#       2. 验证模型（evaluate_metrics）
#       3. 学习率调度（scheduler.step）
#       4. 检查是否是最佳模型（best_ckpt.check）
#       5. 打印训练状态
#       6. 检查是否该停止（early_stop.step）
#
# 每个 epoch 就是一轮完整的"训练 + 验证"
#
# ============================================================================

print("\n" + "=" * 130)
print("【开始训练】")
print("=" * 130)

# ---- 创建两个独立组件 ----
best_ckpt    = BestCheckpoint(path=MODEL_PATH)
# ↑ 创建最佳模型保存器
#   负责在每轮训练后检查并保存最优模型

early_stop   = EarlyStopping(patience=PATIENCE)
# ↑ 创建早停器
#   负责判断"连续 40 轮没改善就停"

for epoch in range(1, EPOCHS + 1):
    # ↑ 主循环：从第 1 轮到第 500 轮
    #   range(1, 501) → 1, 2, 3, ..., 500

    # ---- 第 1 步：训练 ----
    train_mse = train_loop(train_loader, model, loss_fn, optimizer)
    # ↑ 训练一个 epoch，返回训练集的平均 MSE
    #   训练过程中模型权重会被更新

    # ---- 第 2 步：验证 ----
    val_metrics = evaluate_metrics(val_loader, model, loss_fn)
    # ↑ 在验证集上评估模型，返回完整指标字典
    #   val_metrics = {"log_rmse": ..., "rmse": ..., "mae": ..., "r2": ...}

    val_mse = val_metrics["log_rmse"] ** 2
    # ↑ 从 log_rmse 反推 MSE
    #   因为 log_rmse = sqrt(MSE)，所以 MSE = log_rmse²
    #   用 MSE 而不是 log_rmse 作为比较标准（数值更敏感）

    # ---- 第 3 步：学习率调度 ----
    scheduler.step(val_mse)
    # ↑ 告诉调度器当前的验证 loss
    #   如果连续 10 轮没改善，调度器会自动把学习率减半
    #   注意：必须传 val_mse，不能传 train_mse
    #   因为我们要监控的是"模型在新数据上的表现"

    # ---- 第 4 步：检查是否是最佳模型 ----
    is_best = best_ckpt.check(val_mse, model, optimizer, scheduler, epoch, val_metrics)
    # ↑ 检查当前模型是否是历史最佳
    #   如果是 → 保存 checkpoint 到文件，返回 True
    #   如果不是 → 跳过，返回 False

    # ---- 第 5 步：打印训练状态 ----
    if epoch % 10 == 0 or epoch == 1:
        # ↑ 每 10 轮打印一次（第 1 轮也打印）
        #   不每轮都打印是因为太吵了（500 轮 × 18 个 batch = 9000 行输出）

        train_metrics = evaluate_metrics(train_loader, model, loss_fn)
        # ↑ 重新在训练集上评估（为了对比训练集和验证集的表现）
        #   如果训练集指标远好于验证集 → 过拟合了

        best_flag = " ★ NEW BEST" if is_best else ""
        # ↑ 如果当前是最佳模型，加上星号标记

        print(
            f"Epoch {epoch:>3} | "
            # ↑ :>3 表示右对齐，占 3 个字符宽度

            f"[Train] logRMSE={train_metrics['log_rmse']:.5f}  "
            f"R2={train_metrics['r2']:.4f}  "
            f"RMSE=${train_metrics['rmse']:>10,.0f}  "
            f"MAE=${train_metrics['mae']:>10,.0f} | "
            # ↑ 训练集指标
            #   :.5f → 保留 5 位小数
            #   :>10,.0f → 右对齐，占 10 位，千分位逗号，无小数

            f"[Val] logRMSE={val_metrics['log_rmse']:.5f}  "
            f"R2={val_metrics['r2']:.4f}  "
            f"RMSE=${val_metrics['rmse']:>10,.0f}  "
            f"MAE=${val_metrics['mae']:>10,.0f} | "
            # ↑ 验证集指标

            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            # ↑ 当前学习率
            #   optimizer.param_groups → 优化器的参数组（通常只有一组）
            #   [0]['lr'] → 第一组的学习率
            #   :.2e → 科学计数法，保留 2 位小数（如 1.00e-03）

            f"{best_flag}"
        )

    # ---- 第 6 步：检查是否该停止 ----
    if early_stop.step(val_mse):
        # ↑ 更新早停器的状态，检查是否触发早停
        #   如果连续 40 轮 val_mse 没改善 → 返回 True

        print(f"\n连续 {PATIENCE} 个 epoch 验证损失无改善，训练终止于 Epoch {epoch}。")
        break
        # ↑ 跳出 for 循环，结束训练

print("=" * 130)


# ============================================================================
# 【输出最佳结果 & 加载最优模型】
# ============================================================================
#
# 训练结束后，我们要：
#   1. 打印最佳 epoch 的指标（确认训练效果）
#   2. 加载最优模型的权重（不是最后训练的那个，而是最好的那个）
#
# 为什么要加载最优模型？
#   训练结束时的模型可能是过拟合的（第 500 轮）
#   最优模型是验证集表现最好的那个（可能是第 200 轮）
#   我们要用最好的模型来预测测试集
#
# ============================================================================

print("\n" + "=" * 60)
print("【最佳模型性能】")
print("=" * 60)

bm = best_ckpt.best_metrics
# ↑ 获取最佳 epoch 的完整指标字典
#   bm = {"log_rmse": ..., "rmse": ..., "mae": ..., "r2": ...}

print(f"  最佳 Epoch    : {best_ckpt.best_epoch}")
# ↑ 打印最佳 epoch 编号（如 "最佳 Epoch: 187"）

print(f"  验证集 logRMSE: {bm['log_rmse']:.5f}")
# ↑ 打印 Kaggle 评分指标（越小越好）
#   好成绩通常在 0.10 ~ 0.15 之间

print(f"  验证集 R2     : {bm['r2']:.4f}")
# ↑ 打印决定系数（越接近 1 越好）
#   0.85 意味着模型解释了 85% 的房价变异

print(f"  验证集 RMSE   : ${bm['rmse']:,.0f}")
# ↑ 打印原始价格空间的均方根误差（单位：美元）
#   如 RMSE = $25,000 意味着平均偏差 2.5 万美元

print(f"  验证集 MAE    : ${bm['mae']:,.0f}")
# ↑ 打印平均绝对误差（单位：美元）

print("=" * 60)

# ---- 加载最优 checkpoint ----
best_model = NeuralNetwork(input_dim=input_dim).to(device)
# ↑ 创建一个新的模型实例（结构必须和保存时一样）

ckpt_info  = best_ckpt.load(best_model)
# ↑ 加载最优 checkpoint 的权重
#   best_model 的权重被替换为最佳 epoch 的权重
#   ckpt_info 是 checkpoint 字典，包含 epoch、best_loss 等信息

best_model.eval()
# ↑ 切换到评估模式（关闭 Dropout，准备做预测）

print(f"\n最佳模型已加载：Epoch {ckpt_info['epoch']}，Loss={ckpt_info['best_loss']:.6f}")


# ============================================================================
# 【生成提交文件】
# ============================================================================
#
# 最后一步：用最优模型预测测试集，生成 Kaggle 提交文件
#
# 提交文件格式：
#   Id,SalePrice
#   1461,169000.1
#   1462,187000.3
#   ...
#
# 上传到 Kaggle 后，Kaggle 会用 logRMSE 给你打分
#
# ============================================================================

print("\n" + "=" * 60)
print("【生成提交文件】")
print("=" * 60)

all_test_preds = []
# ↑ 存放所有测试集的预测结果

with torch.no_grad():
    # ↑ 关闭梯度（只需要前向传播，不需要反向传播）

    for X in test_loader:
        # ↑ 从测试集 DataLoader 取一个 batch
        #   注意：测试集没有标签，所以 X 直接就是特征

        pred = best_model(X.to(device)).squeeze(1).cpu().numpy()
        # ↑ 用最优模型做预测
        #
        #   X.to(device) → 搬到 GPU
        #   best_model(...) → 前向传播，得到预测值
        #   .squeeze(1) → 从 (64, 1) 压缩成 (64,)
        #   .cpu() → 搬回 CPU
        #   .numpy() → 转为 numpy 数组

        all_test_preds.append(pred)

all_test_preds = np.concatenate(all_test_preds)
# ↑ 把所有 batch 的预测结果拼接成一个大数组
#   形状：(1459,) → 1459 个预测值

all_test_preds = np.clip(np.expm1(all_test_preds), 0, None)
# ↑ 两步处理：
#
#   np.expm1(all_test_preds) → 从 log 空间还原回原始价格
#     expm1(x) = e^x - 1
#     例：expm1(12.247) = 208500
#
#   np.clip(..., 0, None) → 把负数截断为 0
#     None 表示没有上限
#     房价不可能是负数，如果模型预测出负数，就当 0 处理

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': all_test_preds})
# ↑ 创建提交 DataFrame
#   两列：Id（房子编号）和 SalePrice（预测房价）

submission.to_csv(SUBMISSION_PATH, index=False)
# ↑ 保存为 CSV 文件
#   index=False → 不保存行号（Kaggle 不需要）

print(f"提交文件已生成：{SUBMISSION_PATH}")
print(submission.head())
# ↑ 打印前 5 行，确认格式正确

print("\n训练流程完成。")
