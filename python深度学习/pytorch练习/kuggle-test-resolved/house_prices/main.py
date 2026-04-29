# ============================================================================
# main.py — Kaggle House Prices 主程序入口
# ============================================================================
#
# 使用方式（任选一种）：
#   方式1: 直接运行
#     python house_prices/main.py
#   方式2: 作为模块运行（在 house_prices 的父目录下）
#     python -m house_prices.main
#
# 整体流程：
#   1. 加载数据        → HousePricesDataset 从 CSV 加载 + 自动特征工程
#   2. 划分数据集      → random_split 按 80/20 划分训练集和验证集
#   3. 创建 DataLoader → 自动分 batch + 打乱顺序
#   4. 创建模型组件    → 模型、损失函数、优化器
#   5. 训练主循环      → 训练 → 验证 → 打印状态 → 早停检查
#   6. 加载最优模型    → 从文件恢复最佳 epoch 的权重
#   7. 验证集评估      → 计算 MAE 和 RMSE
#   8. 生成提交文件    → 用最优模型预测测试集，输出 CSV
#
# 项目结构：
#   config.py         → 全局配置和超参数
#   dataset.py        → 自定义 Dataset 类（含特征工程）
#   model.py          → 神经网络结构
#   checkpoint.py     → 早停机制 + 最佳模型保存
#   metrics.py        → 评估指标计算
#   trainer.py        → 训练和验证循环
#   main.py           → 主程序（本文件）
#
# ============================================================================

import numpy as np
# ↑ NumPy：科学计算库
#   用于数组操作、数学计算等

import pandas as pd
# ↑ Pandas：数据分析库
#   用于读取 CSV、DataFrame 操作等

import torch
# ↑ PyTorch：深度学习框架
#   用于张量操作、模型训练等

from torch import nn
# ↑ nn 是 PyTorch 的神经网络模块
#   这里用到 nn.MSELoss（均方误差损失函数）

from torch.utils.data import DataLoader, random_split
# ↑ DataLoader   → 自动把数据分成小批次（batch），还能打乱顺序
#   random_split → 按比例随机划分数据集（如 80% 训练，20% 验证）

# ---- 从本包的其他模块导入 ----
# 兼容两种运行方式：
#   方式1: python main.py          （直接运行，用绝对导入）
#   方式2: python -m house_prices.main  （作为包运行，用相对导入）

try:
    # ↑ 先尝试相对导入（作为包运行时有效）
    from .config import (
        device, SEED, TRAIN_PATH, TEST_PATH, MODEL_PATH, SUBMISSION_PATH,
        BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE,
    )
    from .dataset import HousePricesDataset
    from .model import NeuralNetwork
    from .checkpoint import EarlyStopping
    from .metrics import evaluate_metrics
    from .trainer import train_loop, val_loop
except ImportError:
    # ↑ 相对导入失败（直接运行 main.py 时），改用绝对导入
    from config import (
        device, SEED, TRAIN_PATH, TEST_PATH, MODEL_PATH, SUBMISSION_PATH,
        BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE,
    )
    from dataset import HousePricesDataset
    from model import NeuralNetwork
    from checkpoint import EarlyStopping
    from metrics import evaluate_metrics
    from trainer import train_loop, val_loop


# ============================================================================
# 1. 加载数据
# ============================================================================
# HousePricesDataset 会自动完成：
#   读取 CSV → 分离标签 → 排除 Id → 填充缺失值 → LabelEncoder → 标准化

print("=" * 60)
print("【加载数据】")
print("=" * 60)

train_set = HousePricesDataset(
    TRAIN_PATH,
    exclude_cols=['Id'],
    # ↑ 排除 Id 列（编号对预测没用）

    fit_scaler=True,
    # ↑ 训练集：学习数据的均值和标准差（用于标准化）

    fit_encoders=True,
    # ↑ 训练集：学习类别列的映射关系（用于 LabelEncoder）
)
test_set = HousePricesDataset(
    TEST_PATH,
    exclude_cols=['Id'],

    scaler=train_set.scaler,
    # ↑ 测试集：复用训练集的 scaler
    #   不能重新 fit！否则用了测试集的信息 → 数据泄漏

    encoders=train_set.encoders,
    # ↑ 测试集：复用训练集的 encoders
    #   保证同一个类别在训练集和测试集中被编码成同一个数字
)


# ============================================================================
# 2. 划分训练集 / 验证集 & 创建 DataLoader
# ============================================================================
#
# 为什么用 random_split 而不是 train_test_split？
#   random_split 是 PyTorch 内置的，直接返回 Subset 对象
#   Subset 可以直接传给 DataLoader，更方便
#   缺点是纯随机划分，没有分层功能
#
# ============================================================================

train_size = int((1 - 0.2) * len(train_set))
# ↑ 训练集大小 = 总数 × 80%
#   例：1460 × 0.8 = 1168

val_size = len(train_set) - train_size
# ↑ 验证集大小 = 总数 - 训练集大小
#   例：1460 - 1168 = 292

train_subset, val_subset = random_split(train_set, [train_size, val_size])
# ↑ 按比例随机划分
#   train_subset → 训练集子集（1168 条）
#   val_subset   → 验证集子集（292 条）

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
# ↑ 训练集的 DataLoader
#   batch_size=32 → 每次送 32 条数据进模型
#   shuffle=True  → 每个 epoch 开始时打乱数据顺序
#   打乱顺序的作用：防止模型学到数据的顺序规律

val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
# ↑ 验证集的 DataLoader
#   shuffle=False → 验证时不需要打乱，顺序无所谓

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
# ↑ 测试集的 DataLoader
#   shuffle=False → 测试时不需要打乱

print(f"\n数据划分：训练集 {train_size} 条，验证集 {val_size} 条")


# ============================================================================
# 3. 模型 & 损失函数 & 优化器
# ============================================================================
#
# 这三个组件是训练神经网络的核心：
#
#   模型（Model）      → 定义网络结构，学习特征到预测的映射
#   损失函数（Loss）   → 告诉模型"你错了多少"
#   优化器（Optimizer）→ 告诉模型"怎么改才能更准"
#
# ============================================================================

input_dim = len(train_set.feature_names)
# ↑ 获取特征数量
#   train_set.feature_names 是 HousePricesDataset 保存的特征名列表
#   例：79 个特征 → input_dim = 79

model = NeuralNetwork(input_dim=input_dim).to(device)
# ↑ 创建模型实例，并搬到 GPU（如果可用）
#   .to(device) → 把模型的参数搬到指定设备上

print(f"模型已初始化，输入维度={input_dim}，运行设备={device}")

loss_fn = nn.MSELoss()
# ↑ 均方误差损失函数
#   MSE = mean((pred - y)²)
#   回归任务的标准损失函数

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ↑ Adam 优化器
#   model.parameters() → 获取模型的所有可训练参数（权重和偏置）
#   lr=1e-3            → 学习率（从 config 导入）
#
#   Adam 是最常用的优化器：
#     自适应学习率：每个参数有自己的学习率
#     动量：记住历史梯度方向，加速收敛
#     比 SGD 更稳定，通常不需要手动调学习率


# ============================================================================
# 4. 训练主循环
# ============================================================================
#
# 这是整个程序的核心！把前面所有组件串起来：
#
#   for epoch in range(500):
#       1. 训练模型（train_loop）
#       2. 验证模型（val_loop）
#       3. 打印训练状态（每 10 轮）
#       4. 检查是否该停止（early_stopping）
#
# ============================================================================

print("\n" + "=" * 60)
print("【开始训练】")
print("=" * 60)

early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_PATH)
# ↑ 创建早停器
#   patience=20 → 连续 20 轮没改善就停
#   path=MODEL_PATH → 最佳模型保存到这个文件

for epoch in range(1, EPOCHS + 1):
    # ↑ 主循环：从第 1 轮到第 500 轮
    #   range(1, 501) → 1, 2, 3, ..., 500

    # ---- 第 1 步：训练 ----
    train_mse = train_loop(train_loader, model, loss_fn, optimizer, device)
    # ↑ 训练一个 epoch，返回训练集的平均 MSE
    #   训练过程中模型权重会被更新

    # ---- 第 2 步：验证 ----
    val_mse = val_loop(val_loader, model, loss_fn, device)
    # ↑ 验证一个 epoch，返回验证集的平均 MSE
    #   验证过程中模型权重不会被更新

    # ---- 计算 RMSE（方便打印） ----
    train_rmse = train_mse ** 0.5
    val_rmse = val_mse ** 0.5

    # ---- 第 3 步：打印训练状态（每 10 轮 + 第 1 轮） ----
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3} | 训练 RMSE: ${train_rmse:>10,.0f} | 验证 RMSE: ${val_rmse:>10,.0f}")
        # ↑ 格式化输出
        #   {epoch:>3}     → 右对齐，宽度 3
        #   ${train_rmse:>10,.0f} → 右对齐，宽度 10，千分位分隔，无小数

    # ---- 第 4 步：检查是否该停止 ----
    early_stopping(val_mse, model)
    # ↑ 更新早停器的状态
    #   如果 val_mse 是新的最佳 → 保存模型，计数器归零
    #   如果 val_mse 没改善 → 计数器 +1

    if early_stopping.early_stop:
        # ↑ 如果触发了早停
        print(f"\n连续 {PATIENCE} 个 epoch 无改善，训练终止于 Epoch {epoch}。")
        print(f"最佳验证 RMSE: ${early_stopping.best_rmse:,.0f}")
        break
        # ↑ 跳出 for 循环，结束训练

print(f"\n训练完成，最佳模型已保存至 {MODEL_PATH}")


# ============================================================================
# 5. 加载最佳模型 → 在验证集上评估
# ============================================================================
#
# 训练结束后，我们要：
#   1. 加载最优模型的权重（不是最后训练的那个，而是最好的那个）
#   2. 在验证集上评估，计算 MAE 和 RMSE
#
# 为什么要加载最优模型？
#   训练结束时的模型可能是过拟合的（第 500 轮）
#   最优模型是验证集表现最好的那个（可能是第 187 轮）
#   我们要用最好的模型来预测测试集
#
# ============================================================================

print("\n" + "=" * 60)
print("【验证集评估】")
print("=" * 60)

best_model = NeuralNetwork(input_dim=input_dim).to(device)
# ↑ 创建一个新的模型实例（结构必须和保存时一样）

best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# ↑ 加载最优 checkpoint 的权重
#   torch.load(MODEL_PATH) → 从文件加载 state_dict 字典
#   map_location=device    → 把张量加载到指定设备（CPU 或 GPU）
#     如果模型是在 GPU 上保存的，但当前没有 GPU，这行确保不会报错
#   load_state_dict(...)   → 把权重加载到模型中
#     模型结构必须和保存时完全一致，否则报错

best_model.eval()
# ↑ 切换到评估模式（关闭 Dropout，准备做预测）

all_preds, all_labels = [], []
# ↑ 存放所有预测值和真实值

with torch.no_grad():
    for X, y in val_loader:
        pred = best_model(X.to(device)).squeeze(1).cpu()
        # ↑ 前向传播 + 压缩维度 + 搬回 CPU
        #   squeeze(1) → (32, 1) → (32,)

        all_preds.append(pred)
        all_labels.append(y)

all_preds = torch.cat(all_preds)
# ↑ 把所有 batch 的预测值拼接成一个大张量

all_labels = torch.cat(all_labels)
# ↑ 把所有 batch 的真实标签拼接成一个大张量

mae = (all_preds - all_labels).abs().mean().item()
# ↑ MAE = 平均绝对误差
#   (preds - labels) → 每个样本的误差
#   .abs()           → 取绝对值（消除正负号）
#   .mean()          → 取平均
#   .item()          → 从张量取出 Python 数字

rmse = ((all_preds - all_labels) ** 2).mean().sqrt().item()
# ↑ RMSE = 均方根误差
#   (... ** 2) → 误差的平方
#   .mean()    → 取平均（得到 MSE）
#   .sqrt()    → 开平方根（得到 RMSE）

print(f"  MAE  = ${mae:,.0f}")
# ↑ 打印 MAE（单位：美元）
#   例：MAE = $15,000 意味着平均偏差 1.5 万美元

print(f"  RMSE = ${rmse:,.0f}")
# ↑ 打印 RMSE（单位：美元）


# ============================================================================
# 6. 生成 Kaggle 提交文件
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
# 上传到 Kaggle 后，Kaggle 会用 RMSE 给你打分
#
# ============================================================================

print("\n" + "=" * 60)
print("【生成提交文件】")
print("=" * 60)

all_preds = []
# ↑ 存放测试集的预测结果

with torch.no_grad():
    for X in test_loader:
        # ↑ 测试集没有标签，只有 X
        #   DataLoader 返回的是单个张量，不是 (X, y) 元组

        pred = best_model(X.to(device)).squeeze(1).cpu()
        # ↑ 前向传播 + 压缩维度 + 搬回 CPU

        all_preds.append(pred)

submission = pd.read_csv(TEST_PATH)[['Id']]
# ↑ 从原始测试集 CSV 中读取 Id 列
#   为什么重新读 CSV？因为 Dataset 可能打乱了顺序
#   用原始 CSV 保证 Id 和预测值对齐

submission['SalePrice'] = torch.cat(all_preds).numpy()
# ↑ 把预测值添加到 submission DataFrame
#   torch.cat(all_preds) → 拼接所有 batch 的预测
#   .numpy() → 转成 numpy 数组（Pandas 需要）

submission.to_csv(SUBMISSION_PATH, index=False)
# ↑ 保存为 CSV 文件
#   index=False → 不保存行号（Kaggle 不需要）

print(f"提交文件已生成：{SUBMISSION_PATH}")
print(submission.head())
# ↑ 打印前 5 行，确认格式正确

print("\n训练流程完成。")
