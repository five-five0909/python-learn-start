# ============================================================================
# main.py — Kaggle House Prices 主程序入口
# ============================================================================
#
# 使用方式（任选一种）：
#   方式1: 直接运行
#     python house-prices/main.py
#   方式2: 作为模块运行（在 house-prices 的父目录下）
#     python -m house-prices.main
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
# ============================================================================

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# ---- 从本包的其他模块导入 ----
# 兼容两种运行方式：
#   方式1: python main.py          （直接运行，用绝对导入）
#   方式2: python -m house-prices.main  （作为包运行，用相对导入）

try:
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
    exclude_cols=['Id'],       # Id 是编号，对预测没用
    fit_scaler=True,           # 训练集：学习数据的均值和标准差
    fit_encoders=True,         # 训练集：学习类别列的映射关系
)

test_set = HousePricesDataset(
    TEST_PATH,
    exclude_cols=['Id'],
    scaler=train_set.scaler,   # 测试集：复用训练集的 scaler（避免数据泄漏）
    encoders=train_set.encoders,  # 测试集：复用训练集的 encoders
)


# ============================================================================
# 2. 划分训练集 / 验证集 & 创建 DataLoader
# ============================================================================
# random_split 是 PyTorch 内置的，直接返回 Subset 对象，可以传给 DataLoader

train_size = int((1 - 0.2) * len(train_set))
val_size = len(train_set) - train_size

train_subset, val_subset = random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
# ↑ shuffle=True → 每个 epoch 打乱数据顺序，防止模型学到顺序规律

val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n数据划分：训练集 {train_size} 条，验证集 {val_size} 条")


# ============================================================================
# 3. 模型 & 损失函数 & 优化器
# ============================================================================
# 这三个组件是训练神经网络的核心：
#   模型（Model）      → 定义网络结构，学习特征到预测的映射
#   损失函数（Loss）   → 告诉模型"你错了多少"
#   优化器（Optimizer）→ 告诉模型"怎么改才能更准"

input_dim = len(train_set.feature_names)
# ↑ 获取特征数量，例：79 个特征 → input_dim = 79

model = NeuralNetwork(input_dim=input_dim).to(device)
# ↑ 创建模型实例，并搬到 GPU（如果可用）

print(f"模型已初始化，输入维度={input_dim}，运行设备={device}")

loss_fn = nn.MSELoss()
# ↑ 均方误差损失函数：MSE = mean((pred - y)²)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ↑ Adam 优化器：自适应学习率 + 动量，比 SGD 更稳定


# ============================================================================
# 4. 训练主循环
# ============================================================================
# 每个 epoch：训练 → 验证 → 打印状态 → 早停检查

print("\n" + "=" * 60)
print("【开始训练】")
print("=" * 60)

early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_PATH)
# ↑ 早停器：连续 PATIENCE 轮没改善就停止训练

for epoch in range(1, EPOCHS + 1):

    # ---- 训练 & 验证 ----
    train_mse = train_loop(train_loader, model, loss_fn, optimizer, device)
    val_mse = val_loop(val_loader, model, loss_fn, device)

    train_rmse = train_mse ** 0.5
    val_rmse = val_mse ** 0.5

    # ---- 每 10 轮打印一次状态 ----
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3} | 训练 RMSE: ${train_rmse:>10,.0f} | 验证 RMSE: ${val_rmse:>10,.0f}")

    # ---- 早停检查 ----
    early_stopping(val_mse, model)
    if early_stopping.early_stop:
        print(f"\n连续 {PATIENCE} 个 epoch 无改善，训练终止于 Epoch {epoch}。")
        print(f"最佳验证 RMSE: ${early_stopping.best_rmse:,.0f}")
        break

print(f"\n训练完成，最佳模型已保存至 {MODEL_PATH}")


# ============================================================================
# 5. 加载最佳模型 → 在验证集上评估
# ============================================================================
# 训练结束时的模型可能是过拟合的，所以要加载最优模型（验证集表现最好的那个）

print("\n" + "=" * 60)
print("【验证集评估】")
print("=" * 60)

best_model = NeuralNetwork(input_dim=input_dim).to(device)
# ↑ 创建一个新模型实例（结构必须和保存时一样）

best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# ↑ 加载最优 checkpoint 的权重

# ---- 用 evaluate_metrics 函数计算指标 ----
# 直接复用 metrics.py 里的函数，不用手写循环
results = evaluate_metrics(val_loader, best_model, loss_fn, device)

print(f"  MAE  = ${results['mae']:,.0f}")
# ↑ MAE = 平均绝对误差，例：$15,000 意味着平均偏差 1.5 万美元

print(f"  RMSE = ${results['rmse']:,.0f}")
# ↑ RMSE = 均方根误差


# ============================================================================
# 6. 生成 Kaggle 提交文件
# ============================================================================
# 用最优模型预测测试集，生成 Kaggle 提交文件
# 提交文件格式：Id,SalePrice

print("\n" + "=" * 60)
print("【生成提交文件】")
print("=" * 60)

best_model.eval()
all_preds = []

with torch.no_grad():
    for X in test_loader:
        # ↑ 测试集没有标签，只有 X
        pred = best_model(X.to(device)).squeeze(1).cpu()
        all_preds.append(pred)

submission = pd.read_csv(TEST_PATH)[['Id']]
# ↑ 从原始测试集 CSV 中读取 Id 列（保证 Id 和预测值对齐）

submission['SalePrice'] = torch.cat(all_preds).numpy()
# ↑ 把预测值添加到 submission DataFrame

submission.to_csv(SUBMISSION_PATH, index=False)
# ↑ 保存为 CSV 文件（index=False → 不保存行号）

print(f"提交文件已生成：{SUBMISSION_PATH}")
print(submission.head())
# ↑ 打印前 5 行，确认格式正确

print("\n训练流程完成。")
