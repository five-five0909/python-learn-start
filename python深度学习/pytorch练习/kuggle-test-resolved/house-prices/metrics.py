# ============================================================================
# metrics.py — 评估指标模块
# ============================================================================
#
# 职责：计算 3 个回归评估指标，衡量模型的好坏
#
# 3 个指标：
#   1. MSE（均方误差）→ 损失函数直接用的，越小越好
#   2. RMSE（均方根误差）→ 和原始价格同单位，更直观
#   3. MAE（平均绝对误差）→ 平均偏差多少美元，最容易理解
#
# MSE vs RMSE vs MAE 的区别：
#   MSE  = mean((预测值 - 真实值)²)        → 对大误差非常敏感（平方放大）
#   RMSE = sqrt(MSE)                       → 和原始价格同单位，更直观
#   MAE  = mean(|预测值 - 真实值|)          → 对所有误差一视同仁
#
# 例：预测 20 万，实际 25 万
#   MSE  = (5万)² = 25亿（单位是美元的平方，不好理解）
#   RMSE = 5万（单位是美元，好理解）
#   MAE  = 5万（单位是美元，一样）
#
# ============================================================================

import numpy as np
# ↑ NumPy：科学计算库
#   np.sqrt() 用于计算 RMSE

import torch
# ↑ PyTorch 核心库
#   用于 torch.no_grad() 关闭梯度计算

from sklearn.metrics import mean_squared_error, mean_absolute_error
# ↑ sklearn 内置的评估指标函数
#   mean_squared_error → 计算 MSE（均方误差）
#   mean_absolute_error → 计算 MAE（平均绝对误差）
#
#   为什么要用 sklearn 而不是自己算？
#     1. sklearn 的实现经过充分测试，不会有 bug
#     2. 代码更简洁，一行搞定
#     3. 和其他 sklearn 工具链兼容


def evaluate_metrics(dataloader, model, loss_fn, device):
    """在给定数据集上计算 3 个回归指标。

    流程：
        1. 切换到评估模式（model.eval()）
        2. 关闭梯度计算（torch.no_grad()）
        3. 遍历所有 batch，收集预测值和真实值
        4. 用 sklearn 计算 MSE、RMSE、MAE

    Args:
        dataloader: 要评估的数据 DataLoader（验证集或训练集）
        model:      要评估的模型
        loss_fn:    损失函数（MSELoss）
        device:     计算设备（"cuda" 或 "cpu"）

    Returns:
        dict: {
            "mse":  float,  # 均方误差，越小越好
            "rmse": float,  # 均方根误差，单位是美元
            "mae":  float,  # 平均绝对误差，单位是美元
        }
    """

    model.eval()
    # ↑ 把模型切换到"评估模式"
    #   训练模式 model.train() → Dropout 随机丢弃神经元
    #   评估模式 model.eval()  → Dropout 不丢弃

    all_preds, all_labels = [], []
    # ↑ 存放所有预测值和真实值的列表

    with torch.no_grad():
        # ↑ 关闭梯度计算（评估时不需要反向传播，节省内存）

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # ↑ 搬到 GPU（如果可用）

            pred = model(X)
            # ↑ 前向传播：输入特征，得到预测值

            all_preds.append(pred.squeeze(1).cpu().numpy())
            # ↑ 保存预测值
            #   squeeze(1) → 把 (32, 1) 压成 (32,)
            #   .cpu()     → 从 GPU 搬回 CPU
            #   .numpy()   → 转成 numpy 数组

            all_labels.append(y.cpu().numpy())
            # ↑ 保存真实标签

    # ---- 拼接所有 batch 的结果 ----
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    # ↑ 把所有 batch 的结果拼接成一个大数组

    # ---- 用 sklearn 计算 3 个指标 ----
    mse = mean_squared_error(labels, preds)
    # ↑ MSE = mean((preds - labels)²)
    #   sklearn 帮我们算，不用手写

    rmse = np.sqrt(mse)
    # ↑ RMSE = sqrt(MSE)
    #   单位是美元，例：RMSE = $25,000 意味着平均偏差 2.5 万美元

    mae = mean_absolute_error(labels, preds)
    # ↑ MAE = mean(|preds - labels|)
    #   对所有误差一视同仁

    return {"mse": mse, "rmse": rmse, "mae": mae}
    # ↑ 返回字典，包含所有 3 个指标
