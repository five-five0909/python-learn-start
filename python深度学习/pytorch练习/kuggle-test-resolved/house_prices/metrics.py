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
# 例：有 2 个样本，误差分别是 1 万和 9 万
#   MSE  = mean([1², 9²]) = mean([1, 81]) = 41 → 大误差被平方放大
#   RMSE = sqrt(41) ≈ 6.4
#   MAE  = mean([1, 9]) = 5 → 对所有误差一视同仁
#
# ============================================================================

import numpy as np
# ↑ NumPy：科学计算库
#   np.sqrt()、np.mean()、np.abs()、np.concatenate() 等

import torch
# ↑ PyTorch 核心库
#   这里用于 torch.no_grad() 关闭梯度计算


def evaluate_metrics(dataloader, model, loss_fn, device):
    """在给定数据集上计算 3 个回归指标。

    流程：
        1. 切换到评估模式（model.eval()）
        2. 关闭梯度计算（torch.no_grad()）
        3. 遍历所有 batch，收集预测值和真实值
        4. 计算 MSE、RMSE、MAE

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
    #   本项目没有 Dropout，但养成好习惯很重要

    all_preds, all_labels = [], []
    # ↑ 存放所有预测值和真实值的列表
    #   每个 batch 的结果先 append 到列表，最后 concatenate 合并

    total_loss = 0.0
    # ↑ 累计的 MSE 总和

    with torch.no_grad():
        # ↑ 关闭梯度计算（评估时不需要反向传播）
        #   不关闭也行，但会白白浪费内存和计算资源
        #   torch.no_grad() 上下文管理器，里面的代码不会计算梯度

        for X, y in dataloader:
            # ↑ 从 DataLoader 取一个 batch 的数据
            #   X → 特征张量，形状 (batch_size, num_features)
            #   y → 标签张量，形状 (batch_size,)

            X, y = X.to(device), y.to(device)
            # ↑ 把数据搬到 GPU（如果可用）
            #   模型在 GPU 上，数据也必须在 GPU 上，否则报错

            y = y.unsqueeze(1)
            # ↑ 把 y 从 (batch_size,) 变成 (batch_size, 1)
            #   因为模型输出是 (batch_size, 1)，形状要匹配才能算 loss

            pred = model(X)
            # ↑ 前向传播：输入特征，得到预测值
            #   pred 形状：(batch_size, 1)

            loss = loss_fn(pred, y)
            # ↑ 计算当前 batch 的 MSE

            total_loss += loss.item() * X.size(0)
            # ↑ 累加当前 batch 的 MSE
            #   loss.item() → 从张量中取出 Python 数字
            #   * X.size(0) → 乘以 batch 中的样本数
            #
            #   为什么要乘以样本数？
            #     loss 返回的是"平均 MSE"（每个样本的平均）
            #     乘以样本数 = 这个 batch 的"总 MSE"
            #     最后除以总样本数 = 整个数据集的平均 MSE

            all_preds.append(pred.squeeze(1).cpu().numpy())
            # ↑ 保存当前 batch 的预测值
            #   pred.squeeze(1) → 把形状从 (32, 1) 压缩成 (32,)
            #   .cpu() → 从 GPU 搬回 CPU（numpy 只能在 CPU 上用）
            #   .numpy() → 从 PyTorch 张量转为 NumPy 数组

            all_labels.append(y.squeeze(1).cpu().numpy())
            # ↑ 保存当前 batch 的真实标签（处理方式同上）

    # ---- 拼接所有 batch 的结果 ----
    preds = np.concatenate(all_preds)
    # ↑ 把所有 batch 的预测值拼接成一个大数组
    #   例：[array([200000, 300000]), array([150000])] → array([200000, 300000, 150000])

    labels = np.concatenate(all_labels)
    # ↑ 把所有 batch 的真实标签拼接成一个大数组

    # ---- 计算 3 个指标 ----
    mse = total_loss / len(dataloader.dataset)
    # ↑ 平均 MSE = 总 MSE / 总样本数

    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    # ↑ RMSE = sqrt(MSE)
    #   单位是美元，例：RMSE = $25,000 意味着平均偏差 2.5 万美元

    mae = np.mean(np.abs(preds - labels))
    # ↑ MAE = 平均绝对误差
    #   和 RMSE 的区别：
    #     RMSE 对大误差更敏感（因为平方会放大大误差）
    #     MAE  对所有误差一视同仁

    return {"mse": mse, "rmse": rmse, "mae": mae}
    # ↑ 返回一个字典，包含所有 3 个指标
