# ============================================================================
# trainer.py — 训练 & 验证循环模块
# ============================================================================
#
# 职责：定义训练一个 epoch 和验证一个 epoch 的逻辑
#
# PyTorch 训练循环的标准写法（必须背下来！）：
#
#   训练模式（train_loop）：
#     1. model.train()          → 切换到训练模式
#     2. 前向传播               → 得到预测值
#     3. 计算 loss              → 得到误差
#     4. optimizer.zero_grad()  → 清空之前的梯度
#     5. loss.backward()        → 反向传播，计算梯度
#     6. optimizer.step()       → 用梯度更新参数
#
#   验证模式（val_loop）：
#     1. model.eval()           → 切换到评估模式
#     2. torch.no_grad()        → 关闭梯度计算
#     3. 前向传播               → 得到预测值
#     4. 计算 loss              → 得到误差（只记录，不更新参数）
#
# 训练 vs 验证 的关键区别：
#   训练：需要反向传播 + 参数更新（optimizer.step()）
#   验证：只需要前向传播，记录 loss（不更新参数）
#
# ============================================================================


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """训练一个 epoch：遍历全部训练数据一次，更新模型参数。

    每个 batch 的流程：
        1. 取数据 → 2. 前向传播 → 3. 算 loss → 4. 清梯度 → 5. 反向传播 → 6. 更新参数

    Args:
        dataloader: 训练集的 DataLoader
        model:      要训练的模型
        loss_fn:    损失函数（MSELoss）
        optimizer:  优化器（Adam）
        device:     计算设备（"cuda" 或 "cpu"）

    Returns:
        float: 这个 epoch 的平均训练 MSE
    """

    model.train()
    # ↑ 切换到训练模式
    #   如果有 Dropout → 会随机丢弃神经元
    #   如果有 BatchNorm → 会用当前 batch 的均值和方差

    total_loss = 0.0
    # ↑ 累计 loss（最后除以总样本数得到平均 loss）

    for X, y in dataloader:
        # ↑ 从 DataLoader 取一个 batch
        #   X → 特征，形状 (batch_size, num_features)，如 (32, 79)
        #   y → 标签，形状 (batch_size,)

        X, y = X.to(device), y.to(device)
        # ↑ 搬到 GPU（如果可用）
        #   模型在 GPU 上，数据也必须在 GPU 上，否则报错

        y = y.unsqueeze(1)
        # ↑ 把 y 从 (batch_size,) 变成 (batch_size, 1)
        #   因为模型输出是 (batch_size, 1)，形状要匹配才能算 loss
        #   unsqueeze(1) → 在第 1 维（列方向）增加一个维度
        #   例：(32,) → (32, 1)

        pred = model(X)
        # ↑ 前向传播：输入特征，得到预测值
        #   pred 形状：(batch_size, 1)

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


def val_loop(dataloader, model, loss_fn, device):
    """验证一个 epoch：遍历全部验证数据一次，不更新参数。

    和 train_loop 的区别：
        1. 用 model.eval() 而不是 model.train()
        2. 用 torch.no_grad() 关闭梯度计算
        3. 不调用 optimizer.step()（不更新参数）
        4. 不调用 optimizer.zero_grad()（不需要清空梯度）

    Args:
        dataloader: 验证集的 DataLoader
        model:      要验证的模型
        loss_fn:    损失函数（MSELoss）
        device:     计算设备（"cuda" 或 "cpu"）

    Returns:
        float: 这个 epoch 的平均验证 MSE
    """

    import torch
    # ↑ 局部导入 torch（用于 torch.no_grad()）
    #   放在函数内部而非文件顶部，是避免循环依赖的一种做法

    model.eval()
    # ↑ 切换到评估模式
    #   如果有 Dropout → 不丢弃（所有神经元参与）
    #   如果有 BatchNorm → 用全局统计量

    total_loss = 0.0

    with torch.no_grad():
        # ↑ 关闭梯度计算（节省内存和计算）
        #   验证时不需要反向传播，不需要计算梯度

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.unsqueeze(1)
            # ↑ 同 train_loop，把 y 变成 (batch_size, 1)

            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)
    # ↑ 返回整个 epoch 的平均验证 loss
