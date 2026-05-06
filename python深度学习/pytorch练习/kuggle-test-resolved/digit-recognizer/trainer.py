
# ============================================================================
# trainer.py — 训练 & 验证循环模块（Digit Recognizer）
# ============================================================================
#
# 分类任务的训练循环：
#   损失函数：CrossEntropyLoss（交叉熵）
#   标签类型：long（整数类别索引，不是 one-hot）
#
# ============================================================================

import torch


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """训练一个 epoch。

    Args:
        dataloader: 训练集的 DataLoader
        model:      要训练的模型
        loss_fn:    损失函数（CrossEntropyLoss）
        optimizer:  优化器（Adam）
        device:     计算设备

    Returns:
        float: 这个 epoch 的平均训练 loss
    """

    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # ↑ y 的形状是 (batch_size,)，类型为 long
        #   CrossEntropyLoss 要求标签是 long 类型的类别索引

        pred = model(X)
        # ↑ pred 形状：(batch_size, 10)

        loss = loss_fn(pred, y)
        # ↑ CrossEntropyLoss 接受：
        #   pred: (batch_size, num_classes) — logits
        #   y:    (batch_size,) — 类别索引（0~9）

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def val_loop(dataloader, model, loss_fn, device):
    """验证一个 epoch。

    Args:
        dataloader: 验证集的 DataLoader
        model:      要验证的模型
        loss_fn:    损失函数
        device:     计算设备

    Returns:
        float: 这个 epoch 的平均验证 loss
    """

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)
