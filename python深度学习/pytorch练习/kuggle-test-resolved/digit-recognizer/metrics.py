# ============================================================================
# metrics.py — 评估指标模块（Digit Recognizer）
# ============================================================================
#
# 分类任务的核心指标：准确率（Accuracy）
#   Accuracy = 预测正确的样本数 / 总样本数
#
# ============================================================================

import torch
from sklearn.metrics import accuracy_score
# ↑ sklearn 内置的准确率计算函数
#   比手写 correct / total 更简洁、更不容易出错


def evaluate_metrics(dataloader, model, device):
    """在给定数据集上计算分类准确率。

    Args:
        dataloader: 要评估的数据 DataLoader
        model:      要评估的模型
        device:     计算设备

    Returns:
        dict: {
            "accuracy": float,  # 准确率，0~1 之间
        }
    """

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            # ↑ pred 形状：(batch_size, 10)，每行是 10 个类别的 logits

            predicted = pred.argmax(dim=1)
            # ↑ 取概率最大的类别作为预测结果
            #   argmax(dim=1) → 沿类别维度取最大值的索引
            #   例：[0.1, 0.8, 0.05, ...] → 1（预测为数字 1）

            all_preds.append(predicted.cpu())
            all_labels.append(y.cpu())

    # ---- 用 sklearn 计算准确率 ----
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(labels, preds)
    # ↑ accuracy_score = 预测正确的数量 / 总数量
    #   比手写 correct / total 更简洁

    return {"accuracy": accuracy}
