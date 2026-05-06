# ============================================================================
# checkpoint.py — 早停 & 最佳模型保存模块（Digit Recognizer）
# ============================================================================
#
# 分类任务中，我们监控验证集准确率（越大越好）
#   如果准确率连续 PATIENCE 个 epoch 没提升 → 停止训练
#
# ============================================================================

import torch


class EarlyStopping:
    """早停机制：当验证准确率连续 patience 个 epoch 没有提升时停止训练。

    Args:
        patience: 允许准确率不提升的最大 epoch 数
        path:     最佳模型权重的保存路径
    """

    def __init__(self, patience=10, path="best_model.pt"):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = 0.0
        # ↑ 历史最佳验证准确率（初始为 0）
        self.early_stop = False

    def __call__(self, val_accuracy, model):
        """每轮训练结束后调用。

        Args:
            val_accuracy: 当前验证准确率（0~1）
            model:        当前模型
        """

        if val_accuracy > self.best_score:
            # ↑ 准确率提升了
            self.best_score = val_accuracy
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    @property
    def best_accuracy(self):
        """返回最佳 epoch 的准确率。"""
        return self.best_score
