# ============================================================================
# model.py — 模型定义模块（Digit Recognizer）
# ============================================================================
#
# 网络结构：3 层全连接 MLP
#   输入层：784 个像素
#   隐藏层1：256 神经元 → ReLU
#   隐藏层2：128 神经元 → ReLU
#   输出层：10 个神经元（对应数字 0~9）
#
# 为什么输出层没有激活函数？
#   CrossEntropyLoss 内部已经包含了 Softmax
#   所以模型直接输出 logits（原始分数），不需要 softmax
#
# ============================================================================

import torch.nn as nn


class NeuralNetwork(nn.Module):
    """3 层全连接 MLP，用于手写数字分类。

    Args:
        input_dim:  输入维度（784）
        num_classes: 输出类别数（10）
    """

    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            # ↑ 输出 10 个 logits，不加激活函数
        )

    def forward(self, x):
        return self.net(x)
