# ============================================================================
# model.py — 模型定义模块
# ============================================================================
#
# 职责：定义神经网络结构
#
# 本项目使用一个 3 层全连接神经网络（MLP，多层感知机）
#
# 网络结构：
#   输入层：  ~79 个特征（LabelEncoder 编码后的特征数）
#   隐藏层1：256 个神经元 → ReLU
#   隐藏层2： 64 个神经元 → ReLU
#   输出层：   1 个神经元（预测房价）
#
# 为什么结构这么简单？
#   1460 条数据对深度学习来说太少了
#   复杂模型（512→256→128→64）参数太多，直接背答案（过拟合）
#   简单模型（256→64）被迫学规律，泛化更好
#
# 各组件的作用：
#   Linear → 全连接层：y = Wx + b，学习特征的线性组合
#   ReLU   → 激活函数：引入非线性，让网络能学复杂模式
#
# 为什么没有 BatchNorm 和 Dropout？
#   BatchNorm：小数据集上 batch 统计量不稳定，去掉反而更好
#   Dropout：简单模型本身就不容易过拟合，不需要额外正则化
#
# ============================================================================

import torch.nn as nn
# ↑ nn 是 PyTorch 的神经网络模块
#   包含：nn.Linear（全连接层）、nn.ReLU（激活函数）等


class NeuralNetwork(nn.Module):
    """3 层全连接神经网络（MLP）

    继承 nn.Module 是所有 PyTorch 模型的基类。
    必须实现 __init__（定义层）和 forward（定义前向传播）。

    Args:
        input_dim: 输入特征维度（LabelEncoder 编码后的特征数，约 79）

    网络结构：
        input_dim → 256 → 64 → 1

    使用示例：
        model = NeuralNetwork(input_dim=79)
        pred = model(x)          # 前向传播
        loss = loss_fn(pred, y)  # 计算损失
        loss.backward()          # 反向传播
    """

    def __init__(self, input_dim):
        """定义网络的各层。

        Args:
            input_dim: 输入特征维度
        """

        super().__init__()
        # ↑ 调用父类 nn.Module 的构造函数（必须写！）
        #   不写的话 PyTorch 无法正确管理模型的参数

        self.net = nn.Sequential(
            # ↑ nn.Sequential：按顺序串联各层，数据从第一层流到最后一层
            #   就像流水线：原料（特征）从一端进入，产品（预测值）从另一端出来

            # ---- 第一层：input_dim → 256 ----
            nn.Linear(input_dim, 256),
            # ↑ 全连接层：把 input_dim 维输入映射到 256 维
            #   内部参数：权重矩阵 W (256, input_dim) + 偏置向量 b (256,)
            #   计算：output = input @ W.T + b

            nn.ReLU(),
            # ↑ 激活函数：ReLU(x) = max(0, x)
            #   负数变 0，正数不变
            #   引入非线性，让网络能学复杂模式
            #   为什么叫 ReLU？Rectified Linear Unit（修正线性单元）

            # ---- 第二层：256 → 64 ----
            nn.Linear(256, 64),
            # ↑ 全连接层：256 维 → 64 维

            nn.ReLU(),
            # ↑ 激活函数（同上）

            # ---- 输出层：64 → 1 ----
            nn.Linear(64, 1)
            # ↑ 最终输出一个数字：预测的房价
            #   没有激活函数（因为我们要预测连续值，不需要压缩到某个范围）
        )

    def forward(self, x):
        """前向传播：定义数据如何流过网络。

        Args:
            x: 输入张量，形状 (batch_size, input_dim)
               例：(32, 79) 表示一个 batch 有 32 个样本，每个 79 个特征

        Returns:
            预测值张量，形状 (batch_size, 1)
            例：(32, 1) 表示 32 个预测的房价
        """
        return self.net(x)
        # ↑ 数据依次流过 self.net 中的每一层
        #   Linear → ReLU → Linear → ReLU → Linear
        #   最终输出预测的房价
