# ============================================================================
# checkpoint.py — 早停 & 最佳模型保存模块
# ============================================================================
#
# 职责：
#   1. 判断训练是否应该提前终止（早停机制）
#   2. 自动保存最佳模型权重到文件
#
# 解决的问题：训练多少轮才够？
#
# 如果训练太少轮 → 欠拟合（模型没学会）
# 如果训练太多轮 → 过拟合（模型"背答案"，遇到新数据就不行了）
#
# EarlyStopping 的逻辑：
#   每轮训练后，检查验证集 loss 有没有下降
#   如果下降了 → 计数器归零，保存当前模型（新的最佳！）
#   如果没下降 → 计数器 +1
#   如果连续 PATIENCE（20）轮都没下降 → 停止训练
#
# 就像耐心一样：给模型 20 次机会，如果一直不改善，就不再等了
#
# 为什么把早停和模型保存合在一起？
#   源代码就是这样的设计：EarlyStopping 同时负责"什么时候停"和"保存最好的"
#   这样更简洁——早停器发现更好的模型时，顺手就保存了
#
# ============================================================================

import torch
# ↑ PyTorch 核心库
#   torch.save() → 保存模型权重到文件
#   torch.load() → 从文件加载模型权重


class EarlyStopping:
    """早停机制：当验证损失连续 patience 个 epoch 没有改善时停止训练。

    同时自动保存最佳模型权重到文件。

    工作流程：
        1. 每轮训练结束后，调用 __call__(val_loss, model)
        2. 如果 val_loss 比历史最佳更小 → 更新最佳，保存模型，计数器归零
        3. 如果 val_loss 没有改善 → 计数器 +1
        4. 如果计数器达到 patience → self.early_stop = True
        5. 外部检查 early_stop，如果为 True 就终止训练

    Args:
        patience: 允许验证损失不下降的最大 epoch 数（默认 20）
        path:     最佳模型权重的保存路径

    使用示例：
        early_stop = EarlyStopping(patience=20, path="model.pt")

        for epoch in range(500):
            train(...)
            val_loss = validate(...)
            early_stop(val_loss, model)       # 检查 + 自动保存
            if early_stop.early_stop:         # 如果触发早停
                print("训练终止！")
                break

        # 训练结束后，最佳模型已经在 early_stop.path 指定的文件里了
    """

    def __init__(self, patience=20, path="best_model.pt"):
        """初始化早停器。

        Args:
            patience: 最多容忍多少轮不改善
            path:     最佳模型保存路径
        """

        self.patience = patience
        # ↑ 耐心值：连续多少轮不改善就停止
        #   例：patience=20 → 连续 20 轮 val_loss 没下降就停

        self.path = path
        # ↑ 最佳模型保存到哪个文件

        self.counter = 0
        # ↑ 计数器：记录连续多少轮没有改善
        #   每次有改善 → 归零
        #   每次没改善 → +1
        #   达到 patience → 触发早停

        self.best_loss = float('inf')
        # ↑ 历史最佳验证损失
        #   初始化为正无穷大，第一轮一定会比它小

        self.early_stop = False
        # ↑ 是否触发了早停
        #   一旦设为 True，就一直保持 True

    def __call__(self, val_loss, model):
        """每轮训练结束后调用，检查是否该停止 + 自动保存最佳模型。

        为什么用 __call__ 而不是普通方法？
            __call__ 让对象可以像函数一样被调用：
            early_stop(val_loss, model)  而不是  early_stop.step(val_loss, model)
            更简洁，更 Pythonic

        Args:
            val_loss: 当前验证损失（MSE）
            model:    当前模型（PyTorch nn.Module）
        """

        if val_loss < self.best_loss:
            # ↑ 有改善：当前 loss 比历史最佳小
            self.best_loss = val_loss
            # ↑ 更新最佳 loss

            self.counter = 0
            # ↑ 计数器归零（重新开始计数）

            torch.save(model.state_dict(), self.path)
            # ↑ 保存当前模型的权重到文件
            #   model.state_dict() 返回一个字典：
            #   {
            #     "net.0.weight": tensor(...),   # 第一层 Linear 的权重
            #     "net.0.bias": tensor(...),     # 第一层 Linear 的偏置
            #     "net.2.weight": tensor(...),   # 第二层 Linear 的权重
            #     "net.2.bias": tensor(...),     # 第二层 Linear 的偏置
            #     "net.4.weight": tensor(...),   # 输出层 Linear 的权重
            #     "net.4.bias": tensor(...),     # 输出层 Linear 的偏置
            #   }
            #
            #   torch.save 用 Python 的 pickle 序列化
            #   文件大小通常几 MB（主要是模型权重）
        else:
            # ↑ 没改善
            self.counter += 1
            # ↑ 计数器 +1

            if self.counter >= self.patience:
                # ↑ 连续不改善的次数达到耐心上限
                self.early_stop = True
                # ↑ 触发早停

    @property
    def best_rmse(self):
        """返回最佳 epoch 的 RMSE。

        @property 让这个方法可以像属性一样访问：early_stop.best_rmse
        而不需要写 early_stop.best_rmse()

        best_loss 是 MSE（均方误差），RMSE = sqrt(MSE)
        """
        return self.best_loss ** 0.5
        # ↑ ** 0.5 就是开平方根
        #   例：MSE = 100 → RMSE = 10
