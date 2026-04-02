# ============================================================
# 手动实现神经网络训练（纯函数版，没有class，没有面向对象）
# 安装：pip install tensorflow numpy
# ============================================================
#
# 整个代码就做一件事：
#   拿60000张手写数字图片训练一个两层神经网络，让它学会认数字。
#
# 流程：
#   创建参数 → 加载数据 → 循环训练（前向传播→算loss→反向传播→更新参数）→ 测试
#
# ============================================================

import math
import numpy as np
import tensorflow as tf


# ============================================================
# 第一步：创建所有参数（就是那407,050个数字）
# ============================================================
#
# 整个网络只有4个需要学习的东西：W1, b1, W2, b2
# 训练开始前全是随机值/零，训练就是不断调整它们
#
# 为什么用 tf.Variable？
#   普通张量是固定常量，Variable 是"可修改的变量"。
#   后面反向传播要调整 W 和 b，必须标记成 Variable，
#   否则 GradientTape 算不到它的梯度，也就没法更新。

# ── 第一层的参数 ──────────────────────────────────────────
#
# W1 — 第一层权重矩阵
#   形状：(784, 512)
#   含义：784行 = 每个像素对应一行
#         512列 = 512个神经元，每列是一个神经元的全部权重
#   总数：784 × 512 = 401,408 个权重
#   初始化：0到0.1之间的随机小数（随机是为了打破对称性，
#           如果全一样，512个神经元学出来的东西也一样，等于白叠）
W1 = tf.Variable(tf.random.uniform((784, 512), minval=0, maxval=0.1))

# b1 — 第一层偏置向量
#   形状：(512,)
#   含义：512个神经元，每人一个偏置
#   初始化：全零（偏置不参与对称性问题，零起步没关系）
b1 = tf.Variable(tf.zeros((512,)))

# ── 第二层的参数 ──────────────────────────────────────────
#
# W2 — 第二层权重矩阵
#   形状：(512, 10)
#   含义：512行 = 上一层传来的512个特征
#         10列 = 10个输出神经元，分别对应数字0到9
#   总数：512 × 10 = 5,120 个权重
W2 = tf.Variable(tf.random.uniform((512, 10), minval=0, maxval=0.1))

# b2 — 第二层偏置向量
#   形状：(10,)
#   含义：10个类别，每人一个偏置
b2 = tf.Variable(tf.zeros((10,)))

# 把4个参数放进一个列表，方便后面统一算梯度、统一更新
# all_weights[0] = W1 (784,512)
# all_weights[1] = b1 (512,)
# all_weights[2] = W2 (512,10)
# all_weights[3] = b2 (10,)
all_weights = [W1, b1, W2, b2]

# 验证总参数量：401,408 + 512 + 5,120 + 10 = 407,050
total = sum(w.numpy().size for w in all_weights)
print(f"总参数量：{total}")


# ============================================================
# 第二步：定义前向传播（一张图从进去到出概率）
# ============================================================
#
# 就是笔记里的那条链：
#   x (784,)
#     → z1 = x·W1 + b1       (512,)   ← 784个数做512次点积
#     → h  = ReLU(z1)          (512,)   ← 负数归零
#     → z2 = h·W2 + b2        (10,)    ← 512个数做10次点积
#     → p  = Softmax(z2)       (10,)    ← 压成概率
#
# 实际训练时一次喂128张图，所以 x 的形状是 (128, 784)，
# 但公式完全一样，只是每一行各自独立算。

def forward(x):
    """
    前向传播：输入图片像素，输出10个类别的概率。

    参数
    ----
    x : tf.Tensor, 形状 (batch_size, 784)
        一批图片的像素值，每行784个数（28×28展平后）。
        batch_size 通常是128（训练时）或10000（测试时一次性喂完）。
        值域 0~1（已经除以255归一化过）。

    返回
    ----
    p : tf.Tensor, 形状 (batch_size, 10)
        每行是一张图片对应数字0~9的概率，10个数加起来=1。
        比如某行 [0.01, 0.02, 0.91, ...] 表示模型认为是数字2的概率91%。

    内部计算过程
    ----------
    第一层：z1 = x·W1 + b1        → (batch_size, 512)   矩阵乘法+偏置
            h  = ReLU(z1)          → (batch_size, 512)   负数归零
    第二层：z2 = h·W2 + b2         → (batch_size, 10)    矩阵乘法+偏置
            p  = Softmax(z2)       → (batch_size, 10)    变成概率分布
    """
    # 第一层：线性变换 + ReLU ─────────────────────────────
    # tf.matmul = 矩阵乘法 = 笔记里的 dot(x, W1)
    # (128, 784) × (784, 512) → (128, 512)
    # 128张图同时算，每张图的784个像素和W1的512列分别做点积
    z1 = tf.matmul(x, W1) + b1

    # ReLU：负数归零，正数保留
    # 为什么要这一步？没有ReLU → 不管叠多少层都等于一层线性变换
    # 有了ReLU → 引入非线性 → 网络才能学到复杂规律
    h = tf.nn.relu(z1)                # (128, 512)

    # 第二层：线性变换 + Softmax ──────────────────────────
    # (128, 512) × (512, 10) → (128, 10)
    z2 = tf.matmul(h, W2) + b2

    # Softmax：每行10个数变成概率分布，加起来=1
    # 公式：p[j] = e^z[j] / (e^z[0] + e^z[1] + ... + e^z[9])
    # 先取 e 的次方（保证全是正数），再除以总和（保证加起来=1）
    p = tf.nn.softmax(z2)            # (128, 10)

    return p


# ============================================================
# 第三步：定义loss函数（交叉熵：猜得多离谱）
# ============================================================
#
# 公式：loss = -log(正确类别的概率)
#
# 猜对了（概率接近1）→ -log(1)    = 0       → loss最小
# 半信半疑（概率0.5）→ -log(0.5)  = 0.69    → loss中等
# 猜错了（概率0.01）→ -log(0.01) = 4.61    → loss很大
# 越错罚越狠，逼着模型把正确类别的概率推上去

def compute_loss(predictions, labels):
    """
    计算一批图片的平均交叉熵损失。

    参数
    ----
    predictions : tf.Tensor, 形状 (batch_size, 10)
        前向传播输出的概率矩阵。
        每一行是一张图片对10个数字的预测概率，加起来=1。
        例：[[0.02, 0.01, 0.85, 0.01, ...],   ← 第0张图
             [0.90, 0.01, 0.01, 0.01, ...],   ← 第1张图
             ...]

    labels : numpy.ndarray, 形状 (batch_size,)
        这一批图片的正确答案，每个元素是0~9的整数。
        例：[5, 0, 4, 1, ...]
        表示第0张图是5，第1张图是0，第2张图是4……

    返回
    ----
    average_loss : tf.Tensor, 标量（一个数）
        128张图各自的loss取平均。
        训练过程中这个数应该越来越小（从~2.5降到~0.2）。

    内部计算过程
    ----------
    1. 对每张图：loss_i = -log(predictions_i[labels_i])
       比如第0张图标签=5，predictions_0[5]=0.85
       → loss_0 = -log(0.85) = 0.16

    2. per_sample = [loss_0, loss_1, ..., loss_127]，形状 (128,)

    3. average_loss = (loss_0 + loss_1 + ... + loss_127) / 128
    """
    # sparse_categorical_crossentropy：
    #   sparse = 标签是整数（5），不是one-hot（[0,0,0,0,0,1,0,0,0,0]）
    #   categorical = 分类任务
    #   crossentropy = 交叉熵
    per_sample = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    # per_sample 形状 (batch_size,)，每张图一个loss值

    # tf.reduce_mean：取平均，(128,) → 标量
    return tf.reduce_mean(per_sample)


# ============================================================
# 第四步：定义单次训练步骤（整个训练的核心）
# ============================================================
#
# 每调用一次就做4件事：
#   ① 前向传播：x → p（算出预测概率）
#   ② 算 loss：对比 p 和标签，看猜得多离谱
#   ③ 反向传播：从 loss 出发，用链式法则算出407,050个偏导数
#   ④ 更新参数：每个参数沿着让loss变小的方向微调一步

# 学习率：控制每步迈多远
#   太大（1.0）    → 一脚跳过最优解，loss来回震荡，训练崩溃
#   太小（0.000001）→ 每次只动一丢丢，训练要几百万次
#   0.001          → 经验上安全的起步值
learning_rate = 0.001

def train_one_step(images_batch, labels_batch):
    """
    执行一次完整的训练步骤：前向 → loss → 反向 → 更新。

    参数
    ----
    images_batch : numpy.ndarray, 形状 (batch_size, 784)
        一批图片的像素值（已归一化到0~1）。
        batch_size 通常是128。
        例：第一行 [0.0, 1.0, 0.71, 0.0, 0.78, ...]
            就是一张图的784个像素值

    labels_batch : numpy.ndarray, 形状 (batch_size,)
        这一批图片的正确答案，每个元素是0~9的整数。
        例：[5, 0, 4, 1, 9, 2, ...]

    返回
    ----
    loss : tf.Tensor, 标量
        这一批的平均loss。训练过程中这个数应该越来越小。

    副作用
    -----
    直接修改全局变量 W1, b1, W2, b2 的值（原地更新）。
    每调用一次，407,050个参数各自微调一点点。
    """
    # ── GradientTape：TensorFlow 的"录像机" ──────────────
    #
    # 工作原理：
    #   1. 在 with 块里做前向传播和算loss，tape 默默把每一步运算记下来
    #   2. 调 tape.gradient(loss, all_weights) 时，tape 倒带回放，
    #      用链式法则从 loss 出发反向推导每个参数的偏导数
    #   3. 你不用手写任何求导公式，TF 全部自动搞定
    #
    # 这就是深度学习框架最大的价值：
    #   你只管定义"怎么算"（前向传播），框架帮你算"怎么调"（梯度）
    with tf.GradientTape() as tape:

        # ① 前向传播 ──────────────────────────────────────
        predictions = forward(images_batch)
        # predictions: (128, 10)
        # 每行是一张图片的10个预测概率

        # ② 算 loss ──────────────────────────────────────
        loss = compute_loss(predictions, labels_batch)
        # loss: 一个数，比如 2.3026

    # ③ 反向传播 ──────────────────────────────────────────
    #
    # tape.gradient(loss, all_weights) 返回一个列表，和 all_weights 一一对应：
    #
    #   gradients[0] = ∂loss/∂W1  形状 (784, 512)   401,408个偏导
    #   gradients[1] = ∂loss/∂b1  形状 (512,)            512个偏导
    #   gradients[2] = ∂loss/∂W2  形状 (512, 10)       5,120个偏导
    #   gradients[3] = ∂loss/∂b2  形状 (10,)              10个偏导
    #                                          总计 407,050个偏导数
    #
    # 每个偏导数告诉你：
    #   正数 → 增大这个参数会让loss变大 → 应该减小它
    #   负数 → 增大这个参数会让loss变小 → 应该增大它
    #   接近零 → 这个参数对loss影响不大 → 几乎不动
    #
    # TF 内部用的就是笔记里推导的那些链式法则公式：
    #   ∂loss/∂z2 = p - y                        (Softmax+交叉熵的简洁结果)
    #   ∂loss/∂W2 = h^T · (p - y)                (第二层权重的梯度)
    #   ∂loss/∂b2 = sum(p - y)                    (第二层偏置的梯度)
    #   ∂loss/∂h  = (p - y) · W2^T                (梯度穿过W2传回隐藏层)
    #   ∂loss/∂z1 = ∂loss/∂h ⊙ ReLU'(z1)         (梯度穿过ReLU，死神经元挡住)
    #   ∂loss/∂W1 = x^T · ∂loss/∂z1              (第一层权重的梯度)
    #   ∂loss/∂b1 = sum(∂loss/∂z1)                (第一层偏置的梯度)
    gradients = tape.gradient(loss, all_weights)

    # ④ 更新参数 ──────────────────────────────────────────
    #
    # 公式：w_new = w_old - learning_rate × ∂loss/∂w
    #
    # 为什么减？
    #   梯度指向 loss 增大的方向
    #   我们要让 loss 减小
    #   所以往反方向走 → 减去梯度
    #
    # 展开来看：
    #   W1 = W1 - 0.001 × ∂loss/∂W1   （401,408个权重各调一点）
    #   b1 = b1 - 0.001 × ∂loss/∂b1   （    512个偏置各调一点）
    #   W2 = W2 - 0.001 × ∂loss/∂W2   （  5,120个权重各调一点）
    #   b2 = b2 - 0.001 × ∂loss/∂b2   （     10个偏置各调一点）
    #
    # assign_sub(x) = "自己减去x"，等价于 w = w - x
    for i in range(len(all_weights)):
        all_weights[i].assign_sub(learning_rate * gradients[i])

    return loss


# ============================================================
# 第五步：加载并预处理数据
# ============================================================

# load_data() 返回两组数据：
#   训练集：60000张图 + 60000个标签（用来学习，相当于练习题）
#   测试集：10000张图 + 10000个标签（用来考试，训练时绝不给模型看）
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)
# train_images 原始形状：(60000, 28, 28)，60000张28×28像素的灰度图
# train_labels 原始形状：(60000,)，60000个整数标签（0~9）
# test_images  原始形状：(10000, 28, 28)
# test_labels  原始形状：(10000,)

# ── reshape：展平图片 ────────────────────────────────────
# (60000, 28, 28) → (60000, 784)
# 每张图从28×28的格子变成784个数字排成一行
# Dense 层只吃一维向量，不吃二维图片
train_images = train_images.reshape((60000, 784))
test_images  = test_images.reshape((10000, 784))

# ── astype("float32") ────────────────────────────────────
# 原始像素是 uint8 整数类型（0~255）
# 矩阵乘法和梯度计算需要浮点数，所以转成 float32
train_images = train_images.astype("float32")
test_images  = test_images.astype("float32")

# ── 归一化：除以255 ──────────────────────────────────────
# 像素值从 0~255 缩到 0~1
# 为什么？
#   像素=200，乘权重0.1 → 结果=20，再乘下一层 → 数字越来越大 → 梯度爆炸
#   像素=0.78，乘权重0.1 → 结果=0.078，温和范围，训练稳定
train_images = train_images / 255.0
test_images  = test_images / 255.0

print(f"训练集：{train_images.shape[0]} 张，每张 {train_images.shape[1]} 个像素")
print(f"测试集：{test_images.shape[0]} 张")
print(f"标签示例（前10个）：{train_labels[:10]}")


# ============================================================
# 第六步：训练！
# ============================================================
#
# 两层循环：
#
#   外层 for epoch：
#     一个 epoch = 把60000张训练图全部看完一遍
#     epochs=10 → 看10遍
#
#   内层 for batch：
#     每次从60000张里取128张做一次训练
#     60000 ÷ 128 = 468.75 → 向上取整 = 469批
#
# 总共：10 epoch × 469 batch = 4,690 次参数更新
# 每次更新：407,050个参数各调一点点
# 4,690次之后：loss 从 ~2.5 降到 ~0.2，准确率从 ~10% 升到 ~96%

# ── 超参数 ────────────────────────────────────────────────
# epochs（轮次）：训练几轮
#   太少 → 没学够
#   太多 → 可能过拟合（把练习题答案背下来了，但考试碰到新题不会做）
#   10 → 对这个小任务足够
epochs = 10

# batch_size（批大小）：每次喂多少张图
#   太小（比如1）  → 每次只看1张就调参数，方向不稳，loss来回跳
#   太大（比如60000）→ 一次看完全部，内存装不下，而且更新太慢
#   128            → 经典折中值，够稳定，不太慢，内存也装得下
batch_size = 128

# num_batches（批数）：60000张 ÷ 128张/批 = 469批（向上取整）
num_batches = math.ceil(len(train_images) / batch_size)

for epoch in range(epochs):
    print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

    # 每个 epoch 重置指针，从第0张图重新开始
    index = 0

    for batch in range(num_batches):
        # ── 手动切 batch ─────────────────────────────────
        # 从 index 位置往后切 batch_size 张图和对应的标签
        # Python 切片不会越界，最后一批不足128张也没关系
        #
        # 第  1 次：images[0:128]，      index → 128
        # 第  2 次：images[128:256]，    index → 256
        # 第  3 次：images[256:384]，    index → 384
        # ...
        # 第469次：images[59904:60000]，最后96张
        images_batch = train_images[index : index + batch_size]
        labels_batch = train_labels[index : index + batch_size]
        index += batch_size

        # 做一次完整训练：① 前向 → ② loss → ③ 反向 → ④ 更新
        loss = train_one_step(images_batch, labels_batch)

        # 每100批打印一次，看 loss 在不在下降
        if batch % 100 == 0:
            print(f"  第 {batch:3d}/{num_batches} 批，loss = {loss:.4f}")

    print(f"  本轮结束，loss = {loss:.4f}")


# ============================================================
# 第七步：测试（用从没见过的10000张图考试）
# ============================================================
#
# 训练时模型从来没看过测试集。
# 现在喂进去看它能猜对多少——这是对模型真实水平的检验。
# 这一步只做前向传播，不做反向传播，不更新参数——纯考试。

# 一次性喂入10000张测试图
# （不需要分批，因为只做前向不做反向，不需要存中间变量，内存压力小很多）
predictions = forward(test_images).numpy()
# predictions: (10000, 10)
# 每一行是一张图对10个数字的概率
# 例：predictions[0] = [0.01, 0.01, 0.02, 0.90, 0.01, ...] → 模型觉得是数字3

# ── argmax：取概率最大的位置作为预测结果 ─────────────────
# axis=1 表示在每一行内部找最大值（不是在10000行之间找）
# 比如某行 [0.01, 0.02, 0.91, 0.01, ...] → 第2个位置最大 → 预测是数字2
predicted_labels = np.argmax(predictions, axis=1)
# predicted_labels: (10000,)
# 例：[7, 2, 1, 0, 4, 1, 4, 9, 5, 9, ...]

# ── 计算准确率 ───────────────────────────────────────────
# predicted_labels == test_labels → 逐个对比，返回布尔数组
# 例：[True, True, False, True, True, ...]
# True 算 1，False 算 0，取平均 = 准确率
matches = (predicted_labels == test_labels)   # (10000,) 布尔数组
accuracy = np.mean(matches)                   # 例：0.9612

print(f"\n===== 测试结果 =====")
print(f"10000 张图里猜对了 {int(matches.sum())} 张")
print(f"准确率：{accuracy:.4f}")
# 预期 95%~97%（SGD 优化器的水平）
# 换成 Adam 优化器，同样的网络能到 98%
# 不是网络变了，是"调参策略"变聪明了


# ============================================================
# 附录：等价的 Keras 写法（上面全部代码 = 下面这几行）
# ============================================================
#
# from tensorflow.keras import models, layers
#
# network = models.Sequential([
#     layers.Dense(512, activation='relu', input_shape=(784,)),
#     layers.Dense(10,  activation='softmax'),
# ])
# network.compile(
#     optimizer='sgd',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy'],
# )
# network.fit(train_images, train_labels, epochs=10, batch_size=128)
# test_loss, test_acc = network.evaluate(test_images, test_labels)
#
# 就这几行。上面我们手写的所有逻辑——创建参数、前向传播、
# 算loss、GradientTape录像、链式法则求梯度、循环分批喂数据——
# Keras 全部帮你封装好了。