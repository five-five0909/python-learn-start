# ============================================================
# TensorFlow GradientTape 完整教程
# 分类结构：可以按模块单独执行
# ============================================================
# 安装（在终端运行，不是Python里）：
#   pip install tensorflow
#   pip install tensorflow-macos  ← Mac M1/M2芯片用这个
# ============================================================

import tensorflow as tf

# ============================================================
# 模块一：标量求导（最基础）
# 公式：y = 2x + 3，梯度 = dy/dx = 2
# ============================================================

def demo_scalar_gradient():
    """标量求导 —— 单个数字"""
    print("=" * 50)
    print("模块一：标量求导（单个数字）")
    print("=" * 50)

    x = tf.Variable(0.)          # 可学习变量，初始值=0（必须是浮点数）

    with tf.GradientTape() as tape:
        y = 2 * x + 3            # 前向传播：记录 y = 2x+3 的运算过程

    grad = tape.gradient(y, x)   # 反向：y对x求导 = 2

    print(f"x 的值:     {x.numpy()}")    # 0.0
    print(f"y 的值:     {y.numpy()}")    # 3.0
    print(f"dy/dx 梯度: {grad.numpy()}") # 2.0
    print()
    print("梯度=2（正数）→ 要让y减小，x应该往左减小")
    print("更新规则：x_new = x - 学习率 × 梯度")
    print()


# ============================================================
# 模块二：矩阵求导（张量推广）
# 公式：每个位置独立计算 y[i][j] = 2*x[i][j] + 3
# ============================================================

def demo_matrix_gradient():
    """矩阵求导 —— 张量推广，原理与标量完全相同"""
    print("=" * 50)
    print("模块二：矩阵求导（2×2张量）")
    print("=" * 50)

    x = tf.Variable(tf.zeros((2, 2)))   # 2×2全零矩阵

    with tf.GradientTape() as tape:
        y = 2 * x + 3                   # 逐元素计算，每个位置独立

    grad = tape.gradient(y, x)

    print(f"x 的值:\n{x.numpy()}")
    print(f"dy/dx 梯度:\n{grad.numpy()}")
    print()
    print("关键结论：梯度的形状 = 参数的形状")
    print("  401,408个权重 → 401,408个梯度，一一对应")
    print()


# ============================================================
# 模块三：同时对W和b求导（模拟Dense层）
# 公式：y = matmul(x, W) + b  ← 这正是Dense层的核心运算
# ============================================================

def demo_dense_layer_gradient():
    """同时对W和b求导 —— 真实Dense层的前向+反向"""
    print("=" * 50)
    print("模块三：同时对W和b求导（模拟Dense层）")
    print("=" * 50)

    # 可学习参数（用 tf.Variable 包裹，GradientTape会自动追踪）
    W = tf.Variable(tf.random.uniform((2, 2)))   # 权重矩阵，随机初始化
    b = tf.Variable(tf.zeros((2,)))               # 偏置，初始化为0

    # 输入数据（不是可学习参数，用普通张量即可）
    x = tf.random.uniform((2, 2))

    # 打印 W 和 b 的初始值
    print(f"W 初始值:\n{W.numpy()}")
    print(f"b 初始值:\n{b.numpy()}")
    print(f"x 初始值:\n{x.numpy()}")
    print()

    with tf.GradientTape() as tape:
        y = tf.matmul(x, W) + b                  # Dense层的核心：dot(x,W)+b

    grads = tape.gradient(y, [W, b])              # 同时对W和b求梯度
    grad_W, grad_b = grads[0], grads[1]

    print(f"W 形状: {W.shape}  → 梯度形状: {grad_W.shape}")  # 都是 (2,2)
    print(f"b 形状: {b.shape}   → 梯度形状: {grad_b.shape}") # 都是 (2,)
    print()
    print(f"W 的梯度:\n{grad_W.numpy()}")
    print(f"b 的梯度:\n{grad_b.numpy()}")
    print()
    print("有了梯度，优化器就可以更新参数：")
    print("  W = W - 学习率 × grad_W")
    print("  b = b - 学习率 × grad_b")
    print()


# ============================================================
# 模块四：完整训练循环（前向 + 反向 + 参数更新）
# 这就是 model.fit() 背后做的事！
# ============================================================

def demo_full_training_loop():
    """完整训练一步：前向传播 → 计算loss → 反向传播 → 更新参数"""
    print("=" * 50)
    print("模块四：完整训练循环（model.fit的底层逻辑）")
    print("=" * 50)

    # 超参数
    learning_rate = 0.1
    epochs = 3

    # 参数初始化
    W = tf.Variable(tf.random.uniform((2, 2)))
    b = tf.Variable(tf.zeros((2,)))

    # 数据（假设）
    x_data = tf.random.uniform((2, 2))
    y_true  = tf.ones((2, 2))           # 假设真实值全是1

    # 优化器（对应 model.compile(optimizer='rmsprop')）
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # ① 前向传播
            y_pred = tf.matmul(x_data, W) + b

            # ② 计算 loss（均方误差，对应 loss='mse'）
            loss = tf.reduce_mean((y_pred - y_true) ** 2)

        # ③ 反向传播：链式法则全由框架自动完成
        grads = tape.gradient(loss, [W, b])

        # ④ 更新参数：W = W - lr × grad_W
        optimizer.apply_gradients(zip(grads, [W, b]))

        print(f"  第{epoch + 1}轮：loss = {loss.numpy():.4f}")

    print()
    print("以上就是 model.fit() 在背后做的事！")
    print("Keras 把这些全部封装成了两行代码：")
    print("  model.compile(optimizer='rmsprop', loss='mse')")
    print("  model.fit(x_train, y_train, epochs=5)")
    print()


# ============================================================
# 附录：model.fit() 等价伪代码（注释形式，供参考）
# ============================================================

# model.fit() 的完整底层逻辑：
#
# for epoch in range(epochs):
#     with tf.GradientTape() as tape:
#         y_pred = model(x_train)              # ① 前向传播
#         loss   = loss_fn(y_true, y_pred)     # ② 计算loss
#
#     grads = tape.gradient(                   # ③ 反向传播（链式法则）
#         loss,
#         model.trainable_weights              #    所有406,528个w和b
#     )
#     optimizer.apply_gradients(               # ④ 更新所有参数
#         zip(grads, model.trainable_weights)
#     )


# ============================================================
# 主程序入口：依次执行全部模块
# 也可以单独调用某一个函数来执行对应模块
# ============================================================

if __name__ == "__main__":
    # demo_scalar_gradient()       # 模块一
    # demo_matrix_gradient()       # 模块二
    # demo_dense_layer_gradient()  # 模块三
    demo_full_training_loop()    # 模块四