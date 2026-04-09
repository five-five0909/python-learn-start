import math
import numpy as np
import tensorflow as tf


# ============================================================
# 第一步：加载数据
# ============================================================

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 展平：(60000,28,28)→(60000,784)，转float32，归一化到0~1
train_images = train_images.reshape((60000, 784)).astype("float32") / 255.0
test_images = test_images.reshape((10000, 784)).astype("float32") / 255.0

print(f"训练集：{train_images.shape[0]} 张，每张 {train_images.shape[1]} 个像素")
print(f"测试集：{test_images.shape[0]} 张")


# ============================================================
# 第二步：创建参数（W1, b1, W2, b2 共407,050个）
# ============================================================

# 第一层：W1(784,512) + b1(512,)
W1 = tf.Variable(tf.random.uniform((784, 512), minval=0, maxval=0.1))
b1 = tf.Variable(tf.zeros((512,)))

# 第二层：W2(512,10) + b2(10,)
W2 = tf.Variable(tf.random.uniform((512, 10), minval=0, maxval=0.1))
b2 = tf.Variable(tf.zeros((10,)))

all_weights = [W1, b1, W2, b2]

total = sum(w.numpy().size for w in all_weights)
print(f"总参数量：{total}")


# ============================================================
# 第三步：前向传播
# ============================================================

def forward(x):
    """x(batch,784) → z1=x·W1+b1 → h=ReLU(z1) → z2=h·W2+b2 → p=Softmax(z2)"""
    z1 = tf.matmul(x, W1) + b1
    h = tf.nn.relu(z1)
    z2 = tf.matmul(h, W2) + b2
    p = tf.nn.softmax(z2)
    return p


# ============================================================
# 第四步：loss函数（交叉熵）
# ============================================================

def compute_loss(predictions, labels):
    """loss = -log(正确类别的概率)，取平均"""
    per_sample = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    return tf.reduce_mean(per_sample)


# ============================================================
# 第五步：单次训练步骤
# ============================================================

learning_rate = 0.001

def train_one_step(images_batch, labels_batch):
    """前向 → loss → 反向 → 更新参数"""
    # GradientTape 是 TensorFlow 的"自动求导录像机"
    # 在 with 块内的所有运算都会被记录下来
    # 之后调用 tape.gradient() 时，它会用链式法则反向推导梯度
    with tf.GradientTape() as tape:
        predictions = forward(images_batch)
        loss = compute_loss(predictions, labels_batch)

    # tape.gradient(loss, all_weights) 做了什么？
    # ────────────────────────────────────────────────
    # 它从 loss 这个"终点"出发，沿着运算链反向走回去，
    # 用链式法则算出每个参数对 loss 的影响程度（偏导数）
    #
    # 返回结果是一个列表，和 all_weights 一一对应：
    #   gradients[0] = ∂loss/∂W1  形状 (784, 512) — 401,408 个偏导数
    #   gradients[1] = ∂loss/∂b1  形状 (512,)    —     512 个偏导数
    #   gradients[2] = ∂loss/∂W2  形状 (512, 10) —   5,120 个偏导数
    #   gradients[3] = ∂loss/∂b2  形状 (10,)     —      10 个偏导数
    #
    # 每个偏导数的含义：
    #   正数 → 增大这个参数会让 loss 增大 → 应该减小这个参数
    #   负数 → 增大这个参数会让 loss 减小 → 应该增大这个参数
    #   接近零 → 这个参数对 loss 影响很小 → 几乎不用调
    #
    # 为什么能自动算？因为 GradientTape 记录了前向传播的每一步：
    #   x → matmul → z1 → relu → h → matmul → z2 → softmax → p → crossentropy → loss
    # 反向时沿着这条链一步步求导，自动应用链式法则：
    #   ∂loss/∂W2 = ∂loss/∂z2 · ∂z2/∂W2
    #   ∂loss/∂h  = ∂loss/∂z2 . ∂z2/∂h
    #   ∂loss/∂z1 = ∂loss/∂h  . ∂h/∂z1  (注意 ReLU 的导数：正数位置=1，负数位置=0)
    #   ∂loss/∂W1 = ∂loss/∂z1 . ∂z1/∂W1
    gradients = tape.gradient(loss, all_weights)

    # 更新参数：w_new = w_old - learning_rate × ∂loss/∂w
    # 减号是因为梯度指向 loss 增大的方向，我们要让 loss 减小，所以要反着走
    for i in range(len(all_weights)):
        all_weights[i].assign_sub(learning_rate * gradients[i])

    return loss


# ============================================================
# 第六步：训练
# ============================================================

epochs = 10
batch_size = 128
num_batches = math.ceil(len(train_images) / batch_size)

for epoch in range(epochs):
    print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
    index = 0

    for batch in range(num_batches):
        images_batch = train_images[index : index + batch_size]
        labels_batch = train_labels[index : index + batch_size]
        index += batch_size

        loss = train_one_step(images_batch, labels_batch)

        if batch % 100 == 0:
            print(f"  第 {batch:3d}/{num_batches} 批，loss = {loss:.4f}")

    print(f"  本轮结束，loss = {loss:.4f}")


# ============================================================
# 第七步：测试
# ============================================================

predictions = forward(test_images).numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = (predicted_labels == test_labels)
accuracy = np.mean(matches)

print(f"\n===== 测试结果 =====")
print(f"10000 张图里猜对了 {int(matches.sum())} 张")
print(f"准确率：{accuracy:.4f}")