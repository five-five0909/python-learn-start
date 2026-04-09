from keras.datasets import mnist

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 查看训练数据的形状
# 第一维：60000
# 这是样本数量，表示这个张量包含了60,000个独立的样本
# 每个样本都是一个完整的28×28矩阵
# 在机器学习中，这通常代表训练数据集的规模
# 第二维：28
# 这是行数，每个矩阵有28行
# 在图像数据中，这对应图像的高度（28像素）
# 第三维：28
# 这是列数，每个矩阵有28列
# 在图像数据中，这对应图像的宽度（28像素）
print("训练图像形状:", train_images.shape)  # (60000, 28, 28)
# 与之前训练集的图片对应，每个图片代表的真实标签【这里也就是图片表示的数字含义】
print("训练标签数量:", len(train_labels))  # 60000
print("训练标签示例:", train_labels[:10])  # 显示前10个标签
# 查看测试数据的形状
print("\n测试图像形状:", test_images.shape)  # (10000, 28, 28)
print("测试标签数量:", len(test_labels))  # 10000
print("测试标签示例:", test_labels[:10])  # 显示前10个标签
# 查看单个图像的信息
print("\n第一个训练图像的形状:", train_images[0].shape)  # (28, 28)
print("第一个训练图像的标签:", train_labels[0])  # 标签值

# 示例 1: 选择第 10 到第 100 个数字（不包括第 100 个）
# 形状将变为 (90, 28, 28)
my_slice = train_images[10:100]
print("切片 1 形状 (10:100):", my_slice.shape)

# 下面两种写法与上面完全等同：
# 写法 A: 使用 : 代替选择整个轴
my_slice_a = train_images[10:100, :, :]
print("切片 1A 形状 (使用 :):", my_slice_a.shape)

# 写法 B: 给出具体的索引范围
my_slice_b = train_images[10:100, 0:28, 0:28]
print("切片 1B 形状 (明确范围):", my_slice_b.shape)


# 示例 2: 在所有图像的右下角选出 14x14 像素的区域
# [:, 14:, 14:] 表示：所有样本，从第14行到最后，从第14列到最后
my_slice_corner = train_images[:, 14:, 14:]
print("右下角切片形状:", my_slice_corner.shape)  # (60000, 14, 14)


# 示例 3: 使用负数索引（相对于轴终点的位置）
# 在图像中心裁剪出 14x14 像素的区域
# 28像素宽高的图片，去掉上下各7像素，左右各7像素，剩下的就是中间的 14x14
my_slice_center = train_images[:, 7:-7, 7:-7]
print("中心裁剪切片形状:", my_slice_center.shape)  # (60000, 14, 14)

# --- 补充：2.2.7 数据批量的概念 ---
# 通常深度学习中第一个轴（轴0）被称为 样本轴 (samples axis) 或 样本维度
batch = train_images[:128]  # 这就是获取第一个“批量”(batch) 的数据
