import tensorflow as tf
import numpy as np

print(f"TensorFlow 版本: {tf.__version__}")
print(f"NumPy 版本: {np.__version__}")
print(f"GPU 是否可用: {tf.config.list_physical_devices('GPU')}\n")

# ==========================================
# 3.5.1 常数张量和变量
# ==========================================
print("=" * 50)
print("3.5.1 常数张量和变量")
print("=" * 50)

# --------------------------------------------------
# 1. 创建全1张量和全0张量
# --------------------------------------------------
print("\n--- 代码清单 3-1: 全1张量或全0张量 ---")

# 【API 详解】tf.ones(shape, dtype=tf.float32, name=None)
#   - shape: 必选参数，指定张量的形状，可以是 list 或 tuple，例如 (2, 3) 或 [2, 3]。
#   - dtype: 可选参数，指定张量元素的数据类型，默认为 tf.float32。
#            常见类型包括 tf.float16, tf.float32, tf.float64, tf.int32, tf.int64 等。
#   - name:  可选参数，为该操作指定一个名称，便于在计算图中识别和调试。
#   - 返回值: 一个指定形状和类型的张量，所有元素均为 1。
#   - 注意: 返回的张量是常量（tf.Tensor），不可直接修改其元素值。
x_ones = tf.ones(shape=(2, 1))
print("全1张量 (默认 dtype=tf.float32):\n", x_ones)
print("数据类型:", x_ones.dtype)

# 指定数据类型创建全1张量
x_ones_int = tf.ones(shape=(2, 1), dtype=tf.int32)
print("\n全1张量 (dtype=tf.int32):\n", x_ones_int)
print("数据类型:", x_ones_int.dtype)

# 【API 详解】tf.zeros(shape, dtype=tf.float32, name=None)
#   - 参数与 tf.ones 完全一致，区别仅在于所有元素初始化为 0 而非 1。
#   - 常用于初始化偏置项（bias）或创建占位张量。
x_zeros = tf.zeros(shape=(2, 1))
print("\n全0张量:\n", x_zeros)

# 【补充 API】tf.fill(dims, value, name=None)
#   - dims:   指定输出张量的形状，例如 [2, 3]。
#   - value:  填充的标量值，张量中所有元素都将等于该值。
#   - 返回值: 一个指定形状的张量，所有元素均为 value。
#   - 与 tf.ones/tf.zeros 的区别: 可以填充任意标量值，而不仅限于 0 或 1。
x_fill = tf.fill(dims=(2, 3), value=7.0)
print("\n使用 tf.fill 填充指定值 (7.0):\n", x_fill)

# 【补充 API】tf.ones_like(input) / tf.zeros_like(input)
#   - input: 一个张量（或类张量对象），新张量将继承其形状和数据类型。
#   - 返回值: 与 input 形状和类型相同的全1/全0张量。
#   - 使用场景: 当你需要创建一个与已有张量形状相同的零张量或一张量时非常方便，
#              避免手动提取 shape 再传入 tf.ones/tf.zeros。
sample = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print("\n原始张量 sample:\n", sample)
print("ones_like:\n", tf.ones_like(sample))
print("zeros_like:\n", tf.zeros_like(sample))


# --------------------------------------------------
# 2. 创建随机张量
# --------------------------------------------------
print("\n--- 代码清单 3-2: 随机张量 ---")

# 【API 详解】tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#   - shape:  必选参数，指定输出张量的形状。
#   - mean:   正态分布的均值（中心位置），默认为 0.0。
#   - stddev: 正态分布的标准差（离散程度），必须为正数，默认为 1.0。
#             stddev 越大，生成的值越分散；越小则越集中在 mean 附近。
#   - dtype:  输出张量的数据类型，仅支持浮点类型 (float16, float32, float64)。
#   - seed:   随机种子，设置后可保证每次运行生成相同的随机数序列，用于实验复现。
#   - 返回值: 从正态分布 N(mean, stddev²) 中独立采样得到的张量。
#   - 数学含义: 约 68% 的值落在 [mean-stddev, mean+stddev] 区间内，
#              约 95% 落在 [mean-2*stddev, mean+2*stddev] 内。
x_normal = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
print("正态分布随机张量 (mean=0, stddev=1):\n", x_normal)

# 使用 seed 确保可复现
tf.random.set_seed(42)
x_normal_seed = tf.random.normal(shape=(3, 1), mean=0., stddev=1., seed=42)
print("\n带 seed 的正态分布随机张量 (可复现):\n", x_normal_seed)

# 调整 mean 和 stddev 的效果
x_normal_custom = tf.random.normal(shape=(5, 1), mean=10., stddev=0.1)
print("\n正态分布 (mean=10, stddev=0.1, 值集中在10附近):\n", x_normal_custom)

# 【API 详解】tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
#   - shape:  必选参数，指定输出张量的形状。
#   - minval: 均匀分布的下界（包含），默认为 0。
#   - maxval: 均匀分布的上界（不包含），即取值范围为 [minval, maxval)。
#             如果 dtype 是整数类型，maxval 必须显式指定。
#   - dtype:  输出张量的数据类型。若为整数类型 (int32, int64)，则 minval 和 maxval 也应为整数。
#   - seed:   随机种子，作用与 tf.random.normal 中的 seed 相同。
#   - 返回值: 在 [minval, maxval) 区间内均匀采样的张量，每个值被抽到的概率相等。
#   - 使用场景: 常用于数据增强中的随机裁剪、随机翻转阈值，或初始化权重。
x_uniform = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
print("\n均匀分布随机张量 [0, 1):\n", x_uniform)

# 均匀分布整数随机张量
x_uniform_int = tf.random.uniform(shape=(3, 1), minval=0, maxval=10, dtype=tf.int32)
print("\n均匀分布整数随机张量 [0, 10):\n", x_uniform_int)

# 【补充 API】tf.random.gamma(shape, alpha, beta=1.0, dtype=tf.float32, seed=None, name=None)
#   - alpha: 伽马分布的形状参数（shape parameter），必须为正数。
#   - beta:  伽马分布的逆尺度参数（rate parameter），默认为 1.0。
#   - 返回值: 从 Gamma(alpha, beta) 分布中采样的张量。
#   - 使用场景: 在贝叶斯神经网络或某些概率模型中用到。
x_gamma = tf.random.gamma(shape=(3, 1), alpha=2.0)
print("\n伽马分布随机张量 (alpha=2.0):\n", x_gamma)


# --------------------------------------------------
# 3. 常数张量的不可变性 (NumPy vs TensorFlow)
# --------------------------------------------------
print("\n--- 常数张量的不可变性 (NumPy vs TensorFlow) ---")

# 代码清单 3-3: NumPy 数组是可赋值的（Mutable）
# NumPy 数组底层是一块连续的内存区域，Python 通过引用直接操作这块内存，
# 因此可以通过索引直接修改其中的元素值。
x_np = np.ones(shape=(2, 2))
print("原始 NumPy 数组:\n", x_np)
x_np[0, 0] = 0.           # 直接通过索引修改第一个元素
x_np[1, 1] = 99.          # 修改另一个元素
print("修改后的 NumPy 数组:\n", x_np)
print("NumPy 数组是可变的 (Mutable): 可以通过索引直接赋值修改元素 ✅")

# 代码清单 3-4: TensorFlow 常数张量是不可赋值的（Immutable）
# tf.Tensor 是 TensorFlow 计算图中的一个节点（操作的输出），设计上是不可变的。
# 不可变性的好处:
#   1. 便于计算图优化（如常量折叠、公共子表达式消除）。
#   2. 保证线程安全，多线程/多 GPU 并行计算时不会出现数据竞争。
#   3. 支持自动微分——梯度计算依赖于前向计算的确定性。
# 如果需要修改张量中的值，必须使用 tf.Variable（可变变量）。
x_tf = tf.ones(shape=(2, 2))
print("\n原始 TensorFlow 张量:\n", x_tf)
print("尝试通过索引修改 TensorFlow 常数张量 (预期会报错)...")
try:
    x_tf[0, 0] = 0.
except Exception as e:
    print("【程序报错拦截成功】:", type(e).__name__, "-", e)
    print("原因: tf.Tensor 是不可变对象 (Immutable)，不支持 item assignment。")

# 【补充】TensorFlow 张量虽然不能直接修改，但支持丰富的索引/切片读取操作
x_slice = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
print("\n原始张量 x_slice:\n", x_slice)
print("读取单个元素 x_slice[0, 1]:", x_slice[0, 1].numpy())
print("切片读取 x_slice[0:2, :]:\n", x_slice[0:2, :].numpy())
print("步长切片 x_slice[::2, ::2]:\n", x_slice[::2, ::2].numpy())
print("负索引 x_slice[-1, -1]:", x_slice[-1, -1].numpy())


# --------------------------------------------------
# 4. TensorFlow 变量 (Variable)
# --------------------------------------------------
print("\n--- 代码清单 3-5 ~ 3-8: TensorFlow 变量 (Variable) ---")

# 【API 详解】tf.Variable(initial_value=None, trainable=True, name=None, dtype=None, ...)
#   - initial_value: 必选参数（在某些版本中可省略但不推荐），指定变量的初始值。
#                    可以是一个张量、NumPy 数组、Python 标量或返回张量的函数。
#   - trainable:     布尔值，默认为 True。设为 True 时，该变量会被加入到
#                    tf.trainable_variables() 列表中，意味着梯度下降优化器会更新它。
#                    对于不需要训练的变量（如 BatchNorm 的移动均值），应设为 False。
#   - name:          变量的名称，在 TensorBoard 可视化和模型保存/加载时非常有用。
#   - dtype:         变量的数据类型。如果 initial_value 已指定类型，此参数可以省略。
#   - 核心特性:
#       1. tf.Variable 是可变的 (Mutable)，其内部值可以通过 .assign() 等方法修改。
#       2. 在计算图中，变量是一个有状态的节点，跨多次 session.run() / tf.function 调用保持值。
#       3. 变量的值存储在内存（或 GPU 显存）中，而非计算图的定义中。
#   - 与 tf.Tensor 的核心区别:
#       tf.Tensor    → 不可变，无状态，是计算图中某个操作的输出。
#       tf.Variable  → 可变，有状态，是计算图中一个持久化的存储单元。

v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)), name="my_variable")
print("变量名称:", v.name)
print("是否可训练:", v.trainable)
print("初始变量 v:\n", v.numpy())

# 【补充 API】tf.Variable 的常用属性
print("\n--- tf.Variable 常用属性 ---")
print("v.shape:", v.shape)         # 张量形状，等价于 v.get_shape()
print("v.dtype:", v.dtype)         # 数据类型
print("v.name:", v.name)           # 变量名称
print("v.device:", v.device)       # 变量所在的设备 (CPU/GPU)
print("v.trainable:", v.trainable) # 是否参与训练

# 【API 详解】.assign(value, use_locking=False, name=None, read_value=True)
#   - value:       要赋的新值，必须与变量具有相同的形状和数据类型（或可被广播）。
#   - use_locking: 是否在赋值操作期间加锁，防止多线程竞争，默认为 False。
#   - read_value:  若为 True（默认），返回赋值后的变量值；若为 False，返回赋值操作本身。
#   - 返回值: 赋值后的变量值（tf.Tensor）。
#   - 注意: .assign() 会替换变量的全部内容，形状必须与原始变量兼容。
# 代码清单 3-6: 整体赋值
v.assign(tf.ones((3, 1)))
print("\n使用 assign 整体赋值后的 v:\n", v.numpy())

# 【API 详解】.assign_sub(value, use_locking=False, name=None, read_value=True)
#   - 功能: 将变量减去 value，等价于 v = v - value（即 Python 中的 -= 操作）。
#   - 参数: 与 .assign() 相同。
#   - 使用场景: 在某些自定义训练循环中手动更新权重（如权重衰减）。
v.assign_sub(tf.ones((3, 1)) * 0.5)
print("\n使用 assign_sub 减去 0.5 后的 v:\n", v.numpy())

# 【API 详解】对变量的子集进行赋值
#   - 变量支持通过索引获取子张量（返回 tf.Tensor），然后对该子张量调用 .assign()。
#   - 这实际上是对变量底层内存中特定位置的原地修改。
# 代码清单 3-7: 为子集赋值
v[0, 0].assign(3.)
v[1, 0].assign(5.)
v[2, 0].assign(7.)
print("\n为各个子元素分别赋值后的 v:\n", v.numpy())

# 【API 详解】.assign_add(value, use_locking=False, name=None, read_value=True)
#   - 功能: 将变量加上 value，等价于 v = v + value（即 Python 中的 += 操作）。
#   - 参数: 与 .assign() 相同。
#   - 使用场景: 累积梯度、计数器递增、损失累加等。
# 代码清单 3-8: 使用 assign_add
v.assign_add(tf.ones((3, 1)))
print("\n使用 assign_add 加上全1后的 v:\n", v.numpy())

# 【补充】批量操作示例: 模拟一个简单的梯度更新过程
print("\n--- 补充: 模拟梯度更新 ---")
weight = tf.Variable(tf.random.normal(shape=(2, 2)), name="weight")
learning_rate = 0.01
fake_gradient = tf.constant([[0.5, 0.3], [0.1, 0.8]])  # 假设的梯度

print("更新前的 weight:\n", weight.numpy())
# 梯度下降: w = w - lr * grad
weight.assign_sub(learning_rate * fake_gradient)
print(f"学习率={learning_rate} 更新后的 weight:\n", weight.numpy())

# 【补充 API】tf.scatter_nd 和 tf.Variable.scatter_nd_update
#   - 用于对变量的稀疏位置进行批量更新，比逐个 .assign() 更高效。
indices = tf.constant([[0, 0], [1, 1]])
updates = tf.constant([100., 200.])
weight.scatter_nd_update(indices, updates)
print("\n使用 scatter_nd_update 稀疏更新后的 weight:\n", weight.numpy())


# ==========================================
# 3.5.2 张量运算：用 TensorFlow 进行数学运算
# ==========================================
print("\n" + "=" * 50)
print("3.5.2 张量运算：用 TensorFlow 进行数学运算")
print("=" * 50)

# --------------------------------------------------
# 1. 基本数学运算
# --------------------------------------------------
print("\n--- 代码清单 3-9: 基本数学运算 ---")

a = tf.ones((2, 2)) * 4.0  # 创建一个 2x2 的全4张量，方便肉眼验证结果
print("原始张量 a:\n", a.numpy())

# 【API 详解】tf.square(x, name=None)
#   - x:    输入张量，可以是任意形状和数据类型（需为数值类型）。
#   - name: 操作名称（可选）。
#   - 返回值: 与 x 形状相同的张量，每个元素为原元素的平方。
#   - 数学: y[i] = x[i]²
#   - 注意: 这是逐元素 (element-wise) 运算，不是矩阵乘法。
#   - 使用场景: 计算 MSE 损失时需要对预测值与真实值之差求平方。
b = tf.square(a)
print("\n张量 a 的平方 (tf.square):\n", b.numpy())
print("验证: 4.0² =", 4.0 ** 2)  # 应等于 16.0

# 【API 详解】tf.sqrt(x, name=None)
#   - x:    输入张量，元素值必须为非负数（否则会产生 NaN）。
#   - 返回值: 与 x 形状相同的张量，每个元素为原元素的正平方根。
#   - 数学: y[i] = √x[i]
#   - 注意: 输入必须是浮点类型。如果输入整数张量，需要先用 tf.cast() 转换。
#   - 使用场景: 计算标准差 (std = √variance)、欧几里得距离、归一化等。
c = tf.sqrt(a)
print("\n张量 a 的平方根 (tf.sqrt):\n", c.numpy())
print("验证: √4.0 =", np.sqrt(4.0))  # 应等于 2.0

# 【补充 API】逐元素基本运算汇总
print("\n--- 逐元素基本运算汇总 ---")
demo = tf.constant([[1.0, 4.0], [9.0, 16.0]])

# tf.exp(x): 自然指数函数 e^x
print("tf.exp (e^x):\n", tf.exp(demo).numpy())

# tf.log(x) / tf.math.log(x): 自然对数 ln(x)，x 必须为正数
print("tf.log (ln(x)):\n", tf.math.log(demo).numpy())

# tf.abs(x): 逐元素取绝对值
demo_neg = tf.constant([[-3.0, 2.0], [-7.0, 5.0]])
print("tf.abs (|x|):\n", tf.abs(demo_neg).numpy())

# tf.sign(x): 逐元素取符号，正数返回 1.0，负数返回 -1.0，零返回 0.0
print("tf.sign (sign(x)):\n", tf.sign(demo_neg).numpy())

# tf.clip_by_value(x, clip_value_min, clip_value_max): 将值裁剪到指定范围
#   - 小于 clip_value_min 的值被替换为 clip_value_min
#   - 大于 clip_value_max 的值被替换为 clip_value_max
#   - 使用场景: 梯度裁剪（防止梯度爆炸）、ReLU 变体等
print("tf.clip_by_value (裁剪到 [2, 10]):\n",
      tf.clip_by_value(demo, clip_value_min=2.0, clip_value_max=10.0).numpy())


# --------------------------------------------------
# 2. 算术运算 (逐元素 + 广播)
# --------------------------------------------------
print("\n--- 算术运算 (逐元素运算与广播) ---")

m = tf.constant([[1.0, 2.0], [3.0, 4.0]])
n = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print("张量 m:\n", m.numpy())
print("张量 n:\n", n.numpy())

# 【API 详解】逐元素算术运算
#   - tf.add(x, y) / x + y:       逐元素加法，y[i] = x[i] + y[i]
#   - tf.subtract(x, y) / x - y:   逐元素减法
#   - tf.multiply(x, y) / x * y:   逐元素乘法（Hadamard 积），注意不是矩阵乘法！
#   - tf.divide(x, y) / x / y:     逐元素除法
#   - tf.pow(x, y) / x ** y:       逐元素幂运算
#   - tf.mod(x, y):                逐元素取模（余数）
#   - 以上运算都支持广播 (Broadcasting):
#       当两个张量形状不完全相同时，TensorFlow 会自动扩展较小张量的维度以匹配较大张量。
#       广播规则（从最右侧维度开始对齐）:
#         1. 如果某维度大小相同 → 直接运算
#         2. 如果某维度一个是 1 → 扩展为另一个的大小
#         3. 如果某维度既不同也不是 1 → 报错

print("\n逐元素加法 m + n:\n", tf.add(m, n).numpy())
print("逐元素减法 m - n:\n", tf.subtract(m, n).numpy())
print("逐元素乘法 m * n (Hadamard积):\n", tf.multiply(m, n).numpy())
print("逐元素除法 n / m:\n", tf.divide(n, m).numpy())
print("逐元素幂运算 m ** 2:\n", tf.pow(m, 2).numpy())

# 广播示例
scalar = tf.constant(10.0)
row_vec = tf.constant([[1.0, 2.0]])  # 形状 (1, 2)
print("\n广播: 张量 + 标量 (10.0):\n", tf.add(m, scalar).numpy())
print("广播: 张量 + 行向量 [[1, 2]]:\n", tf.add(m, row_vec).numpy())

# 【补充 API】tf.math.mod(x, y, name=None)
#   - 逐元素取模运算，返回 x 除以 y 的余数。
#   - 对于浮点数，结果的符号与 y 相同。
print("逐元素取模 (10 % 3):\n",
      tf.math.mod(tf.constant([10.0, 11.0, 12.0]), 3.0).numpy())


# --------------------------------------------------
# 3. 矩阵运算
# --------------------------------------------------
print("\n--- 矩阵运算 ---")

A = tf.constant([[1.0, 2.0], [3.0, 4.0]])
B = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print("矩阵 A:\n", A.numpy())
print("矩阵 B:\n", B.numpy())

# 【API 详解】tf.matmul(a, b, transpose_a=False, transpose_b=False, ...)
#   - a, b: 输入张量，维度 >= 2。最后两个维度被视为矩阵维度，
#           前面的维度被视为批次维度（batch），会被批量矩阵乘法处理。
#   - transpose_a: 若为 True，在乘法之前先对 a 做转置。
#   - transpose_b: 若为 True，在乘法之前先对 b 做转置。
#   - 数学: 对于 2D 张量，C[i,j] = Σ_k A[i,k] * B[k,j]
#   - 与 tf.multiply 的区别:
#       tf.multiply(a, b) → 逐元素乘法 (Hadamard积)，要求 a 和 b 形状相同或可广播。
#       tf.matmul(a, b)   → 矩阵乘法，要求 a 的列数等于 b 的行数。
#   - 使用场景: 全连接层的前向传播 (output = input @ weights + bias)。
C_matmul = tf.matmul(A, B)
print("\n矩阵乘法 A @ B (tf.matmul):\n", C_matmul.numpy())
# 手动验证: C[0,0] = 1*5 + 2*7 = 19, C[0,1] = 1*6 + 2*8 = 22, ...

# 【补充】@ 运算符等价于 tf.matmul
C_at = A @ B
print("矩阵乘法 A @ B (使用 @ 运算符):\n", C_at.numpy())

# 【补充 API】tf.transpose(a, perm=None, name=None)
#   - a: 输入张量。
#   - perm: 排列顺序，一个整数列表，指定各维度的新顺序。
#           例如 perm=[1, 0] 表示交换第0维和第1维（即矩阵转置）。
#           perm=[0, 2, 1] 对于 3D 张量表示交换最后两个维度。
#   - 如果 perm 为 None，默认反转所有维度 (完全转置)。
print("矩阵 A 的转置:\n", tf.transpose(A).numpy())

# 【补充 API】tf.linalg.trace(x)
#   - 计算方阵的迹（对角线元素之和）。
#   - 数学: trace(A) = Σ_i A[i, i]
print("矩阵 A 的迹 (trace):", tf.linalg.trace(A).numpy())

# 【补充 API】tf.linalg.det(input)
#   - 计算方阵的行列式 (determinant)。
#   - 对于 2x2 矩阵 [[a, b], [c, d]]，det = ad - bc
print("矩阵 A 的行列式 (det):", tf.linalg.det(A).numpy())

# 【补充 API】tf.linalg.inv(input)
#   - 计算方阵的逆矩阵 A⁻¹，使得 A @ A⁻¹ = I（单位矩阵）。
#   - 注意: 仅对方阵且行列式不为零的矩阵有意义。
A_inv = tf.linalg.inv(A)
print("矩阵 A 的逆矩阵:\n", A_inv.numpy())
print("验证 A @ A⁻¹ (应为单位矩阵):\n", (A @ A_inv).numpy())


# --------------------------------------------------
# 4. 约减运算 (Reduction Operations)
# --------------------------------------------------
print("\n--- 约减运算 (Reduction) ---")

r = tf.constant([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print("张量 r (shape=2x3):\n", r.numpy())

# 【API 详解】tf.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)
#   - input_tensor: 输入张量。
#   - axis:         约减的维度。None 表示对所有维度求和（返回标量）；
#                   0 表示沿第0维（行方向）求和，结果形状为 (3,)；
#                   1 表示沿第1维（列方向）求和，结果形状为 (2,)。
#                   可以是一个列表，同时约减多个维度，如 axis=[0, 1]。
#   - keepdims:     若为 True，约减后保留被约减维度（大小变为 1），便于后续广播运算。
#   - 返回值: 约减后的张量，维度比输入少（除非 keepdims=True）。
#   - 使用场景: 计算损失函数的总和、批量数据的汇总统计等。
print("reduce_sum (全部元素求和):", tf.reduce_sum(r).numpy())
print("reduce_sum (axis=0, 按列求和):", tf.reduce_sum(r, axis=0).numpy())
print("reduce_sum (axis=1, 按行求和):", tf.reduce_sum(r, axis=1).numpy())
print("reduce_sum (axis=1, keepdims=True):\n", tf.reduce_sum(r, axis=1, keepdims=True).numpy())

# 【API 详解】tf.reduce_mean / tf.reduce_max / tf.reduce_min
#   - 参数与 tf.reduce_sum 完全一致。
#   - reduce_mean: 计算均值。
#   - reduce_max:  计算最大值。
#   - reduce_min:  计算最小值。
#   - 使用场景: reduce_mean 常用于计算平均损失；reduce_max/min 用于找极值。
print("\nreduce_mean (全部均值):", tf.reduce_mean(r).numpy())
print("reduce_mean (axis=0, 每列均值):", tf.reduce_mean(r, axis=0).numpy())
print("reduce_max (全部最大值):", tf.reduce_max(r).numpy())
print("reduce_max (axis=1, 每行最大值):", tf.reduce_max(r, axis=1).numpy())
print("reduce_min (全部最小值):", tf.reduce_min(r).numpy())

# 【补充 API】tf.reduce_prod(input_tensor, axis=None, keepdims=False)
#   - 计算张量中所有元素（或沿指定维度）的乘积。
print("reduce_prod (全部元素乘积):", tf.reduce_prod(r).numpy())  # 1*2*3*4*5*6 = 720

# 【补充 API】tf.argmax / tf.argmin
#   - tf.argmax(input, axis=None, output_type=tf.int64):
#       返回沿指定维度上最大值所在的索引，而非最大值本身。
#   - tf.argmin(input, axis=None, output_type=tf.int64):
#       返回沿指定维度上最小值所在的索引。
#   - 使用场景: 分类任务中，取概率最大的类别索引作为预测结果。
print("\nargmax (axis=1, 每行最大值的索引):", tf.argmax(r, axis=1).numpy())
print("argmin (axis=0, 每列最小值的索引):", tf.argmin(r, axis=0).numpy())


# --------------------------------------------------
# 5. 张量变形操作
# --------------------------------------------------
print("\n--- 张量变形操作 ---")

t = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
print("原始 1D 张量:", t.numpy(), "形状:", t.shape)

# 【API 详解】tf.reshape(tensor, shape, name=None)
#   - tensor: 输入张量。
#   - shape:  目标形状，可以是一个 list 或 tuple。
#             可以使用 -1 表示"自动推断该维度的大小"，
#             但最多只能有一个维度为 -1。
#             例如 tf.reshape(t, [2, -1]) 会自动计算第二维 = 总元素数 / 2 = 4。
#   - 返回值: 与输入元素总数相同但形状不同的新张量。
#   - 注意: reshape 不复制数据（大多数情况下），只是改变了张量的视图 (view)。
t_2x4 = tf.reshape(t, shape=(2, 4))
print("reshape 为 (2, 4):\n", t_2x4.numpy())

t_auto = tf.reshape(t, shape=(2, 2, -1))  # -1 自动推断为 2
print("reshape 为 (2, 2, -1) 自动推断:\n", t_auto.numpy())

# 【补充 API】tf.expand_dims(input, axis, name=None)
#   - 在指定位置插入一个大小为 1 的新维度。
#   - axis: 新维度的位置。正数表示从前往后数，负数表示从后往前数。
#   - 使用场景: 为数据添加批次维度 (batch) 或通道维度。
#     例如一个 shape=(224, 224, 3) 的图片，expand_dims(x, axis=0) 后变为 (1, 224, 224, 3)。
img = tf.random.normal(shape=(224, 224, 3))
print("\n原始图片 shape:", img.shape)
img_batched = tf.expand_dims(img, axis=0)
print("expand_dims(axis=0) 后 shape:", img_batched.shape)

# 【补充 API】tf.squeeze(input, axis=None, name=None)
#   - 移除大小为 1 的维度，与 expand_dims 互为逆操作。
#   - axis: 指定要移除的维度（该维度大小必须为 1）。若为 None，移除所有大小为 1 的维度。
#   - 使用场景: 去掉多余的批次维度或通道维度。
squeezed = tf.squeeze(img_batched)
print("squeeze 后 shape:", squeezed.shape)

# 【补充 API】tf.stack(values, axis=0, name=None)
#   - 将多个同形状的张量沿一个新维度堆叠起来。
#   - values: 一个张量列表。
#   - axis:   新维度插入的位置。
#   - 与 tf.concat 的区别: stack 增加一个新维度，concat 在已有维度上拼接。
a1 = tf.constant([1, 2, 3])
b1 = tf.constant([4, 5, 6])
c1 = tf.constant([7, 8, 9])
stacked = tf.stack([a1, b1, c1], axis=0)
print("\nstack (axis=0):\n", stacked.numpy(), "shape:", stacked.shape)

# 【补充 API】tf.concat(values, axis, name=None)
#   - 将多个张量沿指定维度拼接在一起。
#   - values: 张量列表，除拼接维度外其他维度大小必须相同。
#   - axis:   拼接的维度。
concatenated = tf.concat([a1, b1, c1], axis=0)
print("concat (axis=0):", concatenated.numpy(), "shape:", concatenated.shape)


# --------------------------------------------------
# 6. 比较与逻辑运算
# --------------------------------------------------
print("\n--- 比较与逻辑运算 ---")

x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
y = tf.constant([5.0, 4.0, 3.0, 2.0, 1.0])

# 【API 详解】逐元素比较运算
#   - tf.equal(x, y) / x == y:         逐元素判断是否相等，返回布尔张量。
#   - tf.not_equal(x, y) / x != y:     逐元素判断是否不等。
#   - tf.greater(x, y) / x > y:        逐元素判断 x > y。
#   - tf.greater_equal(x, y) / x >= y: 逐元素判断 x >= y。
#   - tf.less(x, y) / x < y:           逐元素判断 x < y。
#   - tf.less_equal(x, y) / x <= y:    逐元素判断 x <= y。
#   - 返回值: 与输入形状相同的布尔张量 (dtype=bool)。
#   - 使用场景: 生成掩码 (mask)、准确率计算 (predicted == label) 等。
print("x == y (tf.equal):", tf.equal(x, y).numpy())
print("x > y (tf.greater):", tf.greater(x, y).numpy())
print("x < 3 (tf.less):", tf.less(x, 3.0).numpy())

# 【补充 API】tf.where(condition, x=None, y=None)
#   - 当 x 和 y 都提供时: 根据 condition 逐元素选择 x 或 y。
#     condition 为 True 的位置取 x 的值，为 False 的位置取 y 的值。
#   - 当 x 和 y 都为 None 时: 返回 condition 中为 True 的元素的索引。
#   - 使用场景: 三元运算符的张量版本、非极大值抑制 (NMS)、根据条件过滤数据。
result = tf.where(x > 3.0, x, tf.zeros_like(x))
print("tf.where (x > 3.0 取 x, 否则取 0):", result.numpy())

indices = tf.where(x > 2.0)
print("tf.where (返回 x > 2.0 的索引):", indices.numpy().flatten())


# --------------------------------------------------
# 7. 类型转换
# --------------------------------------------------
print("\n--- 类型转换 ---")

# 【API 详解】tf.cast(x, dtype, name=None)
#   - x:    输入张量。
#   - dtype: 目标数据类型。
#   - 返回值: 与 x 形状相同但元素类型为 dtype 的新张量。
#   - 常见用法:
#       1. 浮点 → 整数: tf.cast(3.7, tf.int32) → 3（截断，不是四舍五入）
#       2. 整数 → 浮点: tf.cast(tf.constant([1, 2]), tf.float32) → [1.0, 2.0]
#       3. 布尔 → 整数: tf.cast(tf.constant([True, False]), tf.int32) → [1, 0]
#   - 使用场景: 损失计算前确保标签和预测值类型一致；将布尔掩码转为 0/1 用于乘法。
int_tensor = tf.constant([1, 2, 3])
float_tensor = tf.cast(int_tensor, tf.float32)
print("整数张量:", int_tensor.numpy(), "dtype:", int_tensor.dtype)
print("转换为浮点:", float_tensor.numpy(), "dtype:", float_tensor.dtype)

bool_tensor = tf.constant([True, False, True, False])
int_from_bool = tf.cast(bool_tensor, tf.int32)
print("布尔张量:", bool_tensor.numpy())
print("布尔 → 整数:", int_from_bool.numpy())  # [1, 0, 1, 0]

# 精度转换: float32 → float16 (用于混合精度训练，减少显存占用)
f32 = tf.constant([1.23456789, 9.87654321], dtype=tf.float32)
f16 = tf.cast(f32, tf.float16)
print("float32:", f32.numpy())
print("float16:", f16.numpy(), "(精度损失)")


# --------------------------------------------------
# 8. 综合示例: 用 TensorFlow 实现 L2 正则化损失计算
# --------------------------------------------------
print("\n--- 综合示例: L2 正则化损失 ---")

# 模拟一个简单的神经网络权重
weights = tf.Variable(tf.random.normal(shape=(4, 3)), name="dense_weights")
l2_lambda = 0.01  # 正则化系数

# L2 正则化: loss_l2 = lambda * Σ(w²)
# 步骤:
#   1. tf.square(weights):       逐元素平方
#   2. tf.reduce_sum(...):       所有元素求和
#   3. l2_lambda * ...:          乘以正则化系数
l2_loss = l2_lambda * tf.reduce_sum(tf.square(weights))

print("权重矩阵 (4x3):\n", weights.numpy())
print(f"L2 正则化系数: {l2_lambda}")
print(f"L2 正则化损失: {l2_loss.numpy():.6f}")

# 等价写法: 使用 tf.nn.l2_loss
# 【API 详解】tf.nn.l2_loss(t, name=None)
#   - 计算 L2 损失 = sum(t²) / 2（注意有除以 2）
#   - 返回值: 一个标量张量。
l2_loss_api = l2_lambda * 2 * tf.nn.l2_loss(weights)  # 乘以 2 补偿 API 中的 /2
print(f"L2 损失 (使用 tf.nn.l2_loss): {l2_loss_api.numpy():.6f}")
print(f"两种方法结果一致: {np.isclose(l2_loss.numpy(), l2_loss_api.numpy())}")
