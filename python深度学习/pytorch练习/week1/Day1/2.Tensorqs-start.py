# 导入 PyTorch 和 NumPy 库
import torch
import numpy as np

# =================================================================
# 1. 初始化张量 (Initializing a Tensor)
# =================================================================
# 张量有多种创建方式，以下是几种最常用的。

# --- 方式 1: 直接从 Python 列表数据创建 ---
# 数据类型会被自动推断，比如这里会被推断为 int64
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# 维度应该是(2,2)
print(f"从列表创建的张量:\n{x_data}\n")

# --- 方式 2: 从 NumPy 数组创建 ---
# 这是一种在 PyTorch 和 NumPy 之间进行转换的非常方便的方法
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# 维度也是(2,2)
print(f"从NumPy数组创建的张量:\n{x_np}\n")

# --- 方式 3: 从另一个张量创建 (保留属性) ---
# 新张量会保留原张量的形状和数据类型，除非你明确指定去覆盖它们。
# 创建一个和 x_data 形状、数据类型相同，但全是 1 的张量
x_ones = torch.ones_like(x_data)
print(f"保留属性的 Ones 张量:\n{x_ones}\n")

# 创建一个和 x_data 形状相同，但数据类型为浮点型的随机张量
# 这里我们用 dtype 参数来覆盖原来的整数类型
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"覆盖数据类型的 Random 张量:\n{x_rand}\n")

# --- 方式 4: 使用随机值或常量值，直接指定形状 ---
# shape 参数是一个元组，用来定义输出张量的维度。
shape = (2, 3,) # 创建一个 2 行 3 列的张量

rand_tensor = torch.rand(shape)    # 随机张量，元素值服从 [0, 1) 区间的均匀分布
ones_tensor = torch.ones(shape)    # 常量张量，所有元素都是 1
zeros_tensor = torch.zeros(shape)  # 常量张量，所有元素都是 0

print(f"随机张量 (2x3):\n{rand_tensor}\n")
print(f"Ones 张量 (2x3):\n{ones_tensor}\n")
print(f"Zeros 张量 (2x3):\n{zeros_tensor}\n")

# --- 方式 5: 正态分布初始化（常用于权重矩阵） ---
# 直接用 torch.randn 生成标准正态分布 N(0,1) 的张量
weight_matrix_1 = torch.randn(3, 4)
print(f"标准正态分布 (直接创建):\n{weight_matrix_1}\n")

# 或者先创建一个张量，再用 nn.init.normal_ 原地修改它的值
# 比如创建一个全零的 3x4 张量，然后用均值为 0，标准差为 0.1 的正态分布覆盖
weight_matrix_2 = torch.zeros(3, 4)
torch.nn.init.normal_(weight_matrix_2, mean=0.0, std=0.1)
print(f"使用 nn.init.normal_ (mean=0, std=0.1):\n{weight_matrix_2}\n")

# =================================================================
# 2. 张量的属性 (Attributes of a Tensor)
# =================================================================
# 张量有三个最基本的属性：形状、数据类型和所在的设备。
tensor = torch.rand(3, 4) # 创建一个 3x4 的随机张量作为示例

print(f"张量的形状: {tensor.shape}")    # 输出: torch.Size([3, 4])
print(f"张量的数据类型: {tensor.dtype}") # 输出: torch.float32
print(f"张量存储的设备: {tensor.device}")# 输出: cpu (默认在CPU上)
print("-" * 30)



# =================================================================
# 3. 张量的操作 (Operations on Tensors)
# =================================================================
# PyTorch 提供了超过 1200 种张量运算，涵盖算术、线性代数、矩阵操作等。
# 默认情况下，张量是在 CPU 上创建的。以下演示如何将其移到 GPU 上。
# 注意：如果你的机器没有可用的加速器，这段代码会跳过移动操作。
if torch.accelerator.is_available():
    # 如果有 GPU，就把它移到 GPU 上
    device = torch.accelerator.current_accelerator()
    tensor = tensor.to(device)
    print(f"张量已移动到: {device}")
else:
    print("未检测到加速器，张量保留在 CPU 上。")

#  直接在创建张量时指定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = torch.rand(4, 4, device=device)  # 直接在 GPU 上创建

# --- 索引与切片 (Indexing and Slicing) ---
# 和 NumPy 一样，你可以用标准的 Python 切片语法来操作张量。
tensor = torch.ones(4, 4) # 创建一个 4x4，全为 1 的张量
print(f"\n原始张量:\n{tensor}")

# 修改第 2 列（索引为 1）的所有行为 0
# 切片语法说明：[start:stop:step] start：起始索引（包含）；stop：结束索引（不包含）；step：步长（每隔几个取一个）；如果某个位置不写，就会使用默认值：
# 原则上，你有 N 维，就应该写 N 个索引（逗号分隔），每个索引可以是整数、切片或单个值。如果想省事，可以用 ... 把连续不操作的维度全部包掉，让代码更简洁
tensor[:, 1] = 0
print(f"将第2列置为0后:\n{tensor}")

# 打印第一行、第一列和最后一列
print(f"第一行: {tensor[0]}")
print(f"第一列: {tensor[:, 0]}")
print(f"最后一列: {tensor[..., -1]}")

# 切片综合训练
x = torch.arange(40).reshape(2,4,5)

print(f"x[:, :, 0]结果: {x[:, :, 0]}")     # 所有通道，所有行，第 0 列 → 形状 (2,4)
print(f"x[0, :, 1:4]结果: {x[0, :, 1:4]}")   # 第 0 个通道，所有行，列 1 到 3 → (4,3)
print(f"x[:, ::2, -1]结果: {x[:, ::2, -1]}")     # 所有通道，每隔一行取，最后一列 → (2,2)

# --- 使用省略号 ... 的等价写法 ---
# ... 代表“尽可能多的冒号”，自动填充剩下的维度
print(f"x[..., 0]结果: {x[..., 0]}")          # 等价于 x[:, :, 0]，形状 (2,4)
print(f"x[0, ..., 1:4]结果: {x[0, ..., 1:4]}")     # 等价于 x[0, :, 1:4]，形状 (4,3)
# 注意：x[:, ::2, -1] 已经精确用了3个维度，无法再用 ... 简化


# --- 连接张量 (Concatenating Tensors) ---
# 使用 torch.cat 可以沿某个维度连接一系列张量。
# 原来的张量（4x4，全1矩阵，但其中第2列被置为0了）
tensor = torch.ones(4, 4)
tensor[:, 1] = 0

# --- 行方向拼接 (dim=0) ---
# dim=0 表示沿第0维（行）方向拼接，会增加行数，列数保持不变
t_cat_rows = torch.cat([tensor, tensor, tensor], dim=0)
print(f"沿行方向(dim=0)拼接后的张量 (形状 {t_cat_rows.shape}):\n{t_cat_rows}\n")

# --- 列方向拼接 (dim=1) ---
# dim=1 表示沿第1维（列）方向拼接，会增加列数，行数保持不变
t_cat_cols = torch.cat([tensor, tensor, tensor], dim=1)
print(f"沿列方向(dim=1)拼接后的张量 (形状 {t_cat_cols.shape}):\n{t_cat_cols}")

# --- 算术运算 (Arithmetic Operations) ---
# 计算两个张量的矩阵乘法 (y1, y2, y3 结果都是一样的)
y1 = tensor @ tensor.T         # @ 运算符表示矩阵乘法
y2 = tensor.matmul(tensor.T)   # .matmul 方法，同上

y3 = torch.rand_like(y1)       # 先创建一个形状相同的结果占位张量
torch.matmul(tensor, tensor.T, out=y3) # 将结果存入 y3

print(f"\n矩阵乘法结果 (y1):\n{y1}")
print(f"矩阵乘法结果 (y3):\n{y3}")

# 计算元素级别的乘法 (z1, z2, z3 结果一致)
z1 = tensor * tensor      # * 运算符表示元素乘法
z2 = tensor.mul(tensor)   # .mul 方法
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f"元素乘法结果 (z1):\n{z1}")

# --- 单元素张量 (Single-element Tensor) ---
# 如果你有一个单元素张量，可以使用 .item() 方法将其转换为 Python 数值
agg = tensor.sum()         # 对所有元素求和，得到单元素张量
agg_item = agg.item()      # 转换为 Python 的 float 或 int
print(f"\n求和得到的张量: {agg}, 其 Python 值是: {agg_item}, 类型是: {type(agg_item)}")

# --- 原地操作 (In-place Operations) ---
# 原地操作会直接把结果存到原张量里，通常有下划线后缀，如 add_(), copy_()
print(f"\n原地操作前的原始张量:\n{tensor}")
tensor.add_(5) # 每个元素加 5，结果直接替换原 tensor
print(f"执行 tensor.add_(5) 后:\n{tensor}")
# 注意：原地操作虽然节省内存，但可能会在梯度计算时引起问题，官方不推荐使用。

# =================================================================
# 4. 与 NumPy 桥接 (Bridge with NumPy)
# =================================================================
# CPU 上的张量和 NumPy 数组可以共享同一块底层内存，修改一个会直接影响另一个。

# --- 张量 -> NumPy 数组 (Tensor to NumPy Array) ---
t = torch.ones(5)
print(f"\n张量 t: {t}")
n = t.numpy() # 将张量 t 转换为 NumPy 数组 n
print(f"NumPy 数组 n: {n}")

# 修改张量 t，观察 NumPy 数组 n 是否也变化
t.add_(1) # 张量 t 中所有值加 1
print(f"t 加 1 后: {t}")
print(f"n 也变成了: {n}") # n 的值也会跟着改变，因为它们共享内存

# --- NumPy 数组 -> 张量 (NumPy Array to Tensor) ---
n = np.ones(5)
print(f"\nNumPy 数组 n: {n}")
t = torch.from_numpy(n) # 将 NumPy 数组 n 转换为张量 t
print(f"转换后的张量 t: {t}")

# 修改 NumPy 数组 n，观察张量 t 是否也变化
np.add(n, 1, out=n) # NumPy 数组 n 中所有值加 1
print(f"n 加 1 后: {n}")
print(f"t 也变成了: {t}") # t 也会跟着改变