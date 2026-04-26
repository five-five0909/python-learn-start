import torch

# ============================================================================
# 张量广播（Broadcasting）示例
# ============================================================================
# 广播是 PyTorch/NumPy 在不同形状的张量进行运算时，自动扩展较小张量的机制。
# 规则（从后往前比较两个张量的形状）：
#   1. 对应维度大小相等
#   2. 其中一个维度大小为 1
#   3. 其中一个张量缺少该维度（会补成 1）
# 如果都不满足，就会报错。

# 如果两个张量维度数不同，先在维度少的张量最左边补 1，直到维数一样，再逐维比较。
# 例如 (2,4,5) 和 (1,3)：
# 先把 (1,3) 左边补 1 变成 (1,1,3)
# 再对齐：
# (2, 4, 5)
# (1, 1, 3)
# 第一维：2 vs 1 ✅（有 1）
# 第二维：4 vs 1 ✅（有 1）
# 第三维：5 vs 3 ❌（不等且都不是 1）→ 不能广播

print("=" * 50)
print("1. 标量与张量相加（最简单的广播）")
print("=" * 50)
a = torch.ones(3, 4)        # 形状 (3, 4)
b = 5                       # 标量，相当于形状 ()
c = a + b                   # 标量被广播成形状 (3, 4) 的满 5 矩阵
print(f"a 形状: {a.shape}")
print(f"b 是标量: {b}")
print(f"a + b 结果:\n{c}\n")


print("=" * 50)
print("2. 形状 (3, 4) 与 (4,) 相加 —— 一维向量广播成二维")
print("=" * 50)
a = torch.ones(3, 4)                     # (3, 4)
b = torch.tensor([1.0, 2.0, 3.0, 4.0])  # (4,)
c = a + b
# 规则检查（从最后一维向前比）：
#   最后一维：4 vs 4 → 相等，OK
#   倒数第二维：a 有维度 3，b 没有该维度 → 视为 b 有维度 1，广播成 3
# 结果形状 (3, 4)：b 在每一行都加上 [1, 2, 3, 4]
print(f"a 形状: {a.shape}，b 形状: {b.shape}")
print(f"a + b 结果:\n{c}\n")


print("=" * 50)
print("3. 形状 (3, 1) 与 (1, 4) 相加 —— 两边都是 1，相互广播")
print("=" * 50)
a = torch.ones(3, 1)   # (3, 1)
b = torch.ones(1, 4)   # (1, 4)
c = a + b               # 结果形状 (3, 4)
# 规则检查：
#   最后一维：1 vs 4 → a 的 1 广播成 4
#   倒数第二维：3 vs 1 → b 的 1 广播成 3
print(f"a 形状: {a.shape}，b 形状: {b.shape}")
print(f"a + b 结果形状: {c.shape}\n")


print("=" * 50)
print("4. 图像数据批次减去均值（常见实战）")
print("=" * 50)
# 假设一个批次 64 张 3 通道 32x32 的图片
images = torch.randn(64, 3, 32, 32)           # (B, C, H, W)
# 每个通道的均值，形状 (3,)
mean = torch.tensor([0.485, 0.456, 0.406])
# 要让 mean 正确广播到 (64, 3, 32, 32)，需要将其形状调整为 (1, 3, 1, 1)
# 或者直接用 view 调整：mean.view(1, 3, 1, 1) 再相减
images_normalized = images - mean.view(1, 3, 1, 1)
# 广播过程：
#   (64, 3, 32, 32)
# - ( 1, 3,  1,  1)
#   → mean 在第 0,2,3 维上自动扩展，不会实际复制内存
print(f"images 形状: {images.shape}")
print(f"mean 形状: {mean.shape}")
print(f"调整后 mean 形状: {mean.view(1,3,1,1).shape}")
print(f"归一化结果形状: {images_normalized.shape}\n")


print("=" * 50)
print("5. 广播失败的例子")
print("=" * 50)
a = torch.ones(3, 4)   # (3, 4)
b = torch.ones(2, 4)   # (2, 4)
print(f"a 形状: {a.shape}，b 形状: {b.shape}")
print("尝试 a + b 会报错，因为第一个维度 3 和 2 不相等，也没有一个是 1。")
# 取消下面这行注释即可看到报错：
# c = a + b   # RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0


print("=" * 50)
print("总结：广播让你可以用简洁的代码实现隐式形状扩展，但需要时刻注意规则。")
print("=" * 50)