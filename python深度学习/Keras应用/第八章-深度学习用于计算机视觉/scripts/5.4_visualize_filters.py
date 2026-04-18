"""
5.4_visualize_filters.py - 卷积滤波器可视化（对应《Python深度学习》5.4.2节）

功能：
1. 加载 VGG16 卷积基
2. 使用梯度上升法生成滤波器响应最大的输入图像
3. 对 block1~4 的 conv1 层各生成前 64 个 filter 的可视化网格

参考：书中图 5-29~32 风格
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from pathlib import Path

# 项目路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# 参数
IMG_SIZE = 150


def generate_pattern(layer_name, filter_index, size=150, steps=40, step_size=1.):
    """
    使用梯度上升法生成滤波器响应最大的输入图像

    思路：
    - 构建损失函数：指定滤波器的激活值均值
    - 计算输入图像梯度
    - 沿梯度方向更新图像，最大化滤波器响应

    Args:
        layer_name: 层名称
        filter_index: 滤波器索引
        size: 生成图像尺寸
        steps: 梯度上升迭代次数
        step_size: 每步更新幅度

    Returns:
        img: 生成的可视化图像 (size, size, 3)
    """
    # 加载 VGG16 卷积基
    model = VGG16(weights='imagenet', include_top=False)

    # 获取指定层的输出
    layer_output = model.get_layer(layer_name).output

    # 构建损失模型：输出指定滤波器的激活均值
    loss_model = keras.Model(
        inputs=model.input,
        outputs=layer_output[:, :, :, filter_index]  # 取指定通道
    )

    # 初始化随机噪声图像
    img = tf.random.uniform((1, size, size, 3), minval=0, maxval=255)
    img = tf.Variable(img, dtype=tf.float32)

    # 梯度上升迭代
    for i in range(steps):
        with tf.GradientTape() as tape:
            # 损失：滤波器激活值的均值（最大化）
            loss = tf.reduce_mean(loss_model(img))

        # 计算梯度
        grads = tape.gradient(loss, img)

        # 归一化梯度（防止梯度爆炸）
        grads = tf.math.l2_normalize(grads)

        # 更新图像（沿梯度上升方向）
        img.assign_add(grads * step_size)

        # 限制像素值范围
        img.assign(tf.clip_by_value(img, 0, 255))

    # 后处理：归一化到 [0, 255]
    img_array = img.numpy()[0]
    img_array -= img_array.mean()
    img_array /= img_array.std() + 1e-5
    img_array *= 64 + 128
    img_array = np.clip(img_array, 0, 255).astype('uint8')

    return img_array


def visualize_filters_for_layer(layer_name, n_filters=64, size=150):
    """
    可视化某一层的多个滤波器

    Args:
        layer_name: 层名称
        n_filters: 要可视化的滤波器数量
        size: 每个可视化图像的尺寸

    Returns:
        grid: 网格图像
    """
    print(f"[INFO] 生成 {layer_name} 层的滤波器可视化...")

    margin = 5  # 网格间距
    n_cols = 8
    n_rows = n_filters // n_cols

    # 创建网格画布
    grid_size = n_cols * size + (n_cols - 1) * margin
    grid_height = n_rows * size + (n_rows - 1) * margin
    grid = np.zeros((grid_height, grid_size, 3), dtype='uint8')

    for i in range(n_filters):
        print(f"  生成 filter {i+1}/{n_filters}...")
        filter_img = generate_pattern(layer_name, i, size=size)

        # 计算位置
        row = i // n_cols
        col = i % n_cols

        y_start = row * (size + margin)
        x_start = col * (size + margin)

        grid[y_start:y_start+size, x_start:x_start+size] = filter_img

    return grid


def main():
    """
    主函数：对多个层的滤波器进行可视化
    """
    print("\n" + "="*60)
    print("卷积滤波器可视化（梯度上升法）")
    print("="*60)

    RESULTS_DIR.mkdir(exist_ok=True)

    # 要可视化的层
    layers_to_visualize = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1'
    ]

    for layer_name in layers_to_visualize:
        print(f"\n处理 {layer_name}...")
        grid = visualize_filters_for_layer(layer_name, n_filters=64, size=150)

        # 保存
        save_path = RESULTS_DIR / f'filter_visualization_{layer_name}.png'
        plt.figure(figsize=(12, 12))
        plt.imshow(grid)
        plt.title(f'Filters in {layer_name}', fontsize=14)
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 已保存: {save_path}")

    print("\n可视化说明:")
    print("- block1_conv1: 简单的边缘、颜色检测")
    print("- block2_conv1: 开始组合边缘，形成纹理")
    print("- block3_conv1: 更复杂的纹理、形状")
    print("- block4_conv1: 类似于自然图像片段的模式")


if __name__ == "__main__":
    main()