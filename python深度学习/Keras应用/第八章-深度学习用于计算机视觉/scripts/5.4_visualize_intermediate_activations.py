"""
5.4_visualize_intermediate_activations.py - 中间激活可视化（对应《Python深度学习》5.4.1节）

功能：
1. 加载训练好的模型
2. 输入一张测试图片
3. 提取前8个卷积/池化层的激活输出
4. 以网格形式可视化每层的所有通道

参考：书中图 5-27 风格
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import random

# 项目路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 参数
IMG_SIZE = (150, 150)


def load_test_image():
    """
    从测试集随机加载一张猫的图片

    Returns:
        img_array: 预处理后的图像数组 (1, 150, 150, 3)
        img_path: 图片路径
    """
    cats_dir = DATA_DIR / 'test' / 'cats'
    cat_images = list(cats_dir.glob('*.jpg'))

    if len(cat_images) == 0:
        raise FileNotFoundError("测试集无猫图片，请先运行 download_data.py")

    # 随机选择一张
    random.seed(42)
    img_path = random.choice(cat_images)

    # 加载并预处理
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加 batch 维度
    img_array /= 255.0  # 归一化

    print(f"[INFO] 加载图片: {img_path}")
    return img_array, img_path


def create_activation_model(model):
    """
    创建多输出模型，提取前8个卷积/池化层的激活

    Args:
        model: 原始训练好的模型

    Returns:
        activation_model: 多输出 keras Model
        layer_names: 层名称列表
    """
    # 找出所有卷积和池化层
    layer_outputs = []
    layer_names = []

    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.MaxPooling2D)):
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
            if len(layer_names) >= 8:  # 只取前8层
                break

    # 创建多输出模型
    activation_model = keras.Model(
        inputs=model.input,
        outputs=layer_outputs
    )

    print(f"[INFO] 激活层列表: {layer_names}")
    return activation_model, layer_names


def visualize_activations(activations, layer_names, img_path, save_path):
    """
    可视化每层的激活输出

    Args:
        activations: 各层激活输出列表
        layer_names: 层名称列表
        img_path: 输入图片路径
        save_path: 保存路径
    """
    # 每层显示 16 个通道（4x4 网格）
    images_per_row = 16

    # 创建大画布
    n_layers = len(layer_names)
    fig_height = n_layers * 3  # 每层高度

    plt.figure(figsize=(images_per_row * 1.5, fig_height))

    for layer_idx, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        n_features = layer_activation.shape[-1]  # 该层通道数
        size = layer_activation.shape[1]  # 特征图尺寸

        # 计算网格行数
        n_cols = images_per_row
        n_rows = n_features // n_cols
        if n_rows == 0:
            n_rows = 1
            n_cols = min(n_features, images_per_row)

        # 创建该层的网格
        display_grid = np.zeros((size * n_rows, size * n_cols))

        for col in range(n_cols):
            for row in range(n_rows):
                if row * n_cols + col < n_features:
                    channel_image = layer_activation[0, :, :, row * n_cols + col]

                    # 后处理：归一化到可视范围
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std() + 1e-5
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                    display_grid[row * size : (row + 1) * size,
                                 col * size : (col + 1) * size] = channel_image

        # 绘制该层
        scale = 1. / size
        plt.subplot(n_layers, 1, layer_idx + 1)
        plt.title(f'{layer_name} ({n_features} channels)')
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.axis('off')

    plt.suptitle(f'Intermediate Activations\nInput: {Path(img_path).name}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 可视化已保存: {save_path}")


def main():
    """
    主函数：加载模型 → 输入图片 → 可视化激活
    """
    print("\n" + "="*60)
    print("中间激活可视化")
    print("="*60)

    # 加载模型
    model_path = MODELS_DIR / 'cats_and_dogs_small_2.h5'
    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}")
        print("请先运行 5.2_train_from_scratch.py (实验2)")
        return

    model = keras.models.load_model(model_path)
    print(f"[INFO] 加载模型: {model_path}")
    print(f"模型结构:")
    model.summary()

    # 加载测试图片
    img_array, img_path = load_test_image()

    # 创建激活模型
    activation_model, layer_names = create_activation_model(model)

    # 获取激活输出
    activations = activation_model.predict(img_array, verbose=0)

    print(f"\n各层激活形状:")
    for name, act in zip(layer_names, activations):
        print(f"  {name}: {act.shape}")

    # 可视化
    save_path = RESULTS_DIR / 'intermediate_activations.png'
    visualize_activations(activations, layer_names, img_path, save_path)

    print("\n可视化说明:")
    print("- 第1层：保留原始图像大部分信息")
    print("- 中间层：开始抽象，提取边缘、纹理等特征")
    print("- 后续层：信息越来越抽象，空间信息减少")


if __name__ == "__main__":
    # 检查数据和模型
    if not (DATA_DIR / 'test').exists():
        print("[ERROR] 数据目录不存在，请先运行 download_data.py")
        exit(1)

    if not (MODELS_DIR / 'cats_and_dogs_small_2.h5').exists():
        print("[ERROR] 请先运行 5.2_train_from_scratch.py")
        exit(1)

    main()