"""
utils.py - 猫狗分类项目通用工具函数
《Python深度学习》第5章配套代码
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def plot_training_curves(history, title: str, save_path: str):
    """
    绘制训练过程的 accuracy + loss 四宫格曲线图

    Args:
        history: keras.callbacks.History 对象，包含训练历史数据
        title: 图片标题
        save_path: 保存路径
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 10))

    # Accuracy 子图
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss 子图
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 平滑后的 Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, smooth_curve(acc), 'b-', label='Smoothed training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'r-', label='Smoothed validation acc')
    plt.title('Smoothed Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 平滑后的 Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, smooth_curve(loss), 'b-', label='Smoothed training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'r-', label='Smoothed validation loss')
    plt.title('Smoothed Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 训练曲线已保存至: {save_path}")


def smooth_curve(points, factor=0.8):
    """
    指数移动平均平滑，用于绘制更清晰的曲线

    消除训练过程中的噪声波动，使趋势更明显

    Args:
        points: 原始数据点列表
        factor: 平滑因子，越大则平滑程度越高

    Returns:
        smoothed_points: 平滑后的数据点列表
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def load_and_preprocess_image(img_path: str, target_size=(150, 150)):
    """
    加载图像并做归一化预处理

    Args:
        img_path: 图像文件路径
        target_size: 目标尺寸 (height, width)

    Returns:
        img_array: 预处理后的图像数组，shape=(1, H, W, 3)，值域[0, 1]
    """
    img = Image.open(img_path)
    img = img.convert('RGB')  # 确保三通道
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # 归一化到 [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # 添加 batch 维度
    return img_array


def get_project_root():
    """
    获取项目根目录（基于当前脚本位置）

    Returns:
        Path: 项目根目录路径
    """
    # utils.py 位于项目根目录
    return Path(__file__).resolve().parent