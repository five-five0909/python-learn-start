"""
5.2_train_from_scratch.py - 从零训练 CNN（对应《Python深度学习》5.2节）

实验设置：
- 实验1：小型VGG风格CNN，无数据增强，30 epochs
- 实验2：同模型 + 数据增强 + Dropout，50 epochs

数据：4000张猫狗图片，150x150像素
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# 设置随机种子保证可复现性
tf.random.set_seed(42)
np.random.seed(42)

# 项目路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 创建必要目录
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 图像尺寸和批量大小
IMG_SIZE = (150, 150)
BATCH_SIZE = 32


def build_small_vgg_model(with_dropout=False):
    """
    构建小型 VGG 风格 CNN

    结构：Conv2D→MaxPooling 堆叠 4 次 → Flatten → Dense
    使用 Dropout 时在 Dense 层前加入

    Args:
        with_dropout: 是否添加 Dropout(0.5)

    Returns:
        keras.Model
    """
    model = keras.Sequential(name='cats_and_dogs_small_vgg')

    # 第一个卷积块
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # 第二个卷积块
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 第三个卷积块
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 第四个卷积块
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten 和 Dense 层
    model.add(layers.Flatten())

    if with_dropout:
        model.add(layers.Dropout(0.5))  # Dropout 防止过拟合

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # 二分类输出

    return model


def train_experiment1():
    """
    实验1：无数据增强的基线训练

    使用简单的 ImageDataGenerator 仅做归一化
    """
    print("\n" + "="*60)
    print("实验1：从零训练 CNN（无数据增强）")
    print("="*60)

    # 构建模型
    model = build_small_vgg_model(with_dropout=False)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 数据生成器（仅归一化）
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        DATA_DIR / 'validation',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # 训练
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=30,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE
    )

    # 保存模型
    model_path = MODELS_DIR / 'cats_and_dogs_small_1.h5'
    model.save(model_path)
    print(f"[INFO] 模型已保存至: {model_path}")

    # 绘制训练曲线
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils import plot_training_curves

    plot_training_curves(
        history,
        title='Experiment 1: Training from Scratch (No Augmentation)',
        save_path=str(RESULTS_DIR / 'training_curves_exp1.png')
    )

    # 打印最终结果
    print(f"\n最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"验证准确率峰值: {max(history.history['val_accuracy']):.4f}")

    return history


def train_experiment2():
    """
    实验2：带数据增强和 Dropout 的训练

    使用 ImageDataGenerator 的多种增强技术：
    - rotation_range: 随机旋转角度
    - width_shift/height_shift: 随机平移
    - shear_range: 随机剪切
    - zoom_range: 随机缩放
    - horizontal_flip: 随机水平翻转
    """
    print("\n" + "="*60)
    print("实验2：从零训练 CNN（数据增强 + Dropout）")
    print("="*60)

    # 构建模型（带 Dropout）
    model = build_small_vgg_model(with_dropout=True)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 数据生成器（带增强）
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,       # 随机旋转 0-40 度
        width_shift_range=0.2,   # 随机水平平移 20%
        height_shift_range=0.2,  # 随机垂直平移 20%
        shear_range=0.2,         # 随机剪切变换
        zoom_range=0.2,          # 随机缩放
        horizontal_flip=True,    # 随机水平翻转
        fill_mode='nearest'      # 填充模式
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        DATA_DIR / 'validation',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # 训练（更多 epochs，因为增强后需要更多迭代）
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=50,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE
    )

    # 保存模型
    model_path = MODELS_DIR / 'cats_and_dogs_small_2.h5'
    model.save(model_path)
    print(f"[INFO] 模型已保存至: {model_path}")

    # 绘制训练曲线
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils import plot_training_curves

    plot_training_curves(
        history,
        title='Experiment 2: Training with Data Augmentation + Dropout',
        save_path=str(RESULTS_DIR / 'training_curves_exp2.png')
    )

    # 打印最终结果
    print(f"\n最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"验证准确率峰值: {max(history.history['val_accuracy']):.4f}")

    return history


if __name__ == "__main__":
    # 检查数据是否存在
    if not (DATA_DIR / 'train').exists():
        print("[ERROR] 数据目录不存在，请先运行 download_data.py")
        exit(1)

    # 运行两个实验
    history1 = train_experiment1()
    history2 = train_experiment2()

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"实验1 验证准确率峰值: {max(history1.history['val_accuracy']):.4f}")
    print(f"实验2 验证准确率峰值: {max(history2.history['val_accuracy']):.4f}")
    print("\n对比说明：")
    print("- 实验1无增强，易过拟合，验证准确率约 70-75%")
    print("- 实验2有增强+Dropout，泛化更好，验证准确率约 82-85%")