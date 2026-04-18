"""
5.3_pretrained_feature_extraction.py - 预训练模型特征提取（对应《Python深度学习》5.3节）

方法A：快速特征提取（不含数据增强）
- 用 VGG16 卷积基提取特征，存为 .npy 文件
- 在顶部训练小型 Dense 分类器

方法B：带数据增强的端到端特征提取
- VGG16 卷积基（冻结）+ Dense 分类器
- 使用 ImageDataGenerator 数据增强训练
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# 项目路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
FEATURES_DIR = PROJECT_ROOT / 'features'

# 创建必要目录
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)

# 参数
IMG_SIZE = (150, 150)
BATCH_SIZE = 20  # 特征提取时用较小批量


def extract_features_vgg16(sample_count, generator, save_prefix):
    """
    使用 VGG16 卷积基提取特征并保存为 .npy 文件

    Args:
        sample_count: 样本数量
        generator: ImageDataGenerator.flow_from_directory 生成的迭代器
        save_prefix: 保存文件的前缀 (如 'train', 'validation')

    Returns:
        features_path, labels_path: 保存的文件路径
    """
    print(f"[INFO] 提取 {save_prefix} 集 {sample_count} 个样本的特征...")

    # 加载 VGG16 卷积基（不含顶部的 Dense 层）
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    # VGG16 输出 shape: (batch, 4, 4, 512)
    # 展平后: batch × 4 × 4 × 512 = batch × 8192
    features_shape = (sample_count, 4, 4, 512)
    labels_shape = (sample_count,)

    features = np.zeros(features_shape)
    labels = np.zeros(labels_shape)

    # 批量提取
    batch_count = sample_count // BATCH_SIZE
    for i, (batch_images, batch_labels) in enumerate(generator):
        if i >= batch_count:
            break
        features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = conv_base.predict(batch_images, verbose=0)
        labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = batch_labels
        print(f"  已处理: {(i + 1) * BATCH_SIZE}/{sample_count}")

    # 保存
    features_path = FEATURES_DIR / f'{save_prefix}_features.npy'
    labels_path = FEATURES_DIR / f'{save_prefix}_labels.npy'
    np.save(features_path, features)
    np.save(labels_path, labels)

    print(f"[INFO] 特征已保存: {features_path}")
    print(f"[INFO] 标签已保存: {labels_path}")

    return features_path, labels_path


def method_a_fast_feature_extraction():
    """
    方法A：快速特征提取（不含数据增强）

    步骤：
    1. VGG16 提取特征 → .npy
    2. 构建小型 Dense 分类器
    3. 训练 30 epochs
    """
    print("\n" + "="*60)
    print("方法A：快速特征提取（不含数据增强）")
    print("="*60)

    # 数据生成器（仅归一化）
    datagen = ImageDataGenerator(rescale=1./255)

    # 训练集生成器
    train_generator = datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # 验证集生成器
    val_generator = datagen.flow_from_directory(
        DATA_DIR / 'validation',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # 提取特征
    train_feat_path, train_label_path = extract_features_vgg16(
        sample_count=2000,
        generator=train_generator,
        save_prefix='train'
    )

    val_feat_path, val_label_path = extract_features_vgg16(
        sample_count=1000,
        generator=val_generator,
        save_prefix='validation'
    )

    # 加载特征
    train_features = np.load(train_feat_path)
    train_labels = np.load(train_label_path)
    val_features = np.load(val_feat_path)
    val_labels = np.load(val_label_path)

    # 展平特征: (N, 4, 4, 512) → (N, 8192)
    train_features = train_features.reshape(2000, 4 * 4 * 512)
    val_features = val_features.reshape(1000, 4 * 4 * 512)

    # 构建 Dense 分类器
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(8192,)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='feature_extraction_dense_classifier')

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 训练
    history = model.fit(
        train_features, train_labels,
        epochs=30,
        batch_size=32,
        validation_data=(val_features, val_labels)
    )

    # 绘制曲线
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils import plot_training_curves

    plot_training_curves(
        history,
        title='Method A: Fast Feature Extraction (VGG16 + Dense)',
        save_path=str(RESULTS_DIR / 'feature_extraction_fast.png')
    )

    print(f"\n最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"验证准确率峰值: {max(history.history['val_accuracy']):.4f}")

    return history


def method_b_feature_extraction_with_augmentation():
    """
    方法B：带数据增强的端到端特征提取

    步骤：
    1. 构建 Sequential: VGG16(冻结) + Flatten + Dense(256) + Dense(1)
    2. 使用 ImageDataGenerator 数据增强
    3. 训练 50 epochs
    """
    print("\n" + "="*60)
    print("方法B：带数据增强的端到端特征提取")
    print("="*60)

    # 加载 VGG16 卷积基
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    # 冻结卷积基（关键！防止预训练权重被破坏）
    conv_base.trainable = False
    print(f"[INFO] VGG16 冻结层数: {len(conv_base.layers)}")
    print(f"[INFO] 可训练参数: {conv_base.trainable_variables}")

    # 构建完整模型
    model = keras.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='feature_extraction_with_augmentation')

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 数据增强生成器
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        DATA_DIR / 'validation',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary'
    )

    # 训练
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=50,
        validation_data=val_generator,
        validation_steps=val_generator.samples // 32
    )

    # 保存模型
    model_path = MODELS_DIR / 'feature_extraction_with_aug.h5'
    model.save(model_path)
    print(f"[INFO] 模型已保存至: {model_path}")

    # 绘制曲线
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils import plot_training_curves

    plot_training_curves(
        history,
        title='Method B: Feature Extraction with Data Augmentation',
        save_path=str(RESULTS_DIR / 'feature_extraction_with_aug.png')
    )

    print(f"\n最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"验证准确率峰值: {max(history.history['val_accuracy']):.4f}")

    return history


if __name__ == "__main__":
    # 检查数据
    if not (DATA_DIR / 'train').exists():
        print("[ERROR] 数据目录不存在，请先运行 download_data.py")
        exit(1)

    # 运行两种方法
    history_a = method_a_fast_feature_extraction()
    history_b = method_b_feature_extraction_with_augmentation()

    print("\n" + "="*60)
    print("特征提取实验完成！")
    print("="*60)
    print(f"方法A 验证准确率峰值: {max(history_a.history['val_accuracy']):.4f}")
    print(f"方法B 验证准确率峰值: {max(history_b.history['val_accuracy']):.4f}")
    print("\n对比说明：")
    print("- 方法A 快速但可能过拟合，约 90%")
    print("- 方法B 数据增强后更稳健，约 92-96%")