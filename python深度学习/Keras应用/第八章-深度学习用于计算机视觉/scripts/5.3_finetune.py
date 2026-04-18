"""
5.3_finetune.py - 预训练模型微调（对应《Python深度学习》5.3节）

微调策略：
1. 加载已训练的特征提取模型
2. 解冻 VGG16 顶层卷积块 (block5_conv1~3)
3. 使用较小学习率联合训练

注意：微调必须在特征提取训练之后进行！
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
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

# 参数
IMG_SIZE = (150, 150)
BATCH_SIZE = 32


def finetune_vgg16():
    """
    微调 VGG16 模型

    步骤：
    1. 加载特征提取模型
    2. 解冻 block5_conv1~3
    3. 数据增强训练 50 epochs
    4. 评估测试集
    """
    print("\n" + "="*60)
    print("微调 VGG16 模型")
    print("="*60)

    # 加载特征提取模型
    model_path = MODELS_DIR / 'feature_extraction_with_aug.h5'
    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}")
        print("请先运行 5.3_pretrained_feature_extraction.py 的方法B")
        return None

    model = keras.models.load_model(model_path)
    print(f"[INFO] 已加载模型: {model_path}")

    # 查看 VGG16 卷积基
    conv_base = model.layers[0]
    print(f"\nVGG16 层结构:")
    for i, layer in enumerate(conv_base.layers):
        print(f"  {i}: {layer.name} - trainable={layer.trainable}")

    # 解冻 block5_conv1 及以上层
    # VGG16 层索引: block5_conv1 在索引 15-17
    conv_base.trainable = True

    # 冻结除 block5 以外的所有层
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True  # 从这里开始解冻
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    print(f"\n解冻后的可训练层:")
    trainable_count = 0
    for layer in conv_base.layers:
        if layer.trainable:
            trainable_count += 1
            print(f"  {layer.name}")

    print(f"\n[INFO] 可训练层数: {trainable_count} / {len(conv_base.layers)}")

    # 编译模型（使用更小的学习率）
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),  # 重要！学习率降低 10x
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
    test_datagen = ImageDataGenerator(rescale=1./255)

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

    test_generator = test_datagen.flow_from_directory(
        DATA_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # 测试集不打乱
    )

    # 训练
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=50,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE
    )

    # 绘制曲线（使用平滑）
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils import plot_training_curves, smooth_curve

    plot_training_curves(
        history,
        title='Fine-tuning VGG16 (block5_conv1~3)',
        save_path=str(RESULTS_DIR / 'finetune_curves.png')
    )

    # 评估测试集
    test_loss, test_acc = model.evaluate(
        test_generator,
        steps=test_generator.samples // BATCH_SIZE
    )

    print(f"\n测试集评估:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")

    # 保存微调后的模型
    finetuned_path = MODELS_DIR / 'finetuned_vgg16.h5'
    model.save(finetuned_path)
    print(f"[INFO] 微调模型已保存至: {finetuned_path}")

    return history, test_acc


if __name__ == "__main__":
    # 检查数据和模型
    if not (DATA_DIR / 'train').exists():
        print("[ERROR] 数据目录不存在，请先运行 download_data.py")
        exit(1)

    if not (MODELS_DIR / 'feature_extraction_with_aug.h5').exists():
        print("[ERROR] 请先运行 5.3_pretrained_feature_extraction.py (方法B)")
        exit(1)

    history, test_acc = finetune_vgg16()

    if history:
        print("\n" + "="*60)
        print("微调完成！")
        print("="*60)
        print(f"验证准确率峰值: {max(history.history['val_accuracy']):.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        print("\n微调效果：")
        print("- 特征提取约 92-96%")
        print("- 微调后可达 96-98%")