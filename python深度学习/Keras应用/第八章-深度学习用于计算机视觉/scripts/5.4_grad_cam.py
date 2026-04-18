"""
5.4_grad_cam.py - Grad-CAM 类激活热力图（对应《Python深度学习》5.4.3节）

功能：
1. 加载完整 VGG16（含分类头）
2. 实现 Grad-CAM 算法
3. 生成热力图并叠加到原图

Grad-CAM 原理：
- 计算最后一个卷积层相对于预测类别的梯度
- 使用梯度加权激活图，生成类激活热力图
- 热力图高亮显示模型"关注的区域"

参考：书中图 5-35 风格
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from pathlib import Path

# 项目路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

RESULTS_DIR.mkdir(exist_ok=True)


def load_elephant_image():
    """
    加载非洲象图片

    可以从本地加载或使用预设 URL

    Returns:
        img_array: 预处理后的图像数组 (1, 224, 224, 3)
        original_img: 原始图像数组 (H, W, 3)
    """
    # 预设的示例图片路径（用户需自行准备或下载）
    elephant_path = PROJECT_ROOT / 'sample_images' / 'elephant.jpg'

    if elephant_path.exists():
        # 从本地加载
        img = image.load_img(elephant_path, target_size=(224, 224))
        original_img = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(original_img, axis=0))
        print(f"[INFO] 从本地加载图片: {elephant_path}")
    else:
        # 使用 TensorFlow 示例图片
        print("[INFO] 本地无示例图片，使用 TensorFlow 内置示例...")
        import urllib.request

        # 非洲象图片 URL
        elephant_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/African_Elephant.jpg/800px-African_Elephant.jpg'

        img_path = RESULTS_DIR / 'elephant_sample.jpg'
        urllib.request.urlretrieve(elephant_url, img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        original_img = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(original_img, axis=0))
        print(f"[INFO] 下载图片: {img_path}")

    return img_array, original_img


def generate_grad_cam(model, img_array, last_conv_layer_name='block5_conv3'):
    """
    生成 Grad-CAM 热力图

    步骤：
    1. 获取预测类别
    2. 计算最后一个卷积层的激活和梯度
    3. 使用梯度权重加权激活图
    4. 生成热力图

    Args:
        model: VGG16 模型
        img_array: 预处理后的输入图像
        last_conv_layer_name: 最后一个卷积层名称

    Returns:
        heatmap: 热力图 (H, W)，值域 [0, 1]
        preds: 模型预测结果
    """
    # 创建模型：同时输出最后卷积层激活和预测
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # 计算梯度
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # 获取预测类别的输出值
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 计算梯度：最后卷积层相对于预测类别的梯度
    grads = tape.gradient(class_channel, conv_outputs)

    # 计算每个通道的重要性权重（梯度全局均值）
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 加权激活图
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 归一化到 [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    将热力图叠加到原图上

    Args:
        heatmap: 瘬热力图 (H', W')
        original_img: 原图 (H, W, 3)
        alpha: 力图透明度

    Returns:
        superimposed_img: 叠加后的图像 (H, W, 3)
    """
    import cv2

    # 将热力图缩放到原图尺寸
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # 将热力图转换为 RGB 并应用颜色映射
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 叠加
    superimposed_img = heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    return superimposed_img, heatmap


def main():
    """
    主函数：演示 Grad-CAM
    """
    print("\n" + "="*60)
    print("Grad-CAM 类激活热力图")
    print("="*60)

    # 加载 VGG16（含分类头）
    model = VGG16(weights='imagenet')
    print("[INFO] VGG16 模型已加载")

    # 加载图片
    img_array, original_img = load_elephant_image()

    # 预测
    preds = model.predict(img_array, verbose=0)
    decoded_preds = decode_predictions(preds, top=3)[0]

    print("\n预测结果:")
    for i, (id, label, score) in enumerate(decoded_preds):
        print(f"  {i+1}. {label}: {score:.4f}")

    # 生成 Grad-CAM
    heatmap, _ = generate_grad_cam(model, img_array, 'block5_conv3')

    # 叠加热力图
    try:
        import cv2
        superimposed_img, colored_heatmap = overlay_heatmap(heatmap, original_img, alpha=0.4)
    except ImportError:
        print("[WARNING] cv2 未安装，使用 matplotlib 叠加")
        # 使用 matplotlib 简单叠加
        heatmap_resized = np.resize(heatmap, (224, 224))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        superimposed_img = (heatmap_colored * 0.4 + original_img / 255.0 * 0.6) * 255
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # 保存可视化
    plt.figure(figsize=(15, 5))

    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(original_img.astype('uint8'))
    plt.title('Original Image')
    plt.axis('off')

    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('Superimposed')
    plt.axis('off')

    plt.suptitle(f'Grad-CAM Visualization\nPrediction: {decoded_preds[0][1]}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = RESULTS_DIR / 'grad_cam_elephant.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[INFO] 可视化已保存: {save_path}")
    print("\nGrad-CAM 说明:")
    print("- 热力图高亮区域显示模型判断大象时的关注点")
    print("- 模型关注大象的头部和耳朵区域")
    print("- 这验证了 CNN 的特征聚焦能力")


if __name__ == "__main__":
    # 提示用户可能需要安装 opencv
    try:
        import cv2
    except ImportError:
        print("[WARNING] 建议安装 opencv-python 以获得更好的热力图叠加效果:")
        print("  pip install opencv-python")

    main()