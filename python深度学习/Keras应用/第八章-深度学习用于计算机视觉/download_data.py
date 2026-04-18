"""
download_data.py - 数据准备脚本
《Python深度学习》第5章配套代码

功能：
1. 检查数据目录是否已有数据
2. 从原始 train.zip 解压后的目录中抽取并划分数据
3. 按 2000/1000/1000 划分训练/验证/测试集，猫狗各半

使用方法：
    python download_data.py
    python download_data.py --source_dir /path/to/original/train/
"""

import os
import shutil
import argparse
import random
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据划分参数
TRAIN_SAMPLES = 1000       # 每类训练样本数
VAL_SAMPLES = 500          # 每类验证样本数
TEST_SAMPLES = 500         # 每类测试样本数

# 目标目录
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'validation'
TEST_DIR = DATA_DIR / 'test'


def check_data_exists():
    """
    检查目标目录是否已有划分好的数据

    Returns:
        bool: 数据是否已存在
    """
    cats_train = len(list((TRAIN_DIR / 'cats').glob('*.jpg')))
    dogs_train = len(list((TRAIN_DIR / 'dogs').glob('*.jpg')))

    if cats_train >= TRAIN_SAMPLES and dogs_train >= TRAIN_SAMPLES:
        print(f"[INFO] 数据已存在: train/cats={cats_train}, train/dogs={dogs_train}")
        return True
    return False


def copy_images(src_dir: Path, category: str, dst_dir: Path, count: int):
    """
    从源目录随机抽取并复制图像到目标目录

    Args:
        src_dir: 源目录（包含所有猫/狗图片）
        category: 类别名称 ('cats' 或 'dogs')
        dst_dir: 目标目录
        count: 需要复制的数量
    """
    # 获取源目录中所有该类别的图片
    pattern = f'{category}.*' if category == 'cats' else f'dog.*'
    # Kaggle 数据集命名: cat.0.jpg, dog.0.jpg 等
    all_images = list(src_dir.glob(f'{category}.*.jpg'))

    if len(all_images) < count:
        raise ValueError(f"源目录 {category} 图片不足: 需要 {count}, 实际 {len(all_images)}")

    # 随机抽样
    random.seed(42)  # 可复现性
    selected = random.sample(all_images, count)

    # 复制到目标目录
    dst_category_dir = dst_dir / category
    dst_category_dir.mkdir(parents=True, exist_ok=True)

    for img_path in selected:
        dst_path = dst_category_dir / img_path.name
        shutil.copy(img_path, dst_path)

    print(f"[INFO] 复制 {count} 张 {category} 图片到 {dst_dir}/{category}")


def prepare_data(source_dir: str = None):
    """
    数据准备主函数

    Args:
        source_dir: 原始 train.zip 解压后的目录路径
                    如果为 None，使用默认路径或提示用户
    """
    # 检查是否已有数据
    if check_data_exists():
        print("[INFO] 数据已准备完毕，无需重复处理")
        return

    # 确定源目录
    if source_dir:
        src_path = Path(source_dir)
    else:
        # 封装默认路径假设
        # 常见位置：项目根目录下的 original_train/ 或用户下载目录
        possible_paths = [
            PROJECT_ROOT / 'original_train',
            Path.home() / 'Downloads' / 'dogs-vs-cats' / 'train',
            Path.home() / 'kaggle' / 'dogs-vs-cats' / 'train',
        ]

        src_path = None
        for p in possible_paths:
            if p.exists():
                src_path = p
                break

        if src_path is None:
            print("\n[ERROR] 未找到原始数据目录！")
            print("请按以下步骤操作：")
            print("1. 从 Kaggle 下载 dogs-vs-cats 数据集: https://www.kaggle.com/c/dogs-vs-cats/data")
            print("2. 解压 train.zip 到某个目录")
            print("3. 运行: python download_data.py --source_dir /path/to/unzipped/train/")
            print("\n或者使用 Kaggle API:")
            print("   pip install kaggle")
            print("   kaggle competitions download -c dogs-vs-cats")
            print("   unzip dogs-vs-cats.zip && unzip train.zip")
            return

    print(f"[INFO] 使用源目录: {src_path}")

    # 创建目标目录结构
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for category in ['cats', 'dogs']:
            (split_dir / category).mkdir(parents=True, exist_ok=True)

    # 划分数据
    # 训练集：每类 1000 张
    copy_images(src_path, 'cats', TRAIN_DIR, TRAIN_SAMPLES)
    copy_images(src_path, 'dogs', TRAIN_DIR, TRAIN_SAMPLES)

    # 验证集：每类 500 张
    copy_images(src_path, 'cats', VAL_DIR, VAL_SAMPLES)
    copy_images(src_path, 'dogs', VAL_DIR, VAL_SAMPLES)

    # 测试集：每类 500 张
    copy_images(src_path, 'cats', TEST_DIR, TEST_SAMPLES)
    copy_images(src_path, 'dogs', TEST_DIR, TEST_SAMPLES)

    # 统计最终数量
    print("\n[SUCCESS] 数据划分完成！")
    print("=" * 50)
    for split in ['train', 'validation', 'test']:
        split_dir = DATA_DIR / split
        cats = len(list((split_dir / 'cats').glob('*.jpg')))
        dogs = len(list((split_dir / 'dogs').glob('*.jpg')))
        print(f"{split}: cats={cats}, dogs={dogs}, total={cats+dogs}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备猫狗分类数据集')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='原始 train.zip 解压后的目录路径')

    args = parser.parse_args()
    prepare_data(args.source_dir)