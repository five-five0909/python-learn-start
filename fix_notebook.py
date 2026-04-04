# -*- coding: utf-8 -*-
import json

cells = []

# Cell 0: Title
cells.append({
    "cell_type": "markdown", "id": "a238a705", "metadata": {},
    "source": [
        "# Keras 核心 API 全流程实战\n",
        "\n",
        "**学习路线：**\n",
        "\n",
        "| 章节 | 内容 | 关键词 |\n",
        "|------|------|--------|\n",
        "| §1 | 层与模型的关系 | Layer → Model |\n",
        "| §2 | 自定义层 | `build()` / `call()` |\n",
        "| §3 | 自动形状推断 | 为什么不用写 `input_dim` |\n",
        "| §4 | Sequential vs 函数式 API | 两种搭模型的方式 |\n",
        "| §5 | Transformer 残差块 | 多头注意力 + 残差连接 |\n",
        "| §6 | compile / fit / evaluate / predict | 训练四步曲 |\n",
        "| §7 | 损失函数与指标 | 二分类 / 多分类 / 回归 |\n",
        "| §8 | 可视化训练过程 | 画 loss 和 accuracy 曲线 |\n",
        "| §9 | 模型保存与加载 | save / load |\n",
        "\n",
        "> 全部使用合成数据，可直接运行。跑通后替换成自己的数据即可。"
    ]
})

# Cell 1: Imports
cells.append({
    "cell_type": "code", "execution_count": None, "id": "64112ed4", "metadata": {}, "outputs": [],
    "source": [
        "# === 基础导入 ===\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "\n",
        "print(f\"TensorFlow: {tf.__version__}\")\n",
        "print(f\"Keras: {keras.__version__}\")"
    ]
})

# Cell 2: §1 header
cells.append({
    "cell_type": "markdown", "id": "d518c5a0", "metadata": {},
    "source": [
        "---\n",
        "## §1 层与模型的关系\n",
        "\n",
        "**层（Layer）** = 数据处理模块，输入张量 → 输出张量，内部可能有可训练参数。\n",
        "\n",
        "**模型（Model）** = 多个层组合在一起的结构，Model 本身也是 Layer 的子类。\n",
        "\n",
        "| 概念 | 封装的东西 |\n",
        "|------|----------|\n",
        "| Layer | **状态**（权重 W、偏置 b）+ **计算**（前向传播逻辑） |\n",
        "| Model | 多个 Layer 的拓扑连接 + compile/fit/evaluate/predict 方法 |\n",
        "\n",
        "**标准工作流：**\n",
        "```\n",
        "定义层 → 组合成模型 → compile() → fit() → evaluate() / predict()\n",
        "```"
    ]
})

# Cell 3: §2 header
cells.append({
    "cell_type": "markdown", "id": "0790d286", "metadata": {},
    "source": [
        "---\n",
        "## §2 自定义 Dense 层\n",
        "\n",
        "自定义层需要实现两个方法：\n",
        "\n",
        "| 方法 | 调用时机 | 职责 |\n",
        "|------|---------|------|\n",
        "| `build(input_shape)` | 第一次收到输入时 | 根据输入维度创建权重 |\n",
        "| `call(inputs)` | 每次前向传播 | 定义计算逻辑 |\n",
        "\n",
        "**Dense 层的数学本质：** `output = activation(X @ W + b)`"
    ]
})

# Cell 4: SimpleDense
cells.append({
    "cell_type": "code", "execution_count": None, "id": "36631806", "metadata": {}, "outputs": [],
    "source": [
        "class SimpleDense(keras.layers.Layer):\n",
        "    \"\"\"自定义全连接层：output = activation(X @ W + b)\"\"\"\n",
        "\n",
        "    def __init__(self, units, activation=None, **kwargs):\n",
        "        super().__init__(**kwargs)\n",
        "        self.units = units\n",
        "        self.activation = activation\n",
        "\n",
        "    def build(self, input_shape):\n",
        "        input_dim = input_shape[-1]\n",
        "        self.W = self.add_weight(\n",
        "            shape=(input_dim, self.units),\n",
        "            initializer=\"random_normal\",\n",
        "            trainable=True, name=\"kernel\"\n",
        "        )\n",
        "        self.b = self.add_weight(\n",
        "            shape=(self.units,),\n",
        "            initializer=\"zeros\",\n",
        "            trainable=True, name=\"bias\"\n",
        "        )\n",
        "\n",
        "    def call(self, inputs):\n",
        "        y = tf.matmul(inputs, self.W) + self.b\n",
        "        if self.activation is not None:\n",
        "            y = self.activation(y)\n",
        "        return y"
    ]
})

print(f"已创建 {len(cells)} 个单元格，继续添加更多...")

# Save intermediate
notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.x"}},
    "nbformat": 4, "nbformat_minor": 5
}

path = "E:/code_space/python_space/python-learn-start/python深度学习/第三章-Keras和TensorFlow入门/keras_核心API_全解.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("第一部分写入完成")
