import torch
import torchvision.models as models

# ============================================================================
# 1. 保存和加载模型权重
# ============================================================================
# PyTorch 模型的学习参数存储在内部状态字典 state_dict 中。
# 通过 torch.save 方法可以将 state_dict 持久化保存。

# ---- 加载预训练模型并保存其权重 ----
# 创建一个 VGG16 模型，并加载预训练的权重
model = models.vgg16(weights='IMAGENET1K_V1')
# 保存模型的学习参数到文件
torch.save(model.state_dict(), 'model_weights.pth')

# ---- 加载模型权重 ----
# 首先需要创建一个与保存时结构相同的模型实例（此时不使用预训练权重）
model = models.vgg16() # 不指定 weights，即创建一个未训练的模型
# 使用 load_state_dict() 加载保存的权重
# weights_only=True 是最佳实践，仅在反序列化时执行加载权重所必需的函数
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# 在推理前，务必调用 model.eval()，将 dropout 和 batchNorm 层设置为评估模式
model.eval()
# 此时模型已加载了预训练的权重，可用于预测

# ============================================================================
# 2. 保存和加载带有模型结构的模型
# ============================================================================
# 除了权重，还可以将整个模型的结构与参数一起保存。

# ---- 保存整个模型 ----
# 直接将 model 对象（而非 model.state_dict()）传给 torch.save
torch.save(model, 'model.pth')

# ---- 加载整个模型 ----
# 由于保存时使用了模型对象，加载时可以直接得到整个模型
# weights_only=False 是因为这里加载的是整个模型，属于 torch.save 的遗留用例
model = torch.load('model.pth', weights_only=False)
# 同样，加载后也需要切换到评估模式
model.eval()