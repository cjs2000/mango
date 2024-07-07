import torch
from torchviz import make_dot
from models.eca_module import *
from models.AlexNet import *

model = AlexNet()  # 替换为你的模型
x = torch.randn(1, 3, 224, 224)  # 输入张量，形状根据你的模型调整
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = 'png'  # 保存格式
dot.render('network_structure')  # 保存路径
