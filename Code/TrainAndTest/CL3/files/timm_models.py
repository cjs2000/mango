# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: cls_template
File Name: timm_models.py
Author: chenming
Create Date: 2021/12/18
Description：
-------------------------------------------------
"""
import timm
from models.eca_resnet import *
import torch
# pretrained_resnet_34 = timm.create_model('tf_efficientnet_l2_ns', pretrained=True)
#timm_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)

timm_model = eca_resnet34()
print(timm_model)