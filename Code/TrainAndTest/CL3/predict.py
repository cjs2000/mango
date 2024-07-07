#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cls_template 
@File    ：predict.py
@Author  ：ChenmingSong
@Date    ：2022/1/5 16:23 
@Description：用来推理数据集
'''
import torch
# from train_resnet import SelfNet
import shufflenet_IM.shufflenet_ECA_GC
from train import SELFMODEL
import os
import os.path as osp
import shutil
import torch.nn as nn
from PIL import Image
from torchutils import get_torch_transforms
from models.resnet import *


import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Function
import matplotlib.pyplot as plt





if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_path = r"checkpoints/resnet34/Mango_split224/resnet34_119epochs_accuracy0.99766_weights.pth"  # todo  模型路径
classes_names = ['Ao Mango', 'GuiQi Mango', 'Jinhuang Mango', 'Tainong Mango']  # todo 类名
img_size = 224  # todo 图片大小
model_name = "resnet34"  # todo 模型名称
num_classes = len(classes_names)  # todo 类别数目


def predict_batch(model_path, target_dir, save_dir):
    data_transforms = get_torch_transforms(img_size=img_size)
    valid_transforms = data_transforms['val']
    # 加载网络
    model = shufflenet_IM.shufflenet_ECA_GC.shufflenet_eca_gc()
    # model = nn.DataParallel(model)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    # 读取图片
    image_names = os.listdir(target_dir)
    for i, image_name in enumerate(image_names):
        image_path = osp.join(target_dir, image_name)
        img = Image.open(image_path)
        img = valid_transforms(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        output = model(img)
        label_id = torch.argmax(output).item()
        predict_name = classes_names[label_id]
        save_path = osp.join(save_dir, predict_name)
        if not osp.isdir(save_path):
            os.makedirs(save_path)
        shutil.copy(image_path, save_path)
        print(f"{i + 1}: {image_name} result {predict_name}")


def predict_single(model_path, image_path):
    data_transforms = get_torch_transforms(img_size=img_size)
    # train_transforms = data_transforms['train']
    valid_transforms = data_transforms['val']
    # 加载网络


    model = shufflenet_IM.shufflenet_ECA_GC.shufflenet_eca_gc()
    # model = nn.DataParallel(model)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # 读取图片
    img = Image.open(image_path)
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    label_id = torch.argmax(output).item()

    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence = probabilities[0, label_id].item()
    print(confidence)

    predict_name = classes_names[label_id]
    print(f"{image_path}'s result is {predict_name}")









if __name__ == '__main__':
    # 批量预测函数

    model_path="checkpoints/shufflenet_eca_gc_4/best.pth"
    image_path=""
    # 单张图片预测函数
    #predict_single(model_path=model_path, image_path="M/a.jpg")
    predict_with_cam(model_path=model_path,
                     image_path="M/a.jpg",
                     target_layer_names=["layer12"])

