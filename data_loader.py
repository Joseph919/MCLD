import os
import torch
from torch.utils.data import *
from torchvision import datasets, transforms
from PIL import Image

# TODO Normalize input

# 1. loader class

# 1.1 1d data loader
class SignalDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.files = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        signal = self.files[idx]
        if self.transform:
            signal = self.transform(self.files)               # → Tensor [C,H,W]
        label = self.labels[idx]
        return signal, label

    def __len__(self):
        return len(self.files)

# 1.2 2d data loader
class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.files = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.files[idx]
        if self.transform:
            img = self.transform(self.files)               # → Tensor [C,H,W]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.files)




def get1dDataLoader ():
    # 加载1d tensor
    data = torch.load('cwru_img/train_1d.pt')
    features_1d = data['features']  # Tensor [B, C, W]
    labels_1d = data['labels']

    print(f'二维信号大小: {features_1d.shape}， label大小: {labels_1d.shape}')

    # 设置loader
    transform = transforms.Compose(transforms.ToTensor())
    dataset = ImageDataset(features_1d, labels_1d)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    return loader

def get2dDataLoader ():
    # 加载2d tensor
    data = torch.load('cwru_img/train_2d.pt')
    features_2d = data['features']  # Tensor [B, C, H, W]
    labels_2d = data['labels']

    print(f'二维信号大小: {features_2d.shape}， label大小: {labels_2d.shape}')

    # 设置loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # 显式归一化到[-1,1]
    ])
    dataset = ImageDataset(features_2d, labels_2d)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    return loader

get2dDataLoader()







