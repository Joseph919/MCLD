import os
import torch
from torch.utils.data import *
from torchvision import datasets, transforms
from PIL import Image

# 2. 2d data loaderdddd
# 2.1 class
class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.files = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        self.files[idx]
        if self.transform:
            img = self.transform(img)               # → Tensor [C,H,W]
        label = self.labels[idx]                    # int
        return img, label

    def __len__(self):
        return len(self.files)

# 2.2 加载2d tensor
data = torch.load('cwru_img/train_2d.pt')
features_2d = data['features']  # Tensor [N, C, H, W]
labels_2d  = data['labels']

print(f'二维信号大小: {features_2d.shape}， label大小: {labels_2d.shape}')

# transform = transforms.Compose(transforms.ToTensor())
# dataset = MyImageDataset(files, labels, transform)
# loader  = DataLoader(dataset, batch_size=32, shuffle=True)





def get1dDataLoader ():
    return


def get2dDataLoader ():
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64)
    # return train_loader

    # 设置图片文件夹路径和保存路径
    img_dir = '/Users/pikabp/Documents/Papper/Constrastive Learning/cwru_img/images/train/wavelet'
    save_dir = '/Users/pikabp/Documents/Papper/Constrastive Learning/tensor'

    # 获取所有png文件并排序，确保顺序一致
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    # 准备存储所有图片和标签的列表
    all_images = []
    all_labels = []

    # 图像转换器
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 读取每个图片并处理
    for img_file in img_files:
        # 从文件名获取标签（第一个_前的数字）
        label = int(img_file.split('_')[0])

        # 读取图片
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)

        # 转换为tensor
        img_tensor = transform(img)

        # 添加到列表
        all_images.append(img_tensor)
        all_labels.append(label)

    # 将所有图片和标签转换为tensor
    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 保存张量
    torch.save({
        'images': images_tensor,
        'labels': labels_tensor
    }, os.path.join(save_dir, 'image_data.pt'))

    # 创建数据集和数据加载器
    dataset = TensorDataset(images_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    return dataloader