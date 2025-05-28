import torch
import numpy
from data_loader import *
from modeling import *

def extract_and_save_features (model, dataloader, path) :
    # 1. 设置为评估模式
    model.eval()

    # 2. 创建容器
    all_features = []
    all_labels = []  # 如果需要保存标签的话

    # 3. 正向传播
    with torch.no_grad() :
        for batch_data, batch_labels in dataloader :
            feature = model.extract_features(batch_data)

            all_features.append(feature)
            all_labels.append(batch_labels)

            # print(f"处理批次 , 特征形状: {feature.shape}")

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"特征形状: {all_features.shape}， label形状: {all_labels.shape}")

    torch.save({'all_features': all_features, 'all_labels': all_labels}, path)
    
    print('load and save success')
    return


cnn2d = CNN2D()
dataLoader2d = get2dDataLoader()
features2dPath = 'features/feature_2d.pt'
extract_and_save_features(cnn2d, dataLoader2d, features2dPath)

cnn1d = CNN1D()
dataLoader1d = get1dDataLoader()
features1dPath = 'features/feature_1d.pt'
extract_and_save_features(cnn1d, dataLoader1d, features1dPath)