import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import pywt
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from model_three import CNN1D, CNN2D, FeatureFusionModel

# --------------------- GPU 设备配置 ---------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

all_samples = {'data': [], 'labels': []}


def load_mat_file(filepath):
    mat = sio.loadmat(filepath)
    return mat['DE_time'].squeeze()


def sliding_window(data, window_size):
    samples = []
    step = window_size - 321
    for start in range(0, len(data) - window_size + 1, step):
        sample = data[start:start + window_size]
        samples.append(sample)
    return samples


def store_samples(samples, label, all_samples):
    all_samples['data'].extend(samples)
    all_samples['labels'].extend([label] * len(samples))


def calculate_timefreq_features(data, totalscal):
    images = []
    for i in range(data.shape[0]):
        wavelet = 'cmor'
        signal = data[i]
        scales = np.arange(1, totalscal)
        coeffs = pywt.cwt(signal, scales, wavelet)
        coeffs_abs = np.abs(coeffs[0])
        images.append(coeffs_abs)
    images = np.array(images)
    return images


def multi_view_contrastive_loss(feat1, feat2, labels, margin=2.0):
    """
    计算多视角对比损失函数

    参数:
    feat1: 第一个模态的特征表示 [batch_size, feature_dim]
    feat2: 第二个模态的特征表示 [batch_size, feature_dim]
    labels: 样本标签 [batch_size]
    margin: 正负样本对的间隔参数，默认2.0
    """
    # 获取批次大小
    batch_size = labels.size(0)
    # 构建正样本对掩码：标签相同且不是自身对比的样本对
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (torch.eye(batch_size, device=labels.device) == 0)
    # 构建负样本对掩码：标签不同的样本对
    neg_mask = ~pos_mask & (torch.eye(batch_size, device=labels.device) == 0)
    # 计算特征相似度矩阵 (i,j)表示feat1_i与feat2_j的余弦相似度
    sim_matrix = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=-1)
    # 正样本对损失：最大化相似性 (目标为1)
    pos_loss = (1 - sim_matrix) * pos_mask.float()
    # 负样本对损失：使用margin约束 (目标为相似度小于margin)
    neg_loss = torch.clamp(sim_matrix - margin, min=0.0) * neg_mask.float()
    # 计算有效正样本对数量 (避免除零)
    pos_count = pos_mask.sum() if pos_mask.sum() > 0 else 1
    # 计算有效负样本对数量 (避免除零)
    neg_count = neg_mask.sum() if neg_mask.sum() > 0 else 1
    # 计算总损失：正样本对平均损失 + 负样本对平均损失
    total_loss = (pos_loss.sum() / pos_count) + (neg_loss.sum() / neg_count)
    return total_loss


def evaluate_model(model1d, model2d, fusion_model, data_loader):
    model1d.eval()
    model2d.eval()
    fusion_model.eval()

    correct = 0
    total = 0
    # 测试模型
    with torch.no_grad():
        for inputs1, inputs2, labels in data_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            features1 = model1d(inputs1)
            features2 = model2d(inputs2)
            fused_features = torch.cat((features1, features2), dim=1)
            outputs = fusion_model(fused_features)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def train_model(model1d, model2d, fusion_model, train_loader, val_loader, criterion, optimizer, epochs=50):
    model1d.train()
    model2d.train()
    fusion_model.train()

    best_val_acc = 0.0
    best_model_state = fusion_model.state_dict()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs1, inputs2, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            features1 = model1d(inputs1)
            features2 = model2d(inputs2)
            # 对比学习损失
            cont_loss = multi_view_contrastive_loss(features1, features2, labels)
            fused_features = torch.cat((features1, features2), dim=1)
            outputs = fusion_model(fused_features)
            # 分类损失
            cls_loss = criterion(outputs, labels)
            loss = cont_loss + cls_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 2 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], 训练集: 平均损失 {avg_loss:.4f}, 准确率 {train_accuracy * 100:.2f}%')

        val_accuracy = evaluate_model(model1d, model2d, fusion_model, val_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], 验证集: 准确率 {val_accuracy * 100:.2f}%')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = fusion_model.state_dict()
            print(f"新的最佳模型保存，验证集准确率: {best_val_acc * 100:.2f}%")

    fusion_model.load_state_dict(best_model_state)
    print(f"训练完成，最佳验证集准确率: {best_val_acc * 100:.2f}%")
    return fusion_model


# 设置文件路径和类别标签
data_dir = 'cwru'  # 替换为你的cwru文件夹路径
labels = {'Normal': 0, 'Ball': 1, 'Inner': 2, 'Outer': 3}
diameters = ['0007', '0014', '0021']
conditions = [0, 1, 2, 3]
window_size = 2048
num_train_samples = 80
num_val_samples = 10  # 验证集样本数
num_test_samples = 100

# 处理故障数据
for condition in conditions:
    for diameter in diameters:
        for label_name, label_value in labels.items():
            if label_name == 'Normal':
                continue
            folder_path = os.path.join(data_dir, label_name, diameter)
            file_path = os.path.join(folder_path, f'{condition}.mat')
            data = load_mat_file(file_path)
            samples = sliding_window(data, window_size)
            label = (labels[label_name] - 1) * 3 + diameters.index(diameter) + 1
            store_samples(samples, label, all_samples)

# 处理正常数据
for condition in conditions:
    file_path = os.path.join(data_dir, 'Normal', f'{condition}.mat')
    data = load_mat_file(file_path)
    samples = sliding_window(data, window_size)
    label = 0  # 正常类型标签为0
    store_samples(samples, label, all_samples)

# 转换为numpy数组
all_samples['data'] = np.array(all_samples['data'])
all_samples['labels'] = np.array(all_samples['labels'])

# 划分训练集、验证集和测试集（新增验证集）
data_dict = {'train': {'data': [], 'labels': []},
             'val': {'data': [], 'labels': []},
             'test': {'data': [], 'labels': []}}

for label in np.unique(all_samples['labels']):
    label_indices = np.where(all_samples['labels'] == label)[0]
    np.random.shuffle(label_indices)

    train_indices = label_indices[:num_train_samples]
    val_indices = label_indices[num_train_samples:num_train_samples + num_val_samples]
    test_indices = label_indices[
                   num_train_samples + num_val_samples:num_train_samples + num_val_samples + num_test_samples]

    data_dict['train']['data'].extend(all_samples['data'][train_indices])
    data_dict['train']['labels'].extend(all_samples['labels'][train_indices])

    data_dict['val']['data'].extend(all_samples['data'][val_indices])
    data_dict['val']['labels'].extend(all_samples['labels'][val_indices])

    data_dict['test']['data'].extend(all_samples['data'][test_indices])
    data_dict['test']['labels'].extend(all_samples['labels'][test_indices])

# 转换为numpy数组
for key in data_dict:
    data_dict[key]['data'] = np.array(data_dict[key]['data'])
    data_dict[key]['labels'] = np.array(data_dict[key]['labels'])

print(f"训练数据集大小: {data_dict['train']['data'].shape}; 训练集标签大小：{data_dict['train']['labels'].shape}")
print(f"验证数据集大小: {data_dict['val']['data'].shape}; 验证集标签大小：{data_dict['val']['labels'].shape}")
print(f"测试数据集大小: {data_dict['test']['data'].shape}; 测试集标签大小：{data_dict['test']['labels'].shape}")

# 计算时频图像：训练集、验证集、测试集
scale = 129
train_signal_data = data_dict['train']['data']
train_images_data = calculate_timefreq_features(data=train_signal_data, totalscal=scale)

val_signal_data = data_dict['val']['data']
val_images_data = calculate_timefreq_features(data=val_signal_data, totalscal=scale)

test_signal_data = data_dict['test']['data']
test_images_data = calculate_timefreq_features(data=test_signal_data, totalscal=scale)

# 数据加载：训练集（信号+图像），验证集（信号+图像），测试集（信号+图像）
train_signal_tensor = torch.tensor(train_signal_data, dtype=torch.float32).unsqueeze(1)
train_images_tensor = torch.tensor(train_images_data, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(data_dict['train']['labels'], dtype=torch.long)
train_dataset = TensorDataset(train_signal_tensor, train_images_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_signal_tensor = torch.tensor(val_signal_data, dtype=torch.float32).unsqueeze(1)
val_images_tensor = torch.tensor(val_images_data, dtype=torch.float32).unsqueeze(1)
val_labels = torch.tensor(data_dict['val']['labels'], dtype=torch.long)
val_dataset = TensorDataset(val_signal_tensor, val_images_tensor, val_labels)
val_loader = DataLoader(val_dataset, batch_size=64)

test_signal_tensor = torch.tensor(test_signal_data, dtype=torch.float32).unsqueeze(1)
test_images_tensor = torch.tensor(test_images_data, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(data_dict['test']['labels'], dtype=torch.long)
test_dataset = TensorDataset(test_signal_tensor, test_images_tensor, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64)

print(
    f"训练：信号数据集大小: {train_signal_tensor.shape}；图像数据集大小：{train_images_tensor.shape};标签大小：{train_labels.shape}")
print(
    f"验证：信号数据集大小: {val_signal_tensor.shape}；图像数据集大小：{val_images_tensor.shape};标签大小：{val_labels.shape}")
print(
    f"测试：信号数据集大小: {test_signal_tensor.shape}；图像数据集大小：{test_images_tensor.shape};标签大小：{test_labels.shape}")

# 模型与优化器
# 模型初始化
model1d = CNN1D().to(device)
model2d = CNN2D().to(device)
fusion_model = FeatureFusionModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)
# 三个模型参数的优化器
optimizer = optim.AdamW(
    list(model1d.parameters()) + list(model2d.parameters()) + list(fusion_model.parameters()),
    lr=0.001, weight_decay=1e-4
)

# 训练与测试
best_fusion_model = train_model(
    model1d, model2d, fusion_model,
    train_loader, val_loader,
    criterion, optimizer, epochs=50
)

test_accuracy = evaluate_model(model1d, model2d, best_fusion_model, test_loader)
print(f'测试集准确率: {test_accuracy * 100:.2f}%')