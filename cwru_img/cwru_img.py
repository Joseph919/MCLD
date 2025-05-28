import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import pywt
from fontTools.ttLib.woff2 import bboxFormat
from matplotlib import pyplot as plt

# 用于存储所有样本的列表
all_samples = {'data': [], 'labels': []}


# 读取.mat文件数据
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
    """
    计算时频分析核心特征（与绘图逻辑解耦）

    :param data: 输入数据（形状应为 (样本数, 时间序列长度)）
    :param sampling_period: 采样周期
    :param totalscal: 总尺度数
    :param wavename: 小波基名称
    :return: 时频特征数组（形状：(样本数, 频率数, 时间数)），频率数组列表
    """
    images = []
    for i in range(data.shape[0]):
        # 选择复数小波（或可以选择其他如 'db1', 'haar' 等）
        wavelet = 'cmor'
        # 计算小波参数
        signal = data[i]
        scales = np.arange(1, totalscal)  # 定义尺度范围
        # 连续小波变换
        coeffs = pywt.cwt(signal, scales, wavelet)
        # 获取小波变换系数的幅度
        coeffs_abs = np.abs(coeffs[0])  # 只取小波变换结果的幅度部分
        # coeffs_abs其实就是二维图像的数据
        # 计算幅度特征
        images.append(coeffs_abs)
        if i % 10 == 0 :
            print(f'calculate_timefreq_features : {i}, images size: {len(images)}')
    images = np.array(images)
    return images


def plot_signal(signal_data, signal_label, show = False, train = True):
    """
       绘制(1,2048)数组的波形图

       :param signal_data: 输入信号数据（形状应为 (1, 2048)）
       """
    # 提取一维数据（将(1,2048)转为(2048,)）
    signal = signal_data.flatten()

    # 生成时间轴（0到2047的整数，对应2048个时间点）
    time_points = np.arange(len(signal))  # 结果为[0, 1, 2, ..., 2047]

    # 创建画布并绘图
    plt.figure(figsize=(12, 5))  # 图像尺寸（宽12英寸，高5英寸）
    plt.plot(time_points, signal, color='#2c7fb8', linewidth=0.8 )

    # 图表配置
    # plt.xlabel('', fontsize=11)
    # plt.ylabel('', fontsize=11)
    # plt.grid(linestyle='--', alpha=0.5)  # 添加虚线网格
    # plt.tick_params(axis='both', labelsize=9)  # 调整刻度字体大小
    plt.legend(fontsize=15)  # 显示图例

    if show:
        plt.show()
    else:
        match train:
            case True:
                folder_path = os.path.join("images", "train", "signal")
                save_path = os.path.join(folder_path, f'{condition}.mat')
            case _:
                save_path = ""

        if len(save_path) > 0 :
            print(f'save signal image path is: {save_path}')
            plt.savefig(save_path, dpi=300)
        else:
            print("signal save_path is empty")

def plot_img(coeffs_abs, img_label, scale, show = False, train = True):
    # 创建一个绘图
    plt.figure(figsize=(10, 6))
    scales = np.arange(1, scale)  # 定义尺度范围
    x_len = coeffs_abs.shape[1]

    plt.imshow(coeffs_abs, aspect='auto', extent=[0, x_len, 1, len(scales)], cmap='jet',
               origin='lower')
    # plt.colorbar(label='Magnitude')
    # plt.title('Wavelet Transform of the Signal')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Scale')

    if show:
        plt.show()
    else:
        match train:
            case True:
                save_path = ""
            case _:
                save_path = ""
        if len(save_path) > 0 :
            print(f'save wavelet image path is: {save_path}')
            plt.savefig(save_path, dpi=300)
        else:
            print("wavelet save_path is empty")


# 我的测试

def plot_signal_idx(i, show, phase):
    print(f'start plot_signal_idx: {i}')
    """
       绘制(1,2048)数组的波形图

       :param signal_data: 输入信号数据（形状应为 (1, 2048)）
       """
    # 提取一维数据（将(1,2048)转为(2048,), 获取signal 对应的label
    match phase:
        case "train":
            signal = labeled_data[i].flatten()
            label = labeled_data_labels[i]
        case _:
            signal = test_data[i].flatten()
            label = test_data_labels[i]

    # 生成时间轴（0到2047的整数，对应2048个时间点）
    time_points = np.arange(len(signal))  # 结果为[0, 1, 2, ..., 2047]

    # 创建画布并绘图
    plt.figure(figsize=(12, 5))  # 图像尺寸（宽12英寸，高5英寸）
    plt.plot(time_points, signal, color='#2c7fb8', linewidth=0.8 )

    # 图表配置
    # plt.xlabel('', fontsize=11)
    # plt.ylabel('', fontsize=11)
    # plt.grid(linestyle='--', alpha=0.5)  # 添加虚线网格
    # plt.tick_params(axis='both', labelsize=9)  # 调整刻度字体大小
    plt.legend(fontsize=15)  # 显示图例

    if show:
        plt.show()
    else:
        folder_path = os.path.join("images", phase, "signal")
        save_path = os.path.join(folder_path, f'{label}_{i}.png')

        # 保存图像
        print(f'save signal image path is: {save_path}')
        plt.savefig(save_path, dpi=300)

    plt.close()
    print(f'complete plot_signal_idx: {i}')

def plot_img_idx(i, scale, show, phase):
    print(f'start plot_img_idx: {i}')
    # 创建一个绘图
    plt.figure(figsize=(10, 6))
    scales = np.arange(1, scale)  # 定义尺度范围

    # 获取图像数据
    match phase:
        case "train":
            coeffs_abs = images_train[i]
            label = labeled_data_labels[i]
            x_len = images_train[i].shape[1]
        # case _:
        #     coeffs_abs = images_test[i]
        #     label = test_data_labels[i]
        #     x_len = images_test[i].shape[1]

    plt.imshow(coeffs_abs, aspect='auto', extent=[0, x_len, 1, len(scales)], cmap='jet',
               origin='lower')
    # plt.colorbar(label='Magnitude')
    # plt.title('Wavelet Transform of the Signal')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Scale')

    if show:
        plt.show()
    else:
        folder_path = os.path.join("images", phase, "wavelet")
        save_path = os.path.join(folder_path, f'{label}_{i}.png')

        # 保存图像
        print(f'save time-sequence image path is: {save_path}')
        plt.savefig(save_path, dpi=300)

    plt.close()

    print(f'complete plot_img_idx: {i}')



# 1 设置文件路径和类别标签
data_dir = 'cwru'  # 替换为你的cwru文件夹路径
labels = {'Normal': 0, 'Ball': 1, 'Inner': 2, 'Outer': 3}
diameters = ['0007', '0014', '0021']
conditions = [0, 1, 2, 3]
window_size = 2048
num_train_samples = 2

# 2 处理数据
# 2.1 处理故障数据
for condition in conditions:
    for diameter in diameters:
        for label_name, label_value in labels.items():
            if label_name == 'Normal':
                continue
            folder_path = os.path.join(data_dir, label_name, diameter)
            file_path = os.path.join(folder_path, f'{condition}.mat')
            data = load_mat_file(file_path)
            samples = sliding_window(data, window_size)

            label = (labels[label_name] - 1) * 3 + diameters.index(diameter) + 1  # 为不同故障类型生成唯一标签

            store_samples(samples, label, all_samples)

# 2.2 处理正常数据
for condition in conditions:
    file_path = os.path.join(data_dir, 'Normal', f'{condition}.mat')
    data = load_mat_file(file_path)
    samples = sliding_window(data, window_size)
    label = 0  # 正常类型标签为0
    store_samples(samples, label, all_samples)

# 将所有数据转换为numpy数组
all_samples['data'] = np.array(all_samples['data'])
all_samples['labels'] = np.array(all_samples['labels'])

# 3对每个类型的标记样本、测试样本和无标签样本进行划分
data_dict = {
    'train': {'data': [], 'labels': []},
    'test': {'data': [], 'labels': []},
}

# 3.1划分训练集和测试集
for label in np.unique(all_samples['labels']):
    label_indices = np.where(all_samples['labels'] == label)[0]
    np.random.shuffle(label_indices)

    train_indices = label_indices[:num_train_samples]
    test_indices = label_indices[num_train_samples:]

    data_dict['train']['data'].extend(all_samples['data'][train_indices])
    data_dict['train']['labels'].extend(all_samples['labels'][train_indices])
    data_dict['test']['data'].extend(all_samples['data'][test_indices])
    data_dict['test']['labels'].extend(all_samples['labels'][test_indices])

# 3.2将数据转换为numpy数组
for key in data_dict:
    data_dict[key]['data'] = np.array(data_dict[key]['data'])
    data_dict[key]['labels'] = np.array(data_dict[key]['labels'])

labeled_data = data_dict['train']['data']
labeled_data_labels = data_dict['train']['labels']
test_data = data_dict['test']['data']
test_data_labels = data_dict['test']['labels']

# 3.3 打印数据集大小
print(f"训练数据集大小: {labeled_data.shape}; 训练集标签大小：{labeled_data_labels.shape}")
print(f"测试数据集大小: {test_data.shape}; 测试集标签大小：{test_data_labels.shape}")

# 4 进行小波变换，生成图像数据
scale = 129 # 小波变换尺度

# 4.0 生成训练和测试图像数据
images_train = calculate_timefreq_features(data=labeled_data, totalscal=scale)
# images_test = calculate_timefreq_features(data=test_data, totalscal=scale)

# 测试 绘制第一个样本的时频图（可根据需要选择绘制）
# plot_signal(labeled_data[0], True)
# plot_img(images[0], scale, True)

# 4.1 要绘制的图像和信号个数
num_plots_train = len(labeled_data)
num_plots_test = len(test_data)

print(f"训练信号数据集大小: {labeled_data.shape}; 图像数据集大小: {images_train.shape};  增强标签集大小：{labeled_data_labels.shape}")

# 5.保存1d 和 2d tensor 以及对应的labels
# 5.1 保存1d
features_tensor_1d = torch.from_numpy(labeled_data)
labels_tensor_1d = torch.from_numpy(labeled_data_labels)

print(f'一维信号大小: {features_tensor_1d.shape}， label大小: {labels_tensor_1d.shape}')

torch.save({
    'features' : features_tensor_1d,
    'labels' : labels_tensor_1d,
}, 'train_1d.pt')

# 5.2 保存2d
# 5.2.1 增加通道维度
images_train = np.expand_dims(images_train, axis = 1)

features_tensor_2d = torch.from_numpy(images_train)
labels_tensor_2d = torch.from_numpy(labeled_data_labels)

print(f'二维信号大小: {features_tensor_2d.shape}， label大小: {labels_tensor_2d.shape}')

torch.save({
    'features' : features_tensor_2d,
    'labels' : labels_tensor_2d,
}, 'train_2d.pt')


# # 4.2 绘制并保存训练数据图像
# for i in range(0, num_plots_train):
#     #plot_signal(labeled_data[i], labeled_data_labels[i],False)
#     #plot_img(images_train[i], labeled_data_labels[i], scale, False)
#
#     plot_signal_idx(i, False, "train")
#     plot_img_idx(i, scale, False, 'train')



# 4.3 绘制并保存测试数据图像
# for i in range(0, num_plots_train):
#     #plot_signal(labeled_data[i], test_data_labels[i], False, False)
#     #plot_img(images_train[i], test_data_labels[i], scale, False, False)
#
#     plot_signal_idx(i, False, "test")
#     plot_img_idx(i, scale, False, 'test')

# 4.4 打印数据集大小
# print(f"测试信号数据集大小: {test_data.shape}; 图像数据集大小: {images_test.shape};  增强标签集大小：{labeled_data_labels.shape}")