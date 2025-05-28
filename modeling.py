import torch
import torch.nn as nn
import numpy as np


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        # 卷积层 + 批归一化（1D）
        self.conv1 = nn.Conv1d(1, 16, kernel_size=12, stride=1)
        self.bn1 = nn.BatchNorm1d(16)  # 匹配conv1的输出通道数16
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=4)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1)
        self.bn2 = nn.BatchNorm1d(32)  # 匹配conv2的输出通道数32
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=6, stride=1)
        self.bn3 = nn.BatchNorm1d(64)  # 匹配conv3的输出通道数64
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=4)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(64)  # 匹配conv4的输出通道数64
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算全连接层输入维度（需包含批归一化）
        self.fc_input_dim = self._get_fc_input_dim()
        self.fc = nn.Linear(self.fc_input_dim, 256)

    def _get_fc_input_dim(self):
        # 假输入数据需经过完整的前向流程（包括批归一化）
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 2048)  # (batch, channels, length)
            x = self.conv1(dummy_input)
            x = self.bn1(x)  # 批归一化
            x = torch.relu(x)  # 激活
            x = self.pool1(x)  # 池化

            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = torch.relu(x)
            x = self.pool3(x)

            x = self.conv4(x)
            x = self.bn4(x)
            x = torch.relu(x)
            x = self.pool4(x)

            return int(np.prod(x.size()))  # 展平后的维度

    def forward(self, x):
        # 卷积 → 批归一化 → 激活 → 池化（标准流程）
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)

        x = x.view(-1, self.fc_input_dim)  # 展平
        x = torch.relu(self.fc(x))  # 全连接 + 激活
        return x


class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        # 卷积层 + 批归一化（2D）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)  # 匹配conv1的输出通道数32
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)  # 匹配conv2的输出通道数64
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)  # 匹配conv3的输出通道数128
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # 计算展平后的输入维度（包含批归一化）
        self.flattened_size = self._get_flattened_size()
        self.fc = nn.Linear(self.flattened_size, 256)

    def _get_flattened_size(self):
        # 假输入数据需经过完整的前向流程（包括批归一化）
        with torch.no_grad():
            x = torch.zeros(1, 1, 128, 2048)  # (batch, channels, height, width)
            x = self.conv1(x)
            x = self.bn1(x)  # 批归一化
            x = torch.relu(x)  # 激活
            x = self.pool1(x)  # 池化

            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = torch.relu(x)
            x = self.pool3(x)

            return x.numel()  # 展平后的元素总数

    def forward(self, x):
        # 卷积 → 批归一化 → 激活 → 池化（标准流程）
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc(x))  # 全连接 + 激活
        return x

