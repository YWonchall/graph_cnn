import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_add_pool
from torch.nn import Linear

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm1d(out_channels)
    
    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        data.x = self.bn(data.x)
        return data.x

class BasicBlock(nn.Module):
    """
    ResNet 的基本残差模块 (Basic Block)。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.bn1 = BatchNorm1d(out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 下采样模块

    def forward(self, data):
        identity = data.x
        if self.downsample is not None:
            identity = self.downsample(data)

        data.x = self.conv1(data.x, data.edge_index)
        data.x = self.bn1(data.x)
        data.x = self.relu(data.x)

        data.x = self.conv2(data.x, data.edge_index)
        data.x = self.bn2(data.x)

        data.x += identity
        data.x = self.relu(data.x)

        return data

class FClayer(nn.Module):
    def __init__(self, dim_in, dim1, dim_out):
        super(FClayer, self).__init__()

        self.fc1 = Linear(dim_in, dim1)
        self.bn1 = BatchNorm1d(dim1)
        self.fc2 = Linear(dim1, dim_out)
   
    def forward(self, data):
        x1 = F.relu(self.fc1(data))
        x1 = self.bn1(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x_out = self.fc2(x1)
        return x_out

class ResNet(nn.Module):
    """
    通用的 ResNet 框架。
    """
    def __init__(self, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = GCNConv(75, 64)
        self.bn1 = BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        创建一个残差层。
        Args:
            block: 残差块类型（如 BasicBlock）。
            out_channels: 输出通道数。
            blocks: 残差块的数量。
            stride: 第一个残差块的步幅。
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = DownSample(self.in_channels, out_channels * block.expansion)
        layers = []
        layers.append(block(out_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data):
        data.x = self.conv1(data.x, data.edge_index)
        data.x = self.bn1(data.x)
        data.x = self.relu(data.x)


        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = self.layer4(data)

        return data.x

class ResGraphNet(nn.Module):
    """
    通用的 ResNet 框架。
    """
    def __init__(self, res_layers):
        super(ResGraphNet, self).__init__()
        self.backbone = ResNet(res_layers)
        self.fc = FClayer(512, 128, 1)

    def forward(self, data, device):
        data.x, data.edge_index = data.x.to(device), data.edge_index.to(device)
        x = self.backbone(data)
        x = global_add_pool(data.x, data.batch)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc(x)
        return x