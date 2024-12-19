import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# 定义 PalmprintResNet
class PalmprintResNet(nn.Module):
    def __init__(self):
        super(PalmprintResNet, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 使用 ResNet 的特征提取部分
        self.attention = AttentionModule(channels=2048)  # 调整注意力模块通道数
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, 128)  # 特征维度为 128

    def forward_one_side(self, x):
        x = self.features(x)
        x = self.attention(x)  # 应用注意力模块
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one_side(input1)
        output2 = self.forward_one_side(input2)
        return output1, output2
