"""
Different CNN architectures that can be used for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class BaseConvNet(SerializableModule):
    def __init__(self):
        super(BaseConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.fc1 = nn.Linear(13*13*128, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 13*13*128)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BaseConvNet2(SerializableModule):
    def __init__(self):
        super(BaseConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 5, 1)
        self.fc1 = nn.Linear(4*4*256, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BaseConvNet3(SerializableModule):
    def __init__(self):
        super(BaseConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1)
        self.fc1 = nn.Linear(5*5*256, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BaseConvNet4(SerializableModule):
    def __init__(self):
        super(BaseConvNet4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1)
        self.fc1 = nn.Linear(4*4*256, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# source: https://github.com/MaximumEntropy/welcome_tutorials/blob/pytorch/
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        # self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# adapted from https://github.com/MaximumEntropy/welcome_tutorials/blob/pytorch/
class ResNet1(SerializableModule):
    def __init__(self, num_classes=2):
        super(ResNet1, self).__init__()

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )
        # self.bn1 = nn.BatchNorm2d(64)

        # Create stages 1-4
        self.stage1 = self._create_stage(64, 64, stride=1)
        self.stage2 = self._create_stage(64, 128, stride=2)
        self.stage3 = self._create_stage(128, 256, stride=2)
        self.stage4 = self._create_stage(256, 512, stride=2)
        self.linear = nn.Linear(2048, num_classes)

    # A stage is just two residual blocks for ResNet18
    def _create_stage(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)

def find_model(model_str):
    if model_str == 'BaseConvNet':
        return BaseConvNet
    elif model_str == 'ResNet':
        return ResNet
    elif model_str == 'BaseConvNet2':
        return BaseConvNet2
    elif model_str == 'BaseConvNet3':
        return BaseConvNet3
    elif model_str == 'BaseConvNet4':
        return BaseConvNet4
    elif model_str == 'ResNet1':
        return ResNet1
    else:
        raise("No model with name '" + model_str + "' was found.")
