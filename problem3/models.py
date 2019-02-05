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
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(13*13*64, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 13*13*64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
class VGG(SerializableModule):
    def __init__(self, vgg_name='VGG7'):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_config[vgg_name])
        self.classifier = nn.Linear(2048, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           # nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

vgg_config = {
    'VGG7': [64, 'M', 128, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# adapted from https://github.com/MaximumEntropy/welcome_tutorials/blob/pytorch/
class CIFARResNet18(SerializableModule):
    def __init__(self, num_classes=10):
        super(CIFARResNet18, self).__init__()

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Create stages 1-4
        self.stage1 = self._create_stage(64, 64, stride=1)
        self.stage2 = self._create_stage(64, 128, stride=2)
        self.stage3 = self._create_stage(128, 256, stride=2)
        self.stage4 = self._create_stage(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    # A stage is just two residual blocks for ResNet18
    def _create_stage(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def find_model(model_str):
    if model_str == 'VGG':
        return VGG
    elif model_str == 'BaseConvNet':
        return BaseConvNet
    else:
        raise("No model with name '" + model_str + "' was found.")


# class MyConvNet(SerializableModule):
#     def __init__(self):
#         super(MyConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 48, 3, 1)
#         self.conv2 = nn.Conv2d(48, 48, 3, 1)
#         self.fc1 = nn.Linear(13*13*48, 256)
#         self.fc2 = nn.Linear(256, 2)
#
#     def forward(self, x):
#         # 2 convs, 1 maxpool
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#
#         # 2 convs, 1 maxpool
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#
#         # 2 convs, 1 maxpool
#         # x = F.relu(self.conv2(x))
#         # x = F.relu(self.conv2(x))
#         # x = F.max_pool2d(x, 2, 2)
#
#         x = x.view(-1, 13*13*48)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)