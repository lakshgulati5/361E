"""
Conv5 in PyTorch.
See the paper "FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning"
for more details.
Reference: https://github.com/weichennone/FedMAX/blob/master/digit_object_recognition/models/Nets.py
"""

from torch import nn
import torch.nn.functional as F


class Conv5(nn.Module):

    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv5, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == 'fedmax':
            return x, x_out
        else:
            return x


class Conv5_small(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(Conv5_small, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = self.fc2(x)
        if self.loss_type == 'fedmax':
            return x, x_out
        else:
            return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        if self.loss_type == 'fedmax':
            return out, x_out
        else:
            return out


class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(SimpleMLP, self).__init__()
        # Assumes 32x32 images
        self.fc1 = nn.Linear(channels * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_flat))
        x_out = F.relu(self.fc2(x))
        out = self.fc3(x_out)
        if self.loss_type == 'fedmax':
            return out, x_out
        else:
            return out


class MiniCNN(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x_out = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x_out))
        out = self.fc2(x)
        if self.loss_type == 'fedmax':
            return out, x_out
        else:
            return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=1, loss_type='fedavg'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.loss_type = loss_type

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        x_out = out.view(out.size(0), -1)
        out = self.linear(x_out)
        
        if self.loss_type == 'fedmax':
            return out, x_out
        else:
            return out


def ResNet8(num_classes=10, channels=1, loss_type='fedavg'):
    return ResNet(BasicBlock, [1, 1, 1], num_classes, channels, loss_type)


def ResNet20(num_classes=10, channels=1, loss_type='fedavg'):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, channels, loss_type)


class MobileNetBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MobileNetBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_Tiny(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # stride 1 for 32x32 images
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, channels=1, loss_type='fedavg', width_mult=0.5):
        super(MobileNetV2_Tiny, self).__init__()
        
        # Stride 1 instead of 2 for first layer because images are small
        self.conv1 = nn.Conv2d(channels, int(32*width_mult), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*width_mult))
        
        self.layers = self._make_layers(in_planes=int(32*width_mult), width_mult=width_mult)
        
        self.conv2 = nn.Conv2d(int(320*width_mult), int(1280*width_mult), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(int(1280*width_mult))
        self.linear = nn.Linear(int(1280*width_mult), num_classes)
        self.loss_type = loss_type

    def _make_layers(self, in_planes, width_mult):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            out_planes = int(out_planes * width_mult)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(MobileNetBlock(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = F.avg_pool2d(out, out.size()[3])
        x_out = out.view(out.size(0), -1)
        out = self.linear(x_out)
        
        if self.loss_type == 'fedmax':
            return out, x_out
        else:
            return out


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.a3 = Inception(64,  32, 48, 64, 8, 16, 16)
        self.b3 = Inception(128, 64, 64, 96, 16, 48, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(240, 96,  48, 104, 8,  24, 32)
        self.b4 = Inception(256, 80,  56, 112, 12, 32, 32)

        self.linear = nn.Linear(256, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        
        out = F.avg_pool2d(out, out.size()[3])
        x_out = out.view(out.size(0), -1)
        out = self.linear(x_out)
        
        if self.loss_type == 'fedmax':
            return out, x_out
        else:
            return out


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, channels=1, loss_type='fedavg'):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Linear(128, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_out = self.classifier(x)
        out = self.fc3(x_out)
        if self.loss_type == 'fedmax':
            return out, x_out
        else:
            return out