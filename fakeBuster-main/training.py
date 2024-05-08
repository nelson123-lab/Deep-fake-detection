import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2d3d_full(nn.Module):
    def __init__(self, block2d, block3d, num_blocks, track_running_stats=True):
        super(ResNet2d3d_full, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer2d(block2d, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer2d(block2d, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer3d(block3d, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer3d(block3d, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.apply(self._init_weights)

    def _make_layer2d(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer3d(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class AudioRNN(nn.Module):
    def __init__(self, img_dim, network='resnet50', num_layers_in_fc_layers=1024, dropout=0.5, winLength=30):
        super(AudioRNN, self).__init__()
        self.__nFeatures__ = winLength
        self.__nChs__ = 32
        self.__midChs__ = 32

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512 * 21, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_layers_in_fc_layers),
        )

        self.netcnnlip = ResNet2d3d_full(BasicBlock2d, BasicBlock3d, [2, 2, 2, 2])
        self.last_duration = int(math.ceil(self.__nFeatures__ / 4))
        self.last_size = int(math.ceil(img_dim / 32))

        self.netfclip = nn.Sequential(
            nn.Linear(256 * self.last_size * self.last_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_layers_in_fc_layers),
        )

        self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_lip.weight.data.fill_(1)
        self.final_bn_lip.bias.data.zero_()

        self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        self._initialize_weights(self.final_fc_lip)

        self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_aud.weight.data.fill_(1)
        self.final_bn_aud.bias.data.zero_()

        self.final_fc_aud = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        self._initialize_weights(self.final_fc_aud)

        self._initialize_weights(self.netcnnaud)
        self._initialize_weights(self.netfcaud)
        self._initialize_weights(self.netfclip)

    def forward_aud(self, x):
        (B, N, N, H, W) = x.shape
        x = x.view(B * N, N, H, W)
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfcaud(mid)
        return out

    def forward_lip(self, x):
        (B, N, C, NF, H, W) = x.shape
        x = x.view(B * N, C, NF, H, W)
        feature = self.netcnnlip(x)
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        feature = feature.view(B, N, 256, self.last_size, self.last_size)
        feature = feature.view((feature.size()[0], -1))
        out = self.netfclip(feature)
        return out

    def final_classification_lip(self, feature):
        feature = self.final_bn_lip(feature)
        output = self.final_fc_lip(feature)
        return output

    def final_classification_aud(self, feature):
        feature = self.final_bn_aud(feature)
        output = self.final_fc_aud(feature)
        return output

    def forward_lipfeat(self, x):
        mid = self.netcnnlip(x)
        out = mid.view((mid.size()[0], -1))
        return out

    def _initialize_weights(self, module):
        for m in module:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Dropout):
                pass
            else:
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
