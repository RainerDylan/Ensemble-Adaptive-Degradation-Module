import torch
import torch.nn as nn
import math

class Bottle2neck(nn.Module):
    """Res2Net Bottleneck Block"""
    def __init__(self, inplanes, planes, stride=1, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (scale - 1) / scale))
        
        # FIX: Apply stride here (in conv1) instead of inside the loop
        # This ensures all splits (spx) are the same size immediately
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        
        self.nums = scale - 1
        self.convs = nn.ModuleList([
            # FIX: Stride is strictly 1 here because data is already downsampled by conv1
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False) 
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])
        
        self.conv3 = nn.Conv2d(width * scale, planes * scale, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * scale)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or inplanes != planes * scale:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * scale, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * scale),
            )
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x
        
        # Conv1 now handles downsampling if stride > 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Now spx chunks are all consistent size
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i] # Dimensions now match perfectly
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        out = torch.cat((out, spx[self.scale-1]), 1)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Res2Net(nn.Module):
    """
    Res2Net Model for Spectrograms
    Input: (Batch, 1, Time, Freq)
    Output: (Batch, 2)
    """
    def __init__(self, scale=4, baseWidth=26, layers=[3, 4, 6, 3], num_classes=2):
        super(Res2Net, self).__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        
        # Initial Conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layers
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 512 * scale (4) = 2048 input features for classifier
        self.fc = nn.Linear(512 * scale, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(Bottle2neck(self.inplanes, planes, stride, self.scale))
        self.inplanes = planes * self.scale
        for _ in range(1, blocks):
            layers.append(Bottle2neck(self.inplanes, planes, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x: (B, 1, T, F)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, None