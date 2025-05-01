import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# models.py

import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)

        self.pool = nn.MaxPool3d(2)

        self.up2 = UpBlock(base_filters * 8, base_filters * 4)
        self.up1 = UpBlock(base_filters * 4, base_filters * 2)
        self.up0 = UpBlock(base_filters * 2, base_filters)

        self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d2 = self.up2(b, e3)
        d1 = self.up1(d2, e2)
        d0 = self.up0(d1, e1)
        out = self.final(d0)
        return self.softmax(out)




# _____________________ classification for CIFAR under here

class ResNet9(nn.Module):
    def __init__(self):
        super(ResNet9, self).__init__()
        self.prep = self.convbnrelu(channels=3, filters=64)
        self.layer1 = self.convbnrelu(64, 128)
        self.layer_pool = nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False)
        self.layer1r1 = self.convbnrelu(128, 128)
        self.layer1r2 = self.convbnrelu(128, 128)
        self.layer2 = self.convbnrelu(128, 256)
        self.layer3 = self.convbnrelu(256, 512)
        self.layer3r1 = self.convbnrelu(512, 512)
        self.layer3r2 = self.convbnrelu(512, 512)
        self.out_pool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=512, out_features=10, bias=False)

    def convbnrelu(self, channels, filters):
        layers = []
        layers.append(nn.Conv2d(channels, filters, (3, 3),
                                (1, 1), (1, 1), bias=False))
        layers.append(nn.BatchNorm2d(filters, track_running_stats=False))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer_pool(self.layer1(x))
        r1 = self.layer1r2(self.layer1r1(x)) 
        x = x + r1
        x = self.layer_pool(self.layer2(x))
        x = self.layer_pool(self.layer3(x))
        r3 = self.layer3r2(self.layer3r1(x))
        x = x + r3
        out = self.out_pool(x)
        out = self.flatten(out)
        out = self.linear(out)
        out = out * 0.125

        return out
        
class LocalUpdate(object):

    def __init__(self, lr, local_ep, trainloader):
        self.lr = lr
        self.local_ep = local_ep
        self.trainloader = trainloader

    def update_weights(self, model):

        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)
        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()   
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)