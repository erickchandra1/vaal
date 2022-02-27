from .parts import *
import torch.nn as nn


class UnetClassifierTail(nn.Module):
    def __init__(self, num_classes, input_feature_dim):
        super(UnetClassifierTail, self).__init__()
        # self.fc1 = nn.Linear(input_feature_dim, int(input_feature_dim // 16))
        self.out = nn.Linear(int(input_feature_dim), num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        output = self.out(x)
        return output


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)

        # self.classifier = UnetClassifierTail(num_classes=n_classes, input_feature_dim=512 * 14 * 14) # 512 * 4 * 4
        self.classifier = UnetClassifierTail(num_classes=n_classes, input_feature_dim=256 * 4 * 4)

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        return None, x1, x2, x3, x4 # x5

    def decoder(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def classifier_tail(self, x):
        out = self.classifier(x)
        return out

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        # logits = self.decoder(x1, x2, x3, x4, x5)
        logits = self.classifier_tail(x5)
        return logits
