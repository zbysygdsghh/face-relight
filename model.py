import torch.nn as nn
import torch

class U_Net(nn.Module):
    def __init__(self, inchannel=3, outchannel=3):
        super(U_Net, self, ).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(inchannel, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv8_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv9_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(16, 12, kernel_size=1, stride=1)
        self.conv10_2 = nn.Conv2d(12, outchannel, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        out = self.conv10_2(conv10)

        # out2,out3 = self.Conv11(out1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

class Enhance0(nn.Module):
    def __init__(self, num_classes=10):
        super(Enhance0, self).__init__()

        self.unet = U_Net(inchannel=3, outchannel=3)
        self.unet0 = U_Net(inchannel=6, outchannel=3)
        self.unet1 = U_Net(inchannel=3, outchannel=3)
        self.unet2 = U_Net(inchannel=6, outchannel=3)
        self.unet3 = U_Net(inchannel=6, outchannel=3)
        self.unet4 = U_Net(inchannel=3, outchannel=3)
        self.conv0 = nn.Conv2d(3, 16, kernel_size=1, stride=1)
        self.conv0_0 = nn.Conv2d(3, 16, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(16, 16, kernel_size=1, stride=1)
        self.conv1_0 = nn.Conv2d(16, 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=1, stride=1)
        self.conv2_0 = nn.Conv2d(16, 3, kernel_size=1, stride=1)

    def forward(self, x0, x1, x2):
        outc = self.unet(x1)
        out_t = self.unet0(torch.cat((x1, outc), 1))
        out_mask = self.unet1(x0)
        out = self.unet2(torch.cat((x0, out_t), 1))
        out1 = self.unet3(torch.cat((out, out_mask), 1))
        out = out1 + x0
        out_sima0 = self.unet4(x2)
        out_sima1 = self.unet4(out)

        return out, out_mask, out_t, outc, out_sima0, out_sima1

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt





