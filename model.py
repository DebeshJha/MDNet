import torch
import torch.nn as nn
import torch.nn.functional as F
from mit import MiT
from cbam import cbam
import numpy as np
import cv2

def save_feats_mean(x, size=(512, 512)):
    b, c, h, w = x.shape
    with torch.no_grad():
        x = x.detach().cpu().numpy()
        x = np.transpose(x[0], (1, 2, 0))
        x = np.mean(x, axis=-1)
        x = x/np.max(x)
        x = x * 255.0
        x = x.astype(np.uint8)

        if h != size[1]:
            x = cv2.resize(x, size)

        x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
        x = np.array(x, dtype=np.uint8)
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()

        layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_c),
        )
        if act == True:
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.conv(inputs)

class dilated_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = conv_block(in_c, out_c, dilation=1, padding=1)
        self.c2 = conv_block(in_c, out_c, dilation=3, padding=3)
        self.c3 = conv_block(in_c, out_c, dilation=6, padding=6)
        self.c4 = conv_block(in_c, out_c, dilation=9, padding=9)

        self.c5 = conv_block(out_c*4, out_c, kernel_size=1, padding=0, dilation=1)
        self.cbam = cbam(out_c)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.c5(x)
        x = self.cbam(x)

        return x

class feature_enhancement_dilated_block_f2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))
        self.c1 = conv_block(in_c[0], out_c)
        self.c2 = conv_block(out_c+in_c[1], out_c)
        self.db = dilated_block(out_c, out_c)

    def forward(self, f1, f2):
        x = self.pool(self.c1(f1))
        x = torch.cat([x, f2], dim=1)

        x = self.c2(x)
        x = self.db(x)

        return x

class feature_enhancement_dilated_block_f3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))
        self.c1 = conv_block(in_c[0], out_c)
        self.c2 = conv_block(out_c+in_c[1], out_c)
        self.c3 = conv_block(out_c+in_c[2], out_c)
        self.db = dilated_block(out_c, out_c)

    def forward(self, f1, f2, f3):
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        x = self.pool(self.c1(f1))
        x = torch.cat([x, f2], dim=1)

        x = self.pool(self.c2(x))
        x = torch.cat([x, f3], dim=1)

        x = self.c3(x)
        x = self.db(x)

        return x

class feature_enhancement_dilated_block_f4(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))
        self.c1 = conv_block(in_c[0], out_c)
        self.c2 = conv_block(out_c+in_c[1], out_c)
        self.c3 = conv_block(out_c+in_c[2], out_c)
        self.c4 = conv_block(out_c+in_c[3], out_c)
        self.db = dilated_block(out_c, out_c)

    def forward(self, f1, f2, f3, f4):
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        x = self.pool(self.c1(f1))
        x = torch.cat([x, f2], dim=1)

        x = self.pool(self.c2(x))
        x = torch.cat([x, f3], dim=1)

        x = self.pool(self.c3(x))
        x = torch.cat([x, f4], dim=1)

        x = self.c4(x)
        x = self.db(x)

        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        """ Conv Block Layers """
        self.conv1 = nn.Sequential(
            conv_block(in_c, out_c),
            conv_block(out_c, out_c, act=False)
        )
        self.conv2 = conv_block(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """ Conv """
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        """ Addition """
        x3 = self.relu(x1 + x2)
        return x3

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c[0], in_c[0], kernel_size=2, padding=0, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = nn.Sequential(
            residual_block(in_c[0]+in_c[1], out_c),
            residual_block(out_c, out_c)
        )

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)
        return x


class mask_attention_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = conv_block(out_c, out_c)
        self.c2 = conv_block(out_c, out_c)
        self.c3 = conv_block(out_c*2, out_c, act=False)
        self.c4 = conv_block(in_c, out_c, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        """ Mask processing """
        mask = F.interpolate(mask, size=(x.shape[2:]))
        mask = torch.sigmoid(mask)

        """ Foreground Attention """
        xf = self.c1(x * mask) + mask

        """ Background Attention """
        xb = self.c2(x * (1 - mask)) + mask

        """ Combining both """
        xc = torch.cat([xf, xb], dim=1)
        xc = self.c3(xc)
        xs = self.c4(x)

        x = self.relu(xc + xs)
        return x


class Model(nn.Module):
    def __init__(self,
            name="B2",
            checkpoint_path="MiT/mit_b2.pth",
            image_size=512,
            num_classes=1
        ):
        super().__init__()

        """ MiT Encoder """
        self.image_encoder = MiT("B2")
        save_model = torch.load(checkpoint_path)
        model_dict = self.image_encoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.image_encoder.load_state_dict(model_dict)

        """ Bridge """
        self.db = dilated_block(64, 64)
        self.db_f2 = feature_enhancement_dilated_block_f2([64, 128], 128)
        self.db_f3 = feature_enhancement_dilated_block_f3([64, 128, 320], 320)
        self.db_f4 = feature_enhancement_dilated_block_f4([64, 128, 320, 512], 512)

        """ Decoder 1 """
        self.d11 = decoder_block([128, 64], 64)
        self.m1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

        """ Decoder 2 """
        self.d21 = decoder_block([320, 128], 128)
        self.a21 = mask_attention_block(128, 128)

        self.d22 = decoder_block([128, 64*2], 64)
        self.a22 = mask_attention_block(64, 64)

        self.m2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

        """ Decoder 3 """
        self.d31 = decoder_block([512, 320], 320)
        self.a31 = mask_attention_block(320, 320)

        self.d32 = decoder_block([320, 128*2], 128)
        self.a32 = mask_attention_block(128, 128)

        self.d33 = decoder_block([128, 64*3], 64)
        self.a33 = mask_attention_block(64, 64)

        self.m3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )


    def forward(self, image, heatmap=False):
        """ MiT Encoder """
        f1, f2, f3, f4 = self.image_encoder(image)
        # print(f1.shape, f2.shape, f3.shape, f4.shape)

        """ Bridge """
        f1 = self.db(f1)
        f2 = self.db_f2(f1, f2)
        f3 = self.db_f3(f1, f2, f3)
        f4 = self.db_f4(f1, f2, f3, f4)

        """ Decoder 1 """
        d11 = self.d11(f2, f1)
        m1 = self.m1(d11)

        """ Decoder 2 """
        d21 = self.d21(f3, f2)
        a21 = self.a21(d21, m1)

        cx = torch.cat([f1, d11], dim=1)
        d22 = self.d22(a21, cx)
        a22 = self.a22(d22, m1)

        m2 = self.m2(a22)

        """ Decoder 3 """
        d31 = self.d31(f4, f3)
        a31 = self.a31(d31, m2)

        cx = torch.cat([f2, a21], dim=1)
        d32 = self.d32(a31, cx)
        a32 = self.a32(d32, m2)

        cx = torch.cat([f1, d11, a22], dim=1)
        d33 = self.d33(d32, cx)
        a33 = self.a33(d33, m2)

        m3 = self.m3(a33)

        if heatmap == True:
            d11 = save_feats_mean(d11)
            d22 = save_feats_mean(d22)
            a22 = save_feats_mean(a22)
            d33 = save_feats_mean(d33)
            a33 = save_feats_mean(a33)
            return [m1, m2, m3], [d11, d22, a22, d33, a33]
        else:
            return [m1, m2, m3]


if __name__ == "__main__":
    image = torch.randn((8, 3, 512, 512))
    model = Model()

    m1, m2, m3 = model(image)
    print(m1.shape, m2.shape, m3.shape)
    
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3, 512, 512), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
