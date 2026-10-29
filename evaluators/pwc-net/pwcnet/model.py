"""PWC-DC Net — pure-PyTorch re-implementation.

Drop-in replacement for the original that requires no CUDA-extension build.
Correlation is computed with torch.nn.functional.unfold.
Layer names are identical to the original so pretrained .pth.tar weights load
directly with load_checkpoint().

Reference: Sun et al., PWC-Net: CNNs for Optical Flow Using Pyramid, Warping,
           and Cost Volume, CVPR 2018.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def _deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def _correlation(x1, x2, max_displacement=4):
    """Channel-normalised correlation with all (2d+1)^2 displacements.

    Matches the Caffe correlation layer which divides by the number of channels
    (kernel_size=1, corr_multiply=1 in the original PWC-Net config).
    """
    B, C, H, W = x1.shape
    d = max_displacement
    x2_pad = F.pad(x2, [d, d, d, d])
    patches = F.unfold(x2_pad, kernel_size=2 * d + 1, padding=0, stride=1)
    # patches: B x (C*(2d+1)^2) x (H*W)
    n_displ = (2 * d + 1) ** 2
    patches = patches.view(B, C, n_displ, H * W)
    x1_flat = x1.view(B, C, 1, H * W)
    corr = (x1_flat * patches).sum(dim=1) / C  # B x n_displ x (H*W)
    return corr.view(B, n_displ, H, W)


def _warp(x, flo):
    """Warp x according to optical flow flo (both at the same spatial resolution)."""
    B, C, H, W = x.shape
    xx = torch.arange(0, W, device=x.device, dtype=x.dtype).view(1, -1).expand(H, -1)
    yy = torch.arange(0, H, device=x.device, dtype=x.dtype).view(-1, 1).expand(-1, W)
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    vgrid = grid + flo
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, align_corners=True, padding_mode='zeros')


class PWCDCNet(nn.Module):
    """PWC-DC network (dense connections + dilated context)."""

    def __init__(self, md: int = 4):
        super().__init__()
        self.md = md

        # Feature pyramid
        self.conv1a  = _conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = _conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = _conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = _conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = _conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = _conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = _conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = _conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = _conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = _conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = _conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = _conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = _conv(96,  128, kernel_size=3, stride=2)
        self.conv5aa = _conv(128, 128, kernel_size=3, stride=1)
        self.conv5b  = _conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = _conv(128, 196, kernel_size=3, stride=2)
        self.conv6a  = _conv(196, 196, kernel_size=3, stride=1)
        self.conv6b  = _conv(196, 196, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        nd = (2 * md + 1) ** 2
        import numpy as np
        dd = np.cumsum([128, 128, 96, 64, 32])

        # Level 6 decoder
        od = nd
        self.conv6_0 = _conv(od,        128)
        self.conv6_1 = _conv(od+dd[0],  128)
        self.conv6_2 = _conv(od+dd[1],  96)
        self.conv6_3 = _conv(od+dd[2],  64)
        self.conv6_4 = _conv(od+dd[3],  32)
        self.predict_flow6 = _predict_flow(od + dd[4])
        self.deconv6 = _deconv(2, 2)
        self.upfeat6 = _deconv(od + dd[4], 2)

        # Level 5 decoder
        od = nd + 128 + 4
        self.conv5_0 = _conv(od,        128)
        self.conv5_1 = _conv(od+dd[0],  128)
        self.conv5_2 = _conv(od+dd[1],  96)
        self.conv5_3 = _conv(od+dd[2],  64)
        self.conv5_4 = _conv(od+dd[3],  32)
        self.predict_flow5 = _predict_flow(od + dd[4])
        self.deconv5 = _deconv(2, 2)
        self.upfeat5 = _deconv(od + dd[4], 2)

        # Level 4 decoder
        od = nd + 96 + 4
        self.conv4_0 = _conv(od,        128)
        self.conv4_1 = _conv(od+dd[0],  128)
        self.conv4_2 = _conv(od+dd[1],  96)
        self.conv4_3 = _conv(od+dd[2],  64)
        self.conv4_4 = _conv(od+dd[3],  32)
        self.predict_flow4 = _predict_flow(od + dd[4])
        self.deconv4 = _deconv(2, 2)
        self.upfeat4 = _deconv(od + dd[4], 2)

        # Level 3 decoder
        od = nd + 64 + 4
        self.conv3_0 = _conv(od,        128)
        self.conv3_1 = _conv(od+dd[0],  128)
        self.conv3_2 = _conv(od+dd[1],  96)
        self.conv3_3 = _conv(od+dd[2],  64)
        self.conv3_4 = _conv(od+dd[3],  32)
        self.predict_flow3 = _predict_flow(od + dd[4])
        self.deconv3 = _deconv(2, 2)
        self.upfeat3 = _deconv(od + dd[4], 2)

        # Level 2 decoder
        od = nd + 32 + 4
        self.conv2_0 = _conv(od,        128)
        self.conv2_1 = _conv(od+dd[0],  128)
        self.conv2_2 = _conv(od+dd[1],  96)
        self.conv2_3 = _conv(od+dd[2],  64)
        self.conv2_4 = _conv(od+dd[3],  32)
        self.predict_flow2 = _predict_flow(od + dd[4])

        # Dilated context network
        self.dc_conv1 = _conv(od+dd[4], 128, padding=1,  dilation=1)
        self.dc_conv2 = _conv(128,       128, padding=2,  dilation=2)
        self.dc_conv3 = _conv(128,       128, padding=4,  dilation=4)
        self.dc_conv4 = _conv(128,        96, padding=8,  dilation=8)
        self.dc_conv5 = _conv(96,          64, padding=16, dilation=16)
        self.dc_conv6 = _conv(64,          32, padding=1,  dilation=1)
        self.dc_conv7 = _predict_flow(32)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: B x 6 x H x W — concatenated [im1, im2] in [0, 1].
        Returns flow2 at H/4 x W/4, in normalised units (divide by 20 to get
        pixel-space flow at the input resolution).
        """
        im1, im2 = x[:, :3], x[:, 3:]
        md = self.md

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        corr6 = self.leakyRELU(_correlation(c16, c26, md))

        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6    = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = _warp(c25, up_flow6 * 0.625)
        corr5 = self.leakyRELU(_correlation(c15, warp5, md))
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5    = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = _warp(c24, up_flow5 * 1.25)
        corr4 = self.leakyRELU(_correlation(c14, warp4, md))
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4    = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = _warp(c23, up_flow4 * 2.5)
        corr3 = self.leakyRELU(_correlation(c13, warp3, md))
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3    = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = _warp(c22, up_flow3 * 5.0)
        corr2 = self.leakyRELU(_correlation(c12, warp2, md))
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        return flow2


def load_checkpoint(path: str, device: str = 'cpu') -> PWCDCNet:
    """Load a pretrained PWCDCNet from a .pth.tar checkpoint."""
    model = PWCDCNet()
    data = torch.load(path, map_location=device, weights_only=False)
    state = data['state_dict'] if isinstance(data, dict) and 'state_dict' in data else data
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[PWCNet] missing keys: {missing}")
    if unexpected:
        print(f"[PWCNet] unexpected keys: {unexpected}")
    return model


def infer(model: PWCDCNet, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
    """Run inference on a pair of images.

    Args:
        im1, im2: B x 3 x H x W float32 tensors with values in [0, 255].
                  Tensors are moved to the model's device automatically.
    Returns:
        flow: B x 2 x H x W in pixel units (same spatial size as input).
    """
    device = next(model.parameters()).device
    im1 = im1.to(device)
    im2 = im2.to(device)
    B, C, H, W = im1.shape

    # Normalise
    im1 = im1 / 255.0
    im2 = im2 / 255.0

    # Pad to multiples of 64 (required by the 6-level pyramid)
    H64 = math.ceil(H / 64) * 64
    W64 = math.ceil(W / 64) * 64
    pad_h, pad_w = H64 - H, W64 - W
    if pad_h or pad_w:
        im1 = F.pad(im1, [0, pad_w, 0, pad_h])
        im2 = F.pad(im2, [0, pad_w, 0, pad_h])

    with torch.no_grad():
        flow_low = model(torch.cat([im1, im2], dim=1))  # B x 2 x H64/4 x W64/4

    # De-normalise (PWC-Net trains with flow / 20)
    flow_low = flow_low * 20.0

    # Upsample to full padded resolution; flow values are already in pixel units
    flow = F.interpolate(flow_low, size=(H64, W64), mode='bilinear', align_corners=False)

    # Crop padding
    if pad_h or pad_w:
        flow = flow[:, :, :H, :W]

    return flow
