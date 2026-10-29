"""VCN (full, 5-level) — patched for modern PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/'
import numpy as np
import math

from .submodule import pspnet, conv
from .conv4d import sepConv4d, butterfly4D


class flow_reg(nn.Module):
    """Soft WTA displacement selection."""
    def __init__(self, size, ent=False, maxdisp=4, fac=1):
        B, W, H = size
        super(flow_reg, self).__init__()
        self.ent = ent
        self.md = maxdisp
        self.fac = fac
        self.truncated = True
        self.wsize = 3

        flowrangey = range(-maxdisp, maxdisp + 1)
        flowrangex = range(-int(maxdisp // self.fac), int(maxdisp // self.fac) + 1)
        meshgrid = np.meshgrid(flowrangex, flowrangey)
        flowy = np.tile(np.reshape(meshgrid[0], [1, 2 * maxdisp + 1, 2 * int(maxdisp // self.fac) + 1, 1, 1]), (B, 1, 1, H, W))
        flowx = np.tile(np.reshape(meshgrid[1], [1, 2 * maxdisp + 1, 2 * int(maxdisp // self.fac) + 1, 1, 1]), (B, 1, 1, H, W))
        self.register_buffer('flowx', torch.Tensor(flowx))
        self.register_buffer('flowy', torch.Tensor(flowy))
        self.pool3d = nn.MaxPool3d((self.wsize * 2 + 1, self.wsize * 2 + 1, 1), stride=1, padding=(self.wsize, self.wsize, 0))

    def forward(self, x):
        b, u, v, h, w = x.shape
        oldx = x
        if self.truncated:
            x = x.view(b, u * v, h, w)
            idx = x.argmax(1)[:, np.newaxis]
            mask = torch.zeros(b, u * v, h, w, device=x.device, dtype=x.dtype)
            mask.scatter_(1, idx, 1)
            mask = mask.view(b, 1, u, v, -1)
            mask = self.pool3d(mask)[:, 0].view(b, u, v, h, w)
            ninf = x.clone().fill_(-np.inf).view(b, u, v, h, w)
            x = torch.where(mask.bool(), oldx, ninf)
        b, u, v, h, w = x.shape
        x = F.softmax(x.view(b, -1, h, w), 1).view(b, u, v, h, w)
        outx = torch.sum(torch.sum(x * self.flowx, 1), 1, keepdim=True)
        outy = torch.sum(torch.sum(x * self.flowy, 1), 1, keepdim=True)
        if self.ent:
            local_entropy = (-x * torch.clamp(x, 1e-9, 1 - 1e-9).log()).sum(1).sum(1)[:, np.newaxis]
            if self.wsize == 0:
                local_entropy[:] = 1.
            else:
                local_entropy /= np.log((self.wsize * 2 + 1) ** 2)
            x = F.softmax(oldx.view(b, -1, h, w), 1).view(b, u, v, h, w)
            global_entropy = (-x * torch.clamp(x, 1e-9, 1 - 1e-9).log()).sum(1).sum(1)[:, np.newaxis]
            global_entropy /= np.log(x.shape[1] * x.shape[2])
            return torch.cat([outx, outy], 1), torch.cat([local_entropy, global_entropy], 1)
        else:
            return torch.cat([outx, outy], 1), None


class WarpModule(nn.Module):
    def __init__(self, size):
        super(WarpModule, self).__init__()
        B, W, H = size
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        self.register_buffer('grid', torch.cat((xx, yy), 1).float())

    def forward(self, x, flo):
        B, C, H, W = x.size()
        vgrid = self.grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        mask = ((vgrid[:, :, :, 0].abs() < 1) * (vgrid[:, :, :, 1].abs() < 1)) > 0
        return output * mask.unsqueeze(1).float(), mask


class VCN(nn.Module):
    """VCN full model (5 levels)."""
    def __init__(self, size, md=[4, 4, 4, 4, 4], fac=1.):
        super(VCN, self).__init__()
        self.md = md
        self.fac = fac
        use_entropy = True
        withbn = True
        self.pspnet = pspnet(is_proj=False)
        fdima1 = 128
        fdima2 = 64
        fdimb1 = 16
        fdimb2 = 12
        full = False
        self.f6 = butterfly4D(fdima1, fdimb1, withbn=withbn, full=full)
        self.p6 = sepConv4d(fdimb1, fdimb1, with_bn=False, full=full)
        self.f5 = butterfly4D(fdima1, fdimb1, withbn=withbn, full=full)
        self.p5 = sepConv4d(fdimb1, fdimb1, with_bn=False, full=full)
        self.f4 = butterfly4D(fdima1, fdimb1, withbn=withbn, full=full)
        self.p4 = sepConv4d(fdimb1, fdimb1, with_bn=False, full=full)
        self.f3 = butterfly4D(fdima2, fdimb1, withbn=withbn, full=full)
        self.p3 = sepConv4d(fdimb1, fdimb1, with_bn=False, full=full)
        full = True
        self.f2 = butterfly4D(fdima2, fdimb2, withbn=withbn, full=full)
        self.p2 = sepConv4d(fdimb2, fdimb2, with_bn=False, full=full)
        self.flow_reg64 = flow_reg([fdimb1 * size[0], size[1] // 64, size[2] // 64], ent=use_entropy, maxdisp=self.md[0], fac=self.fac)
        self.flow_reg32 = flow_reg([fdimb1 * size[0], size[1] // 32, size[2] // 32], ent=use_entropy, maxdisp=self.md[1])
        self.flow_reg16 = flow_reg([fdimb1 * size[0], size[1] // 16, size[2] // 16], ent=use_entropy, maxdisp=self.md[2])
        self.flow_reg8  = flow_reg([fdimb1 * size[0], size[1] // 8,  size[2] // 8],  ent=use_entropy, maxdisp=self.md[3])
        self.flow_reg4  = flow_reg([fdimb2 * size[0], size[1] // 4,  size[2] // 4],  ent=use_entropy, maxdisp=self.md[4])
        self.warp5 = WarpModule([size[0], size[1] // 32, size[2] // 32])
        self.warp4 = WarpModule([size[0], size[1] // 16, size[2] // 16])
        self.warp3 = WarpModule([size[0], size[1] // 8,  size[2] // 8])
        self.warp2 = WarpModule([size[0], size[1] // 4,  size[2] // 4])
        self.dc6_conv1 = conv(128 + 4 * fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc6_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc6_conv4 = conv(128, 96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc6_conv5 = conv(96,  64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc6_conv6 = conv(64,  32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv7 = nn.Conv2d(32, 2 * fdimb1, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc5_conv1 = conv(128 + 4 * fdimb1 * 2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc5_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc5_conv4 = conv(128, 96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc5_conv5 = conv(96,  64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc5_conv6 = conv(64,  32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv7 = nn.Conv2d(32, 2 * fdimb1 * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc4_conv1 = conv(128 + 4 * fdimb1 * 3, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc4_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc4_conv4 = conv(128, 96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc4_conv5 = conv(96,  64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc4_conv6 = conv(64,  32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv7 = nn.Conv2d(32, 2 * fdimb1 * 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc3_conv1 = conv(64 + 16 * fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc3_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc3_conv4 = conv(128, 96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc3_conv5 = conv(96,  64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc3_conv6 = conv(64,  32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv7 = nn.Conv2d(32, 8 * fdimb1, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc2_conv1 = conv(64 + 16 * fdimb1 + 4 * fdimb2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc2_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc2_conv4 = conv(128, 96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc2_conv5 = conv(96,  64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc2_conv6 = conv(64,  32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv7 = nn.Conv2d(32, 4 * 2 * fdimb1 + 2 * fdimb2, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def corrf(self, refimg_fea, targetimg_fea, maxdisp, fac=1):
        b, c, height, width = refimg_fea.shape
        cost = torch.zeros(b, c, 2 * maxdisp + 1, 2 * int(maxdisp // fac) + 1, height, width,
                           device=refimg_fea.device, dtype=refimg_fea.dtype)
        for i in range(2 * maxdisp + 1):
            ind = i - maxdisp
            for j in range(2 * int(maxdisp // fac) + 1):
                indd = j - int(maxdisp // fac)
                feata = refimg_fea[:, :, max(0, -indd):height - indd, max(0, -ind):width - ind]
                featb = targetimg_fea[:, :, max(0, +indd):height + indd, max(0, ind):width + ind]
                cost[:, :, i, j, max(0, -indd):height - indd, max(0, -ind):width - ind] = feata * featb
        cost = F.leaky_relu(cost, 0.1, inplace=True)
        return cost

    def forward(self, im, disc_aux=None):
        bs = im.shape[0] // 2
        c06, c05, c04, c03, c02 = self.pspnet(im)
        c16 = c06[:bs]; c26 = c06[bs:]
        c15 = c05[:bs]; c25 = c05[bs:]
        c14 = c04[:bs]; c24 = c04[bs:]
        c13 = c03[:bs]; c23 = c03[bs:]
        c12 = c02[:bs]; c22 = c02[bs:]
        c16n = c16 / (c16.norm(dim=1, keepdim=True) + 1e-9)
        c26n = c26 / (c26.norm(dim=1, keepdim=True) + 1e-9)
        c15n = c15 / (c15.norm(dim=1, keepdim=True) + 1e-9)
        c25n = c25 / (c25.norm(dim=1, keepdim=True) + 1e-9)
        c14n = c14 / (c14.norm(dim=1, keepdim=True) + 1e-9)
        c24n = c24 / (c24.norm(dim=1, keepdim=True) + 1e-9)
        c13n = c13 / (c13.norm(dim=1, keepdim=True) + 1e-9)
        c23n = c23 / (c23.norm(dim=1, keepdim=True) + 1e-9)
        c12n = c12 / (c12.norm(dim=1, keepdim=True) + 1e-9)
        c22n = c22 / (c22.norm(dim=1, keepdim=True) + 1e-9)

        feat6 = self.corrf(c16n, c26n, self.md[0], fac=self.fac)
        feat6 = self.f6(feat6)
        cost6 = self.p6(feat6)
        b, c, u, v, h, w = cost6.shape
        cost6 = cost6.view(-1, u, v, h, w)
        flow6h, ent6h = self.flow_reg64(cost6)
        flow6h = flow6h.view(bs, -1, h, w)
        ent6h  = ent6h.view(bs, -1, h, w)
        x = torch.cat((ent6h.detach(), flow6h.detach(), c16), 1)
        x = self.dc6_conv4(self.dc6_conv3(self.dc6_conv2(self.dc6_conv1(x))))
        va = self.dc6_conv7(self.dc6_conv6(self.dc6_conv5(x)))
        va = va.view(b, -1, 2, h, w)
        flow6 = (flow6h.view(b, -1, 2, h, w) * F.softmax(va, 1)).sum(1)

        up_flow6 = F.interpolate(flow6, [im.size()[2] // 32, im.size()[3] // 32], mode='bilinear', align_corners=False) * 2
        warp5, _ = self.warp5(c25n, up_flow6)
        feat5 = self.corrf(c15n, warp5, self.md[1])
        feat5 = self.f5(feat5)
        cost5 = self.p5(feat5)
        b, c, u, v, h, w = cost5.shape
        cost5 = cost5.view(-1, u, v, h, w)
        flow5h, ent5h = self.flow_reg32(cost5)
        flow5h = flow5h.view(b, c, 2, h, w) + up_flow6[:, np.newaxis]
        flow5h = flow5h.view(bs, -1, h, w)
        ent5h  = ent5h.view(bs, -1, h, w)
        flow5h = torch.cat((flow5h, F.interpolate(flow6h.detach() * 2, [flow5h.shape[2], flow5h.shape[3]], mode='bilinear', align_corners=False)), 1)
        ent5h  = torch.cat((ent5h,  F.interpolate(ent6h, [flow5h.shape[2], flow5h.shape[3]], mode='bilinear', align_corners=False)), 1)
        x = torch.cat((ent5h.detach(), flow5h.detach(), c15), 1)
        x = self.dc5_conv4(self.dc5_conv3(self.dc5_conv2(self.dc5_conv1(x))))
        va5 = self.dc5_conv7(self.dc5_conv6(self.dc5_conv5(x)))
        va5 = va5.view(b, -1, 2, h, w)
        flow5 = (flow5h.view(b, -1, 2, h, w) * F.softmax(va5, 1)).sum(1)

        up_flow5 = F.interpolate(flow5, [im.size()[2] // 16, im.size()[3] // 16], mode='bilinear', align_corners=False) * 2
        warp4, _ = self.warp4(c24n, up_flow5)
        feat4 = self.corrf(c14n, warp4, self.md[2])
        feat4 = self.f4(feat4)
        cost4 = self.p4(feat4)
        b, c, u, v, h, w = cost4.shape
        cost4 = cost4.view(-1, u, v, h, w)
        flow4h, ent4h = self.flow_reg16(cost4)
        flow4h = flow4h.view(b, c, 2, h, w) + up_flow5[:, np.newaxis]
        flow4h = flow4h.view(bs, -1, h, w)
        ent4h  = ent4h.view(bs, -1, h, w)
        flow4h = torch.cat((flow4h, F.interpolate(flow5h.detach() * 2, [flow4h.shape[2], flow4h.shape[3]], mode='bilinear', align_corners=False)), 1)
        ent4h  = torch.cat((ent4h,  F.interpolate(ent5h, [flow4h.shape[2], flow4h.shape[3]], mode='bilinear', align_corners=False)), 1)
        x = torch.cat((ent4h.detach(), flow4h.detach(), c14), 1)
        x = self.dc4_conv4(self.dc4_conv3(self.dc4_conv2(self.dc4_conv1(x))))
        va = self.dc4_conv7(self.dc4_conv6(self.dc4_conv5(x)))
        va = va.view(b, -1, 2, h, w)
        flow4 = (flow4h.view(b, -1, 2, h, w) * F.softmax(va, 1)).sum(1)

        up_flow4 = F.interpolate(flow4, [im.size()[2] // 8, im.size()[3] // 8], mode='bilinear', align_corners=False) * 2
        warp3, _ = self.warp3(c23n, up_flow4)
        feat3 = self.corrf(c13n, warp3, self.md[3])
        feat3 = self.f3(feat3)
        cost3 = self.p3(feat3)
        b, c, u, v, h, w = cost3.shape
        cost3 = cost3.view(-1, u, v, h, w)
        flow3h, ent3h = self.flow_reg8(cost3)
        flow3h = flow3h.view(b, c, 2, h, w) + up_flow4[:, np.newaxis]
        flow3h = flow3h.view(bs, -1, h, w)
        ent3h  = ent3h.view(bs, -1, h, w)
        flow3h = torch.cat((flow3h, F.interpolate(flow4h.detach() * 2, [flow3h.shape[2], flow3h.shape[3]], mode='bilinear', align_corners=False)), 1)
        ent3h  = torch.cat((ent3h,  F.interpolate(ent4h, [flow3h.shape[2], flow3h.shape[3]], mode='bilinear', align_corners=False)), 1)
        x = torch.cat((ent3h.detach(), flow3h.detach(), c13), 1)
        x = self.dc3_conv4(self.dc3_conv3(self.dc3_conv2(self.dc3_conv1(x))))
        va = self.dc3_conv7(self.dc3_conv6(self.dc3_conv5(x)))
        va = va.view(b, -1, 2, h, w)
        flow3 = (flow3h.view(b, -1, 2, h, w) * F.softmax(va, 1)).sum(1)

        up_flow3 = F.interpolate(flow3, [im.size()[2] // 4, im.size()[3] // 4], mode='bilinear', align_corners=False) * 2
        warp2, _ = self.warp2(c22n, up_flow3)
        feat2 = self.corrf(c12n, warp2, self.md[4])
        feat2 = self.f2(feat2)
        cost2 = self.p2(feat2)
        b, c, u, v, h, w = cost2.shape
        cost2 = cost2.view(-1, u, v, h, w)
        flow2h, ent2h = self.flow_reg4(cost2)
        flow2h = flow2h.view(b, c, 2, h, w) + up_flow3[:, np.newaxis]
        flow2h = flow2h.view(bs, -1, h, w)
        ent2h  = ent2h.view(bs, -1, h, w)
        flow2h = torch.cat((flow2h, F.interpolate(flow3h.detach() * 2, [flow2h.shape[2], flow2h.shape[3]], mode='bilinear', align_corners=False)), 1)
        ent2h  = torch.cat((ent2h,  F.interpolate(ent3h, [flow2h.shape[2], flow2h.shape[3]], mode='bilinear', align_corners=False)), 1)
        x = torch.cat((ent2h.detach(), flow2h.detach(), c12), 1)
        x = self.dc2_conv4(self.dc2_conv3(self.dc2_conv2(self.dc2_conv1(x))))
        va = self.dc2_conv7(self.dc2_conv6(self.dc2_conv5(x)))
        va = va.view(b, -1, 2, h, w)
        flow2 = (flow2h.view(b, -1, 2, h, w) * F.softmax(va, 1)).sum(1)

        flow2 = F.interpolate(flow2, [im.size()[2], im.size()[3]], mode='bilinear', align_corners=False)
        return flow2 * 4, ent2h.mean(1)[0]
