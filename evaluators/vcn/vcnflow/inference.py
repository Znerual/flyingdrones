"""VCN inference wrapper.

Handles:
- checkpoint loading (reads mean_L / mean_R from .tar)
- padding to multiples of 64
- recreation of flow_reg / WarpModule for each new spatial size
- flow rescaling back to original resolution
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _pad64(h, w):
    """Return (padded_h, padded_w) that are multiples of 64."""
    ph = int(math.ceil(h / 64) * 64)
    pw = int(math.ceil(w / 64) * 64)
    return ph, pw


def _rebuild_spatial_modules(model_module, max_w, max_h):
    """Recreate size-dependent modules for a new (max_w, max_h) resolution."""
    from vcnflow.models.VCN import flow_reg as flow_reg_full, WarpModule as WarpModule_full
    from vcnflow.models.VCN_small import flow_reg as flow_reg_small, WarpModule as WarpModule_small

    # Detect whether this is VCN or VCN_small by presence of flow_reg4
    is_full = hasattr(model_module, 'flow_reg4')
    flow_reg_cls = flow_reg_full if is_full else flow_reg_small
    warp_cls = WarpModule_full if is_full else WarpModule_small

    dev = next(model_module.parameters()).device

    if hasattr(model_module, 'flow_reg64'):
        fr = model_module.flow_reg64
        model_module.flow_reg64 = flow_reg_cls([1, max_w // 64, max_h // 64],
                                               ent=fr.ent, maxdisp=fr.md, fac=fr.fac).to(dev)
    if hasattr(model_module, 'flow_reg32'):
        fr = model_module.flow_reg32
        model_module.flow_reg32 = flow_reg_cls([1, max_w // 32, max_h // 32],
                                               ent=fr.ent, maxdisp=fr.md, fac=fr.fac).to(dev)
    if hasattr(model_module, 'flow_reg16'):
        fr = model_module.flow_reg16
        model_module.flow_reg16 = flow_reg_cls([1, max_w // 16, max_h // 16],
                                               ent=fr.ent, maxdisp=fr.md, fac=fr.fac).to(dev)
    if hasattr(model_module, 'flow_reg8'):
        fr = model_module.flow_reg8
        model_module.flow_reg8  = flow_reg_cls([1, max_w // 8,  max_h // 8],
                                               ent=fr.ent, maxdisp=fr.md, fac=fr.fac).to(dev)
    if hasattr(model_module, 'flow_reg4'):
        fr = model_module.flow_reg4
        model_module.flow_reg4  = flow_reg_cls([1, max_w // 4,  max_h // 4],
                                               ent=fr.ent, maxdisp=fr.md, fac=fr.fac).to(dev)

    model_module.warp5 = warp_cls([1, max_w // 32, max_h // 32]).to(dev)
    model_module.warp4 = warp_cls([1, max_w // 16, max_h // 16]).to(dev)
    model_module.warp3 = warp_cls([1, max_w // 8,  max_h // 8]).to(dev)
    if hasattr(model_module, 'warp2'):
        model_module.warp2 = warp_cls([1, max_w // 4, max_h // 4]).to(dev)


class VCNInference:
    """Wraps a VCN model for single-pair inference at any resolution."""

    def __init__(self, model: nn.DataParallel, mean_L, mean_R):
        self.model = model
        self.mean_L = np.asarray(mean_L).mean(0)  # (3,)
        self.mean_R = np.asarray(mean_R).mean(0)  # (3,)
        self._last_size = None

    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        img1, img2: (1, 3, H, W) float32 tensors on GPU, range [0, 255], RGB.
        Returns: (1, 2, H, W) float32 flow in pixels, on GPU.
        """
        B, C, H, W = img1.shape
        assert B == 1

        # Pad to multiple of 64
        max_h, max_w = _pad64(H, W)

        # Rebuild spatial modules if size changed
        if (max_h, max_w) != self._last_size:
            module = self.model.module if hasattr(self.model, 'module') else self.model
            _rebuild_spatial_modules(module, max_w, max_h)
            self._last_size = (max_h, max_w)

        # Preprocess: to float [0,1], BGR, mean subtract, pad
        def prep(img, mean):
            # img: (1, 3, H, W) float32 [0,255] RGB
            x = img[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, 3) RGB
            x = x[:, :, ::-1].copy() / 255.0  # BGR
            x -= mean[np.newaxis, np.newaxis, :]
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            # Pad to (max_h, max_w)
            if H != max_h or W != max_w:
                x = F.interpolate(x, size=(max_h, max_w), mode='bilinear', align_corners=False)
            return x.to(img.device)

        imgL = prep(img1, self.mean_L)
        imgR = prep(img2, self.mean_R)

        imgLR = torch.cat([imgL, imgR], 0)  # (2, 3, max_h, max_w)
        self.model.eval()
        pred_disp, _ = self.model(imgLR)  # (1, 2, max_h, max_w)

        # Crop / resize back to (H, W) and scale flow
        if max_h != H or max_w != W:
            pred_disp = F.interpolate(pred_disp, size=(H, W), mode='bilinear', align_corners=False)
            pred_disp[:, 0] *= W / max_w
            pred_disp[:, 1] *= H / max_h

        return pred_disp  # (1, 2, H, W)


def load_vcn(ckpt_path: str, model_type: str = 'VCN', device: str = 'cuda') -> VCNInference:
    """Load checkpoint and return a VCNInference wrapper.

    model_type: 'VCN' or 'VCN_small'
    device: 'cuda' or 'cpu'
    """
    if model_type == 'VCN':
        from vcnflow.models.VCN import VCN
    else:
        from vcnflow.models.VCN_small import VCN

    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mean_L = checkpoint.get('mean_L', [[0.33, 0.33, 0.33]])
    mean_R = checkpoint.get('mean_R', [[0.33, 0.33, 0.33]])

    # Determine maxdisp / fac from checkpoint path heuristic (KITTI → 512/2, Sintel → 448/1.4, else 256/1)
    base = os.path.basename(os.path.dirname(ckpt_path)).lower()
    if 'kitti' in base:
        maxdisp, fac = 512, 2.0
    elif 'sintel' in base:
        maxdisp, fac = 448, 1.4
    else:
        maxdisp, fac = 256, 1.0

    # Use a dummy size; flow_reg will be rebuilt before first call
    model = VCN([1, 256, 256], md=[int(4 * (maxdisp / 256)), 4, 4, 4, 4], fac=fac)
    if device == 'cuda':
        model = nn.DataParallel(model)
        model.cuda()
    else:
        model.to(device)

    state = checkpoint['state_dict']
    state = {k: v for k, v in state.items() if 'grid' not in k and ('flow_reg' not in k or 'conv1' in k)}
    model.load_state_dict(state, strict=False)
    model.eval()

    return VCNInference(model, mean_L, mean_R)
