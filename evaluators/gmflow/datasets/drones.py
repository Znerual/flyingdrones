import numpy as np
import os
import os.path as osp
import sys
from glob import glob

import cv2
import torch
import torch.nn.functional as F
import torch.utils.data as data

# Shared Nature3 helpers live next to evaluate_drones.py (the repo root). Put that
# directory first so a stale copy elsewhere on sys.path cannot shadow it.
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from fd_eval_common import list_samples, TWO_FRAME  # noqa: E402


class DroneFlowDataset(data.Dataset):
    """
    SyntheticDrones loader that returns drone label masks alongside flow.

    Returns (img_t, img_{t-1}, flow_t, mask_t); flow_t is the GT flow t -> t-1.

    Dataset layout:
        <root>/
          Scene1/
            images/   0000.png, 0001.png, ...
            flow/     0000_flow.npy, ...
            labels/   0000_label.png, ...   (optional)
          Scene2/
            ...

    Flow arrays: shape (H, W, 2), float32, channels (u, v) in pixels.
    Label images: grayscale PNG, nonzero = drone pixel.
    """

    def __init__(self, root_path, resize_shape=None):
        self.resize_shape = resize_shape

        # Pairing (fixed 2026-10-07): ``flow/<t>_flow.npy`` is the BACKWARD flow
        # t -> t-1, so each sample is (I_t, I_{t-1}, <t>_flow, <t>_label) with I_t as
        # the reference frame; frame 0000 is never a reference. Built by the shared
        # fd_eval_common.list_samples (flat scene dir or parent of scene dirs).
        # Labels stay optional per scene: if a scene has a labels/ dir, samples
        # without a label file are dropped; otherwise the mask is all-zero.
        samples = list_samples(root_path, TWO_FRAME, require_labels=False)
        self.samples = [s for s in samples
                        if s[-1] is not None
                        or not osp.isdir(osp.join(osp.dirname(osp.dirname(s[-2])), 'labels'))]

        print(f'DroneFlowDataset: {len(self.samples)} pairs (I_t, I_t-1) from {root_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, pf, pl = self.samples[idx]  # p1 = I_t (reference), p2 = I_{t-1}

        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB).astype(np.float32)
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB).astype(np.float32)
        flow = np.load(pf).astype(np.float32)

        if pl is not None:
            mask = (cv2.imread(pl, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        else:
            mask = np.zeros(img1.shape[:2], dtype=np.float32)

        if self.resize_shape is not None:
            H, W = self.resize_shape
            oh, ow = img1.shape[:2]
            img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_AREA)
            flow_t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            flow_t = F.interpolate(flow_t, size=(H, W), mode='bilinear', align_corners=False)
            flow_t[:, 0] *= W / ow
            flow_t[:, 1] *= H / oh
            flow = flow_t.squeeze(0).permute(1, 2, 0).numpy()
            mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask_t, size=(H, W), mode='nearest').squeeze().numpy()

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()

        return img1, img2, flow, mask
