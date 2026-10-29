import numpy as np
import torch
import torch.utils.data as data
import cv2
import os
import os.path as osp
from glob import glob


class SyntheticDrones(data.Dataset):
    """
    Synthetic drone dataset loader.

    Dataset layout:
        <root>/
          Nature1/
            images/  0000.png, 0001.png, ...
            flow/    0000_flow.npy, ...
          Nature2/
            ...

    Flow arrays: shape (H, W, 2), float32, channels are (u, v) in pixels.
    """

    def __init__(self, root='SyntheticDrones', datasets_path='', resize_shape=None):
        self.resize_shape = resize_shape
        root_path = osp.join(datasets_path, root) if datasets_path else root

        self.image_list = []
        self.flow_list = []

        try:
            scene_dirs = sorted([
                d for d in os.listdir(root_path)
                if osp.isdir(osp.join(root_path, d))
            ])
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset root not found: {root_path}")

        if not scene_dirs:
            raise FileNotFoundError(f"No scene directories in {root_path}")

        print(f"Found {len(scene_dirs)} scenes in {root_path}")

        for scene in scene_dirs:
            scene_path = osp.join(root_path, scene)
            image_root = osp.join(scene_path, 'images')
            flow_root = osp.join(scene_path, 'flow')

            if not all(osp.isdir(p) for p in [image_root, flow_root]):
                print(f"  Skipping '{scene}': missing images/ or flow/")
                continue

            images_in_scene = sorted(glob(osp.join(image_root, '*.png')))
            if not images_in_scene:
                continue

            for i in range(len(images_in_scene) - 1):
                img1_path = images_in_scene[i]
                img2_path = images_in_scene[i + 1]
                base_name = osp.splitext(osp.basename(img1_path))[0]
                flow_file = osp.join(flow_root, f"{base_name}_flow.npy")

                if osp.exists(img2_path) and osp.exists(flow_file):
                    self.image_list.append([img1_path, img2_path])
                    self.flow_list.append(flow_file)

        print(f"Total valid image pairs: {len(self.image_list)}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img1 = cv2.imread(self.image_list[index][0])
        img2 = cv2.imread(self.image_list[index][1])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        flow = np.load(self.flow_list[index]).astype(np.float32)

        if self.resize_shape is not None:
            target_h, target_w = self.resize_shape
            original_h, original_w = img1.shape[:2]
            if original_h != target_h or original_w != target_w:
                scale_h = target_h / original_h
                scale_w = target_w / original_w
                img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
                flow = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                flow[..., 0] *= scale_w
                flow[..., 1] *= scale_h

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow[torch.isnan(flow)] = 0
        flow[flow.abs() > 1e9] = 0
        valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()
