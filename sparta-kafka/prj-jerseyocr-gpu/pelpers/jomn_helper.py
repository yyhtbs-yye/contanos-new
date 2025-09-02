import torch
import torch.nn as nn
from torchvision.ops import roi_align

import numpy as np
import os
from pelpers.jomn_arch import JerseyOCRMobileNet

class JOMNHelper:
    def __init__(self, 
                 use_small: bool,
                 weights_path: str,
                 model_input_size: tuple = (256, 192),
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225),
                 device: str = 'cuda'):
            self.model_input_size = model_input_size
            self.mean = torch.tensor(mean, dtype=torch.float32).to(device).view(1, 3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).to(device).view(1, 3, 1, 1)
            self.device = device

            model_size = 'small' if use_small else 'large'

            path_wo_ext, ext = os.path.splitext(weights_path)

            weights_path = f"{path_wo_ext}_{model_size}{ext}"

            if not os.path.exists(f"{model_size}_{weights_path}"):
                print(f"Downloading weights to {weights_path}...")
                dirpath = os.path.dirname(weights_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                if use_small:
                    os.system(f"gdown 1eFB5Wvjnnb2s2AmgXj8mFAZuK0kcEWZt -O {weights_path}")
                else:
                    os.system(f"gdown 16yQDv-n1ApjQUjEQrUV013lZ5Ejkhygj -O {weights_path}")

            self.model = self._load_model(use_small, weights_path)

    def _load_model(self, use_small: bool, weights_path: str):
        model = JerseyOCRMobileNet(pretrained=True, use_small=use_small)
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(self.device)
        return model

    @torch.no_grad()
    def __call__(self, image: np.ndarray, bboxes: list = []):

        # default bbox = full image
        if len(bboxes) == 0:
            h, w = image.shape[:2]
            bboxes = [[0, 0, w, h]]

        # to GPU, CHW, float in [0,1]
        img = torch.from_numpy(image).to(self.device, non_blocking=True)
        if img.ndim == 2:
            img = img.unsqueeze(-1)              # H,W,1
        img = img.permute(2, 0, 1).contiguous()  # C,H,W
        # ensure RGB 3-channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]  # drop alpha
        img = img.float() / 255.0                # [3,H,W]
        img = img.unsqueeze(0)                   # [1,3,H,W]

        H, W = img.shape[2:]
        # clamp boxes to image bounds on GPU
        boxes_xyxy = torch.tensor(bboxes, dtype=torch.float32, device=self.device)
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, W)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, H)

        # roi_align expects (num_boxes, 5): [batch_idx, x1, y1, x2, y2]
        batch_idx = torch.zeros((boxes_xyxy.size(0), 1), dtype=torch.float32, device=self.device)
        boxes = torch.cat([batch_idx, boxes_xyxy], dim=1)

        # GPU crop + resize to fixed (H,W)
        pooled_h, pooled_w = self.model_input_size
        crops = roi_align(
            img, boxes, output_size=(pooled_h, pooled_w),
            spatial_scale=1.0, sampling_ratio=-1, aligned=True
        )  # shape: [B, 3, H, W]
        
        # normalize the images
        crops = (crops - self.mean) / self.std
        len_logits, d1_logits, d2_logits = self.model(crops)

        # probabilities + predictions
        len_probs = torch.softmax(len_logits, dim=1)
        d1_probs  = torch.softmax(d1_logits,  dim=1)
        d2_probs  = torch.softmax(d2_logits,  dim=1)

        len_pred = len_probs.argmax(dim=1)           # [B]
        d1_pred  = d1_probs.argmax(dim=1)
        d2_pred  = d2_probs.argmax(dim=1)

        conf_len = len_probs.gather(1, len_pred.unsqueeze(1)).squeeze(1)
        conf_d1  = d1_probs.gather(1,  d1_pred.unsqueeze(1)).squeeze(1)
        conf_d2  = d2_probs.gather(1,  d2_pred.unsqueeze(1)).squeeze(1)

        B = len_pred.size(0)
        dev = len_pred.device
        minus_one = torch.full((B,), -1, dtype=torch.int64, device=dev)

        # potential number (ignores predicted length)
        both_blank = (d1_pred == 10) & (d2_pred == 10)
        one_blank  = (d1_pred == 10) ^ (d2_pred == 10)
        take_other = torch.where(d1_pred == 10, d2_pred, d1_pred)
        pot = torch.where(
            both_blank, minus_one,
            torch.where(one_blank, take_other, d1_pred * 10 + d2_pred)
        )

        # final number based on predicted length
        num = minus_one.clone()
        single = torch.where(d1_pred == 10, minus_one, d1_pred)
        num = torch.where(len_pred == 1, single, num)
        two = torch.where(
            both_blank, minus_one,
            torch.where(one_blank, take_other, d1_pred * 10 + d2_pred)
        )
        num = torch.where(len_pred == 2, two, num)

        # confidence
        conf = torch.where(
            len_pred == 0, conf_len,
            torch.where(len_pred == 1, (conf_len * conf_d1).sqrt(),
                        (conf_len * conf_d1 * conf_d2).pow(1/3))
        )

        return num.tolist(), pot.tolist(), conf.detach().cpu().tolist()

