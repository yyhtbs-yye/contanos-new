import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

class JerseyOCRMobileNet(nn.Module):
    def __init__(self, 
                 pretrained=True, 
                 dropout=0.1, 
                 num_len=3, 
                 num_digit=11, 
                 use_small=False):
        super().__init__()
        if use_small:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            # Replace classifier with identity; we'll add our own heads.
            self.backbone.classifier = nn.Identity()
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 576
        else:
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
            # Replace classifier with identity; we'll add our own heads.
            self.backbone.classifier = nn.Identity()
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 960

        self.feat_bn = nn.BatchNorm1d(feat_dim)  # mobilenet_v3_small last channels
        self.dropout = nn.Dropout(dropout)
        self.len_head = nn.Linear(feat_dim, num_len)
        self.d1_head  = nn.Linear(feat_dim, num_digit)
        self.d2_head  = nn.Linear(feat_dim, num_digit)

    def forward(self, x):
        # Expect x shape: [B, 3, 256, 192] (H, W) = (256, 192)
        feats = self.backbone.features(x)          # [B, C, h, w]
        pooled = self.pool(feats).flatten(1)       # [B, C]
        pooled = self.feat_bn(pooled)
        pooled = self.dropout(pooled)
        len_logits = self.len_head(pooled)
        d1_logits  = self.d1_head(pooled)
        d2_logits  = self.d2_head(pooled)
        return len_logits, d1_logits, d2_logits

