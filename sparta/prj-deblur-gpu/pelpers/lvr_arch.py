import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW, per-pixel (same semantics as ConvNeXt LN)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps    = eps

    def forward(self, x):
        # mean/var over channel dim only, for each (n,h,w)
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = ChannelLayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma.view(1, -1, 1, 1)
        x = x + shortcut
        return x

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=True)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.pw(self.dw(x)))

class ConvGRUCellDW(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        pad = k // 2
        self.conv_z = nn.Sequential(
            nn.Conv2d(2 * ch, 2 * ch, k, 1, pad, groups=2 * ch, bias=True),
            nn.Conv2d(2 * ch, ch, 1, 1, 0, bias=True)
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(2 * ch, 2 * ch, k, 1, pad, groups=2 * ch, bias=True),
            nn.Conv2d(2 * ch, ch, 1, 1, 0, bias=True)
        )
        self.conv_n = nn.Sequential(
            nn.Conv2d(2 * ch, 2 * ch, k, 1, pad, groups=2 * ch, bias=True),
            nn.Conv2d(2 * ch, ch, 1, 1, 0, bias=True)
        )
    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros_like(x)
        z = torch.sigmoid(self.conv_z(torch.cat([x, h_prev], dim=1)))
        r = torch.sigmoid(self.conv_r(torch.cat([x, h_prev], dim=1)))
        n = torch.tanh(self.conv_n(torch.cat([x, r * h_prev], dim=1)))
        h = (1.0 - z) * n + z * h_prev
        return h

# @ARCH_REGISTRY.register()
class LiteVideoRestorer(nn.Module):
    def __init__(self, 
                 scale=1, 
                 base_channels=16, 
                 depths=(1, 1, 2), 
                 use_convgru=True):
        super().__init__()
        assert scale == 1
        self.scale = scale
        c = base_channels
        self.stem = nn.Sequential(
            DSConv(3, c, k=3, s=2),
            *[ConvNeXtBlock(c) for _ in range(depths[0])]
        )
        self.enc1 = nn.Sequential(
            DSConv(c, c, k=3, s=1),
            DSConv(c, 2 * c, k=3, s=2),
            *[ConvNeXtBlock(2 * c) for _ in range(depths[1])]
        )
        self.use_convgru = use_convgru
        ch_h4 = 2 * c
        if use_convgru:
            self.gru = ConvGRUCellDW(ch_h4, k=3)
        else:
            self.temporal_stack = nn.Sequential(DSConv(ch_h4, ch_h4, 3, 1), ConvNeXtBlock(ch_h4))
        self.bottleneck = nn.Sequential(*[ConvNeXtBlock(ch_h4) for _ in range(depths[2])])
        self.up1 = nn.Sequential(
            nn.Conv2d(ch_h4, 4 * c, 1, 1, 0),
            nn.PixelShuffle(2),
            ConvNeXtBlock(c)
        )
        self.fuse1 = nn.Sequential(
            DSConv(2 * c, c, k=3, s=1),
            ConvNeXtBlock(c)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c, 4 * c, 1, 1, 0),
            nn.PixelShuffle(2),
            DSConv(c, c, k=3, s=1)
        )
        self.head = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1)

        # --- persistent temporal state (used when use_convgru=True) ---
        self._h_state = None  # shape: (N, 2*c, H/4, W/4)

    def reset_state(self):
        """Call at sequence boundaries to clear temporal memory."""
        self._h_state = None

    def forward(self, lqs):
        n, t, _, h, w = lqs.shape
        outs = []
        h_state = None
        for i in range(t):
            x = lqs[:, i]
            x_s = self.stem(x)
            x_e = self.enc1(x_s)
            if self.use_convgru:
                x_t = self.gru(x_e, h_state)
                h_state = x_t
            else:
                x_t = self.temporal_stack(x_e)
            x_b = self.bottleneck(x_t)
            x = self.up1(x_b)
            x = torch.cat([x, x_s], dim=1)
            x = self.fuse1(x)
            x = self.up2(x)
            x = self.head(x)
            x = x + lqs[:, i]
            outs.append(x)
        out = torch.stack(outs, dim=1)
        return out

    def stream_process(self, lq):
        """
        Process a single frame while maintaining internal temporal state.

        Args:
            lq: (N, 3, H, W) low-quality frame(s)

        Returns:
            (N, 3, H, W) restored frame(s)
        """
        # encoder
        x_s = self.stem(lq)   # (N, c,   H/2, W/2)
        x_e = self.enc1(x_s)  # (N, 2c,  H/4, W/4)

        # temporal fusion (GRU keeps state between calls)
        if self.use_convgru:
            # if spatial size / dtype / device changed, drop state
            if (
                self._h_state is not None and
                (self._h_state.shape != x_e.shape or
                 self._h_state.dtype  != x_e.dtype or
                 self._h_state.device != x_e.device)
            ):
                self._h_state = None
            self._h_state = self.gru(x_e, self._h_state)
            x_t = self._h_state
        else:
            x_t = self.temporal_stack(x_e)

        # decoder + skip
        x_b = self.bottleneck(x_t)
        x   = self.up1(x_b)
        x   = torch.cat([x, x_s], dim=1)
        x   = self.fuse1(x)
        x   = self.up2(x)
        x   = self.head(x)
        return x + lq

if __name__ == "__main__":
    model = LiteVideoRestorer()
    model.eval()
    from tqdm import tqdm
    from time import time

    time_start = time()
    inp = torch.randn(1, 750, 3, 540, 960).cuda()
    model = model.cuda()

    with torch.no_grad():
        
        for i in tqdm(range(inp.shape[1])):
            out = model.stream_process(inp[:, i, ...])

        print("in:", inp.shape, "out:", out.shape)

    print("Time taken:", time() - time_start)
