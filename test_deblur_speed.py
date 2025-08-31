import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

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

class LiteVideoRestorer(nn.Module):
    def __init__(self, 
                 scale=1, 
                 base_channels=16, 
                 depths=(1, 2, 2), 
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

# if __name__ == "__main__":
#     model = LiteVideoRestorer()
#     model.eval()
#     from tqdm import tqdm
#     from time import time

#     time_start = time()
#     inp = torch.randn(1, 750, 3, 720, 1280).cuda()
#     model = model.cuda()

#     with torch.no_grad():
        
#         for i in tqdm(range(inp.shape[1])):
#             out = model.stream_process(inp[:, i, ...])

#         print("in:", inp.shape, "out:", out.shape)

#     print("Time taken:", time() - time_start)

def profile_stream_model(model: nn.Module,
                         H: int = 720,
                         W: int = 1280,
                         device: str = "cuda",
                         n_frames: int = 750,
                         use_stream_process: bool = True):
    """
    Profiles per-frame MACs/FLOPs and params for a streaming video model.
    Returns (per_frame_macs, per_frame_flops, total_macs, total_flops, n_params).
    """
    model = model.to(device).eval()

    # Small wrapper so fvcore/thop can call the same entrypoint your loop uses
    wrapped = model
    if use_stream_process and hasattr(model, "stream_process"):
        class _StreamWrapper(nn.Module):
            def __init__(self, m): 
                super().__init__(); self.m = m
            def forward(self, x):
                return self.m.stream_process(x)
        wrapped = _StreamWrapper(model).to(device).eval()

    # Dummy single-frame input (B, C, H, W)
    x = torch.randn(1, 3, H, W, device=device)

    # Count params
    n_params = sum(p.numel() for p in model.parameters())

    # Try fvcore first (robust and detailed)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops_an = FlopCountAnalysis(wrapped, x)
        per_frame_macs = int(flops_an.total())        # fvcore commonly reports MACs
        per_frame_flops = per_frame_macs * 2          # FLOPs ≈ 2 × MACs convention
    except Exception:
        # Fallback to thop
        try:
            from thop import profile
            per_frame_macs, _ = profile(wrapped, inputs=(x,), verbose=False)
            per_frame_macs = int(per_frame_macs)
            per_frame_flops = per_frame_macs * 2
        except Exception as e:
            raise RuntimeError(
                "Could not profile FLOPs/MACs. Please `pip install fvcore` or `pip install thop`."
            ) from e

    total_macs = per_frame_macs * n_frames
    total_flops = per_frame_flops * n_frames
    return per_frame_macs, per_frame_flops, total_macs, total_flops, n_params


# ---- Your original main, with profiling added --------------------------

if __name__ == "__main__":
    from time import time
    from tqdm import tqdm

    model = LiteVideoRestorer()
    model.eval()

    # ---- FLOPs/MACs & params (single frame, plus total for 750 frames) ----
    per_macs, per_flops, tot_macs, tot_flops, n_params = profile_stream_model(
        model,
        H=540,
        W=960,
        device="cuda",
        n_frames=750,
        use_stream_process=True,  # set False if your forward() handles streaming
    )

    def _fmt(n):
        # Pretty print in human-friendly units
        for unit in ["", "K", "M", "G", "T", "P"]:
            if abs(n) < 1000:
                return f"{n:.3f}{unit}"
            n /= 1000.0
        return f"{n:.3f}E"

    print("\n=== Profile (per frame) ===")
    print(f"Params:            {_fmt(n_params)}")
    print(f"MACs (per frame):  {_fmt(per_macs)}")
    print(f"FLOPs (per frame): {_fmt(per_flops)}  (assuming 1 MAC = 2 FLOPs)")

    print("\n=== Estimated totals for 750 frames ===")
    print(f"Total MACs:        {_fmt(tot_macs)}")
    print(f"Total FLOPs:       {_fmt(tot_flops)}\n")

    # ---- Your timing loop (unchanged) ----
    time_start = time()
    inp = torch.randn(1, 3, 540, 960)
    model = model.eval().cuda()

    with torch.no_grad():
        for i in tqdm(range(750)):
            out = model.stream_process(inp.to('cuda'))

        print("in:", inp.shape, "out:", out.shape)

    print("Time taken:", time() - time_start)