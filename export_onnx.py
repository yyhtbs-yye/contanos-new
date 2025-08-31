import torch
import torch.nn as nn
from test_deblur_speed import LiteVideoRestorer
class _ONNXStreamWrapper(nn.Module):
    """
    Wraps LiteVideoRestorer as a pure function:
      (lq, h_prev) -> (y, h_next)
    where lq:  (N,3,H,W)
          h_prev/h_next: (N, 2*c, H/4, W/4) with c = base_channels
    """
    def __init__(self, model: LiteVideoRestorer):
        super().__init__()
        self.model = model
        assert model.use_convgru, "ONNX stream wrapper expects use_convgru=True"

    def forward(self, lq, h_prev):
        # encoder
        x_s = self.model.stem(lq)      # (N, c,   H/2, W/2)
        x_e = self.model.enc1(x_s)     # (N, 2c,  H/4, W/4)

        # temporal (use provided h_prev instead of internal memory)
        x_t = self.model.gru(x_e, h_prev)  # returns h_next

        # decoder + skip
        x_b = self.model.bottleneck(x_t)
        x   = self.model.up1(x_b)
        x   = torch.cat([x, x_s], dim=1)
        x   = self.model.fuse1(x)
        x   = self.model.up2(x)
        x   = self.model.head(x)
        y   = x + lq
        return y, x_t  # (restored frame, next hidden state)


def export_onnx_stream_model(model: LiteVideoRestorer,
                             H: int,
                             W: int,
                             onnx_path: str = "lite_video_restorer_fp32.onnx",
                             fp16_path: str | None = "lite_video_restorer_fp16.onnx",
                             opset: int = 17,
                             device: str = "cuda"):
    """
    Exports a stateless streaming ONNX:
      inputs:  lq:(N,3,H,W), h_prev:(N,2*c,H/4,W/4)
      outputs: y:(N,3,H,W),  h_next:(N,2*c,H/4,W/4)

    If fp16_path is provided, also writes a converted FP16 ONNX.
    """
    model = model.eval().to(device)
    c = model.stem[0].pw.out_channels // 1  # base_channels
    hidden_ch = 2 * c
    assert H % 4 == 0 and W % 4 == 0, "H and W must be divisible by 4."

    # Wrap to expose state explicitly
    wrapped = _ONNXStreamWrapper(model).to(device).eval()

    # Dummy inputs
    x = torch.randn(1, 3, H, W, device=device)
    h = torch.zeros(1, hidden_ch, H // 4, W // 4, device=device)

    # ---- Export FP32 ONNX ----
    torch.onnx.export(
        wrapped,
        (x, h),
        onnx_path,
        input_names=["lq", "h_prev"],
        output_names=["y", "h_next"],
        dynamic_axes={
            "lq":     {0: "N", 2: "H", 3: "W"},
            "h_prev": {0: "N", 2: "H4", 3: "W4"},
            "y":      {0: "N", 2: "H", 3: "W"},
            "h_next": {0: "N", 2: "H4", 3: "W4"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[ONNX] FP32 model saved to: {onnx_path}")

    # ---- Optional: FP16 convert for acceleration ----
    if fp16_path is not None:
        try:
            import onnx
            from onnxconverter_common import float16
        except Exception as e:
            print("[ONNX] Skipping FP16 conversion (install: `pip install onnx onnxconverter-common`).")
            return

        model_onnx = onnx.load(onnx_path)
        # Convert to float16; keep I/O in float16 as well for max throughput.
        model_fp16 = float16.convert_float_to_float16(
            model_onnx,
            keep_io_types=False,  # set True if your runtime feeds FP32 inputs
        )
        onnx.save(model_fp16, fp16_path)
        print(f"[ONNX] FP16 model saved to: {fp16_path}")


if __name__ == "__main__":
    from time import time
    from tqdm import tqdm

    model = LiteVideoRestorer()
    model.eval()
    # ---- Export ONNX (stateless streaming) ----
    export_onnx_stream_model(
        model,
        H=540,
        W=960,
        onnx_path="lite_video_restorer_fp32.onnx",
        fp16_path="lite_video_restorer_fp16.onnx",  # set None to skip FP16
        opset=17,
        device="cuda",
    )
