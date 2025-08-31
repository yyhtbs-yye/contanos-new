# compare_fp32_fp16_ort_trt.py
import time, argparse
import numpy as np
import cupy as cp
import onnx, onnxruntime as ort
from onnx import TensorProto

def elem_type_to_numpy(t):
    return {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}.get(t, np.float32)

def detect_first_input_dtype(path, default=np.float32):
    try:
        m = onnx.load(path)
        return elem_type_to_numpy(m.graph.input[0].type.tensor_type.elem_type)
    except Exception:
        return default

def make_sess(path, ep, device_id=0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    av = ort.get_available_providers()
    prov = []
    if ep == "trt":
        if "TensorrtExecutionProvider" in av:
            prov.append(("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                # increase cache dir if you like: "trt_engine_cache_path": "./trt_cache"
            }))
        else:
            print("⚠️ TRT EP not available; falling back to CUDA EP.")
            ep = "cuda"
    if ep == "cuda":
        if "CUDAExecutionProvider" in av:
            prov.append(("CUDAExecutionProvider", {
                "device_id": device_id,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "enable_cuda_graph": True,
            }))
        else:
            print("⚠️ CUDA EP not available; using CPU.")
            prov = ["CPUExecutionProvider"]
    if ep == "cpu":
        prov = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=prov), ep

def run_stream(sess, H, W, T, warmup, batch, base_channels, io_dtype, device_id=0):
    # I/O names from your export wrapper
    in_lq, in_hprev = "lq", "h_prev"
    out_y, out_hnext = "y", "h_next"
    hidden_ch = 2 * base_channels

    # Allocate on GPU with CuPy (no host copies)
    lq = cp.random.standard_normal((batch, 3, H, W), dtype=cp.float32).astype(io_dtype)
    h  = cp.zeros((batch, hidden_ch, H // 4, W // 4), dtype=io_dtype)
    y  = cp.empty((batch, 3, H, W), dtype=io_dtype)
    hn = cp.empty((batch, hidden_ch, H // 4, W // 4), dtype=io_dtype)

    io = sess.io_binding()
    def bind():
        io.bind_input("lq", "cuda", device_id, io_dtype, list(lq.shape), int(lq.data.ptr))
        io.bind_input("h_prev", "cuda", device_id, io_dtype, list(h.shape), int(h.data.ptr))
        io.bind_output("y", "cuda", device_id, io_dtype, list(y.shape), int(y.data.ptr))
        io.bind_output("h_next", "cuda", device_id, io_dtype, list(hn.shape), int(hn.data.ptr))

    # Warmup
    for _ in range(max(1, warmup)):
        bind()
        sess.run_with_iobinding(io)
        h, hn = hn, h

    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    frames = 0
    for _ in range(T):
        bind()
        sess.run_with_iobinding(io)
        h, hn = hn, h
        frames += 1
    cp.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    return (elapsed / frames) * 1000.0, frames / elapsed, sess.get_providers()

def bench_pair(fp32_path, fp16_path, H, W, T, warmup, batch, base_ch, ep, device_id, fp16_io_override):
    sess32, ep_used32 = make_sess(fp32_path, ep, device_id)
    sess16, ep_used16 = make_sess(fp16_path, ep, device_id)

    io32 = detect_first_input_dtype(fp32_path, np.float32)
    if fp16_io_override == "auto":
        io16 = detect_first_input_dtype(fp16_path, np.float16)
    elif fp16_io_override == "fp32":
        io16 = np.float32
    else:
        io16 = np.float16

    ms32, fps32, prov32 = run_stream(sess32, H, W, T, warmup, batch, base_ch, io32, device_id)
    ms16, fps16, prov16 = run_stream(sess16, H, W, T, warmup, batch, base_ch, io16, device_id)

    print(f"\n=== {ep_used16.upper()} EP Results ===")
    print(f"Providers FP32: {prov32}")
    print(f"Providers FP16: {prov16}")
    print(f"FP32: {ms32:.3f} ms/frame  |  {fps32:.2f} FPS  (dtype={io32})")
    print(f"FP16: {ms16:.3f} ms/frame  |  {fps16:.2f} FPS  (dtype={io16})")
    print(f"Speedup FP16 vs FP32: {ms32/ms16:.2f}×")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="lite_video_restorer_fp32.onnx")
    ap.add_argument("--fp16", default="lite_video_restorer_fp16.onnx")
    ap.add_argument("--H", type=int, default=540)
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--T", type=int, default=750)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--ep", choices=["cuda","trt","cpu"], default="trt",
                    help="Execution Provider preference")
    ap.add_argument("--fp16-io", choices=["auto","fp16","fp32"], default="auto",
                    help="Use fp32 if FP16 model was converted with keep_io_types=True")
    args = ap.parse_args()

    bench_pair(args.fp32, args.fp16, args.H, args.W, args.T, args.warmup,
               args.batch, args.base_channels, args.ep, args.device_id, args.fp16_io)

if __name__ == "__main__":
    main()
