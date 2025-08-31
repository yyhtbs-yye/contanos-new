import cv2
import time
import argparse
from math import sqrt

def measure_rtsp_fps(
    rtsp_url: str,
    duration: float | None = None,
    max_frames: int | None = None,
    print_every: int = 30,
    warmup_frames: int = 0
):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return 1

    # Try to minimize internal buffering (may be ignored on some backends)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    if reported_fps and reported_fps > 0:
        print(f"Stream-reported FPS: {reported_fps:.2f}")
    else:
        print("Stream-reported FPS: (not available)")

    frame_count = 0
    drop_count = 0
    start_time = None
    last_time = None
    interarrival = []  # seconds between received frames (after warmup)

    try:
        while True:
            ret, frame = cap.read()
            now = time.perf_counter()

            if not ret:
                # If a frame can't be read, consider it a drop and keep trying.
                drop_count += 1
                time.sleep(0.001)
                continue

            if start_time is None:
                start_time = now
                last_time = now
                frame_count = 1
                continue

            frame_count += 1

            # Warmup: let buffers stabilize before we start measuring
            if frame_count <= warmup_frames:
                last_time = now
                continue

            dt = now - last_time
            last_time = now
            if dt > 0:
                interarrival.append(dt)

            # Periodic live stats
            if print_every and (frame_count - warmup_frames) % print_every == 0 and interarrival:
                avg_fps = len(interarrival) / sum(interarrival)
                inst_fps = 1.0 / interarrival[-1] if interarrival[-1] > 0 else float("inf")
                elapsed = now - start_time
                print(f"{frame_count - warmup_frames:6d} frames | avg={avg_fps:6.2f} FPS | inst={inst_fps:6.2f} FPS | "
                      f"elapsed={elapsed:6.1f}s | drops={drop_count}")

            # Stop conditions
            if duration is not None and start_time is not None and (now - start_time) >= duration:
                break
            if max_frames is not None and (frame_count - warmup_frames) >= max_frames:
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cap.release()

    # Final summary
    n = len(interarrival)
    if n == 0:
        print("No frames measured (increase duration or reduce warmup).")
        return 0

    total_time = sum(interarrival)
    avg_fps = n / total_time
    # jitter: std dev of inter-arrival times in ms
    mean_dt = total_time / n
    if n > 1:
        var_dt = sum((dt - mean_dt) ** 2 for dt in interarrival) / (n - 1)
        jitter_ms = sqrt(var_dt) * 1000.0
    else:
        jitter_ms = 0.0

    sorted_dt = sorted(interarrival)
    def pct(sorted_list, p):
        # p in [0,100]
        if not sorted_list:
            return float("nan")
        idx = int(round((p / 100.0) * (len(sorted_list) - 1)))
        return sorted_list[idx]

    p50_dt = pct(sorted_dt, 50)
    p95_dt = pct(sorted_dt, 95)

    print("\n==== RTSP Receive FPS Summary ====")
    print(f"Frames measured:       {n}")
    print(f"Dropped reads:         {drop_count}")
    print(f"Average FPS:           {avg_fps:.2f}")
    print(f"Median FPS (p50):      {1.0/p50_dt:.2f} (dt {p50_dt*1000:.2f} ms)")
    print(f"p95 FPS:               {1.0/p95_dt:.2f} (dt {p95_dt*1000:.2f} ms)")
    print(f"Jitter (std of dt):    {jitter_ms:.2f} ms")
    print("==================================")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure RTSP receive FPS (frame arrival rate).")
    parser.add_argument("--url", default="rtsp://192.168.200.206:8554/xstream?tcp", help="RTSP URL")
    parser.add_argument("--duration", type=float, default=10.0, help="How long to measure (seconds).")
    parser.add_argument("--max-frames", type=int, default=1000, help="Stop after this many measured frames.")
    parser.add_argument("--print-every", type=int, default=30, help="Print live stats every N frames.")
    parser.add_argument("--warmup-frames", type=int, default=0, help="Ignore first N frames for measurement.")
    args = parser.parse_args()

    measure_rtsp_fps(
        rtsp_url=args.url,
        duration=args.duration,
        max_frames=args.max_frames,
        print_every=args.print_every,
        warmup_frames=args.warmup_frames
    )
