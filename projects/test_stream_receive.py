#!/usr/bin/env python3
"""
Capture frames from an RTSP stream using TCP transport and save every Nth frame as PNG.
- Forces RTP-over-TCP (interleaved) for RTSP.
- Adds open/read timeouts and simple auto-reconnect.
- Lets you pick stream, save interval, and output dir via CLI.
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Optional

import cv2

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RTSP (TCP) frame grabber")
    p.add_argument(
        "--streams",
        nargs="+",
        default=[
            "rtsp://localhost:8554/annotated_stream",
            "rtsp://localhost:8554/mystream",
            "rtsp://localhost:8554/rawstream",
        ],
        help="One or more RTSP URLs to try in order.",
    )
    p.add_argument(
        "-n", "--nth", type=int, default=5, help="Save every Nth frame (default: 5)."
    )
    p.add_argument(
        "-o",
        "--out",
        default="annotated_frames",
        help="Directory to save PNG frames (default: annotated_frames).",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Number of reconnect attempts on failure (default: 0 = no reconnect).",
    )
    p.add_argument(
        "--reconnect-wait",
        type=float,
        default=2.0,
        help="Seconds to wait before reconnecting (default: 2.0).",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show the live video window (press 'q' to quit).",
    )
    return p

def prefer_tcp_for_rtsp():
    """
    In OpenCV's FFmpeg backend, TCP can be enforced with OPENCV_FFMPEG_CAPTURE_OPTIONS.
    We set it if not already present.
    """
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    opts = os.environ.get(key, "")
    # Ensure rtsp_transport;tcp is present (pipe-separated key;value pairs)
    parts = [p for p in opts.split("|") if p]
    have_tcp = any(seg.strip().lower().startswith("rtsp_transport;tcp") for seg in parts)
    if not have_tcp:
        parts.insert(0, "rtsp_transport;tcp")
    # Sensible timeouts (microseconds in FFmpeg)
    if not any(p.lower().startswith("stimeout;") for p in parts):
        parts.append("stimeout;5000000")  # 5s socket timeout
    if not any(p.lower().startswith("max_delay;") for p in parts):
        parts.append("max_delay;5000000")  # 5s max delay to drop late packets
    os.environ[key] = "|".join(parts)


def open_rtsp_tcp(urls: List[str]) -> Optional[cv2.VideoCapture]:
    """
    Try to open each RTSP URL using FFmpeg backend, preferring TCP.
    """
    # Force TCP in FFmpeg
    prefer_tcp_for_rtsp()

    # Try to reduce initial open latency if supported
    open_timeout_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)

    for url in urls:
        print(f"🔌 Trying (TCP) RTSP: {url}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if open_timeout_prop is not None:
            # 5s open timeout
            cap.set(open_timeout_prop, 5000.0)
        if read_timeout_prop is not None:
            # 5s per read timeout
            cap.set(read_timeout_prop, 5000.0)

        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"✅ Connected via TCP: {url}")
                return cap
        cap.release()
        print(f"⚠️  Failed to open: {url}")
    return None


def save_frame_png(frame, out_dir: str, frame_idx: int) -> Optional[str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fn = f"annotated_frame_f{frame_idx:06d}.png"
    path = os.path.join(out_dir, fn)
    ok = cv2.imwrite(path, frame)
    return path if ok else None


def capture_loop(
    urls: List[str],
    nth: int,
    out_dir: str,
    max_retries: int,
    reconnect_wait: float,
    display: bool,
):
    os.makedirs(out_dir, exist_ok=True)

    attempt = 0
    total_processed = 0
    total_saved = 0

    while True:
        cap = open_rtsp_tcp(urls)
        if cap is None:
            if attempt >= max_retries:
                print("❌ Could not connect to any RTSP stream over TCP.")
                print("🔧 Checks:")
                print("   1) Are containers/services up?   e.g. docker ps")
                print("   2) Server logs (MediaMTX):       docker logs <container>")
                print("   3) Network/firewall allows TCP?  (RTSP interleaved)")
                print("   4) Correct credentials/URL?")
                break
            attempt += 1
            print(f"🔁 Reconnect attempt {attempt}/{max_retries} in {reconnect_wait}s...")
            time.sleep(reconnect_wait)
            continue

        frame_idx = 0
        attempt = 0  # reset after a successful connect

        print(f"📁 Saving frames to: {out_dir}")
        print(f"🎬 Capturing every {nth} frame (press 'q' to quit)...")

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("read failure / timeout")

                frame_idx += 1
                total_processed += 1

                if frame_idx % nth == 0:
                    saved = save_frame_png(frame, out_dir, total_processed)
                    if saved:
                        total_saved += 1
                        print(f"💾 Saved frame {total_processed}: {os.path.basename(saved)}")
                    else:
                        print(f"❌ Failed to save frame {total_processed}")

                if display:
                    cv2.imshow("RTSP (TCP) Stream", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        print("\n🛑 Stopping (user requested).")
                        return

        except KeyboardInterrupt:
            print("\n🛑 Stopping (Ctrl+C).")
            return
        except Exception as e:
            print(f"⚠️  Stream error: {e}")
            cap.release()
            if display:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            if attempt >= max_retries:
                print("❌ Out of reconnect attempts. Exiting.")
                break
            attempt += 1
            print(f"🔁 Reconnecting {attempt}/{max_retries} in {reconnect_wait}s...")
            time.sleep(reconnect_wait)
            continue
        finally:
            try:
                cap.release()
            except Exception:
                pass

    print("\n📊 Summary")
    print(f"   Total frames processed: {total_processed}")
    print(f"   Frames saved:           {total_saved}")
    print(f"   Output directory:       {out_dir}")


def main():
    args = build_arg_parser().parse_args()

    # Make TCP the default for RTSP in FFmpeg backend
    prefer_tcp_for_rtsp()

    capture_loop(
        urls=args.streams,
        nth=max(1, args.nth),
        out_dir=args.out,
        max_retries=max(0, args.max_retries),
        reconnect_wait=max(0.0, args.reconnect_wait),
        display=args.display,
    )


if __name__ == "__main__":
    main()
