import cv2
import os
import time

def capture_rtsp_to_mp4():
    # RTSP stream URL
    rtsp_url = "rtsp://192.168.200.207:5108/annotated_stream?tcp"

    # Output file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "capture.mp4")

    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return

    # Try to read the first frame to get size
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return

    height, width = frame.shape[:2]

    # Pick FPS from stream if available, otherwise default
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:  # NaN check and bogus values
        fps = 25.0
    fps = 25.0

    # Initialize VideoWriter (try a few common MP4 codecs)
    writer = None
    used_codec = None
    for codec in ("mp4v", "avc1", "H264", "X264"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            used_codec = codec
            break

    if writer is None or not writer.isOpened():
        print("Error: Could not open VideoWriter for MP4. "
              "Make sure your OpenCV build has FFMPEG/appropriate codecs.")
        cap.release()
        return

    print(f"Recording to {output_path} at {fps:.2f} FPS using codec {used_codec}")

    frame_count = 0
    max_frames = 750  # Stop after this many frames (adjust or remove to record indefinitely)

    try:
        # We already have the first frame; write it
        writer.write(frame)
        frame_count += 1

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            writer.write(frame)
            frame_count += 1

            if frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\nCapture stopped by user")

    finally:
        cap.release()
        writer.release()
        print(f"Total frames written: {frame_count}")
        print("Saved:", output_path)

if __name__ == "__main__":
    capture_rtsp_to_mp4()
