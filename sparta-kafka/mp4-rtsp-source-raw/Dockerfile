# ---------- base image ----------
FROM jrottenberg/ffmpeg:4.4-ubuntu

# ---------- build-time settings ----------
ARG RTSP_URL=rtsp://localhost:8554/rawstream
ARG VIDEO_FILE=video.mp4

# ---------- runtime env (they survive the build) ----------
ENV RTSP_URL=${RTSP_URL}
ENV VIDEO_FILE=${VIDEO_FILE}

# ---------- set working directory ----------
WORKDIR /source-videos

# ---------- stream on container start ----------
ENTRYPOINT ["sh", "-c", "\
  echo '🎬 Starting video stream from $VIDEO_FILE to $RTSP_URL...' && \
  if [ ! -f /source-videos/media/$VIDEO_FILE ]; then \
    echo '❌ File not found: /source-videos/media/$VIDEO_FILE' && \
    echo '📁 Available files:' && ls -la /source-videos/media/ 2>/dev/null || echo 'Media directory not found' && \
    exit 1; \
  fi && \
  ffmpeg -re -stream_loop -1 -i /source-videos/media/$VIDEO_FILE \
         -c:v copy -preset veryfast -tune zerolatency \
         -c:a aac -f rtsp $RTSP_URL"]


