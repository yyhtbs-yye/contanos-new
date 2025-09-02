# ---------- base image ----------
FROM ubuntu:24.04

# ---------- build-time settings ----------
ARG OUT_RTSP_URL=rtsp://192.168.200.206:8554/mystream        # where to publish
ARG IN_RTSP_URL=rtsp://192.168.200.206:8554/rawstream            # where to receive

RUN apt update
RUN apt install -y sudo git build-essential yasm nasm
RUN git clone https://github.com/yyhtbs-yye/x264_sei
RUN cd x264_sei && \
    ./configure --prefix=/usr/local --enable-shared --enable-pic && \
    make -j$(nproc) && \
    make install

RUN apt install -y pkg-config libvpx-dev libx265-dev libfdk-aac-dev libmp3lame-dev libopus-dev
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg-custom
RUN cd ffmpeg-custom && \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig ./configure   --prefix=/usr/local --enable-gpl --enable-libx264 --extra-cflags="-I/usr/local/include" --extra-ldflags="-L/usr/local/lib" && \
    make -j$(nproc) && \
    make install

# ---------- runtime env (they survive the build) ----------
ENV IN_RTSP_URL=${IN_RTSP_URL}
ENV OUT_RTSP_URL=${OUT_RTSP_URL}
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# ---------- set working directory ----------
WORKDIR /app
