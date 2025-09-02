docker run -d \
    --name mp4_sei_injection \
    -e OUT_RTSP_URL="rtsp://192.168.200.206:8554/mystream" \
    -e IN_RTSP_URL="rtsp://192.168.200.206:8554/rawstream" \
    mp4-transcoder-sei