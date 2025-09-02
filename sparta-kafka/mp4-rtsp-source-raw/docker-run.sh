docker stop mp4_rtsp_container
docker rm mp4_rtsp_container

docker run -d \
    --name mp4_rtsp_container \
    --network host \
    -e RTSP_URL="rtsp://localhost:8554/rawstream" \
    mp4-rtsp-source-raw

docker exec -it mp4_rtsp_container \
    ffmpeg -re -stream_loop -1 -i /videos/rte_far_seg_1.mp4 \
        -c:v copy -preset veryfast -tune zerolatency \
        -c:a aac -f rtsp rtsp://localhost:8554/rawstream