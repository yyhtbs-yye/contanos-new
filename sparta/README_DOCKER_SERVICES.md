# AI Multi-Person Pose Estimation System - Independent Docker Services

This project breaks down the AI multi-person pose estimation system into independent Docker services, communicating through MQTT message queues and managed with YAML configuration files.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RTSP Input    │    │   MQTT Broker   │    │   RTSP Output   │
│ (Video Stream)  │    │  (Message Hub)  │    │ (Visualization) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       ▲
         ▼                       │                       │
┌─────────────────┐              │              ┌─────────────────┐
│  YOLOX Service  │─────────────►│              │ Annotation Svc  │
│   (Detection)   │              │              │ (Visualization) │
└─────────────────┘              │              └─────────────────┘
         │                       │                       ▲
         ▼                       │                       │
┌─────────────────┐              │              ┌─────────────────┐
│ RTMPose Service │─────────────►│              │  ByteTrack Svc  │
│ (Pose Estimate) │              │              │   (Tracking)    │
└─────────────────┘              │              └─────────────────┘
         │                       │                       ▲
         └───────────────────────┼───────────────────────┘
                                │
                         MQTT Topics:
                         - yolox
                         - rtmpose  
                         - bytetrack
```

## 📦 Service Components

### 1. Base Services
- **MQTT Broker**: Message queue broker, handles inter-service communication
- **RTSP Server**: Streaming media server, handles video input/output
- **MP4 Source**: Video source service

### 2. AI Processing Services
- **YOLOX Service**: Human object detection service
- **RTMPose Service**: Human pose estimation service  
- **ByteTrack Service**: Object tracking service
- **Annotation Service**: Visualization annotation service

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (for YOLOX and RTMPose services)
- nvidia-docker runtime

### 1. Configuration File

Ensure `pose_estimation_config.yaml` exists in the project root directory:

```yaml
# Key configuration example
global:
  devices: "cuda"
  log_level: "INFO"
  backend: "onnxruntime"

yolox:
  devices: "cuda"
  input:
    config: "rtsp://rtsp-server:8554,topic=mystream"
  output:
    config: "mqtt://mqtt-broker:1883,topic=yolox,qos=2,queue_max_len=50"

# ... other service configurations
```

### 2. Start Services

```bash
# Enter project directory
cd projects/
docker build -t contanos:base-onnx-gpu ./base-onnx-gpu
# Start all services
./start_pose_estimation_services.sh start

# Or use docker-compose directly
docker-compose up -d
```

### 3. Check Service Status

```bash
# Check service status
./start_pose_estimation_services.sh status

# View service logs
./start_pose_estimation_services.sh logs

# View specific service logs
docker-compose logs -f yolox-service
```

## 🔧 Management Commands

```bash
# Start services
./start_pose_estimation_services.sh start

# Stop services
./start_pose_estimation_services.sh stop

# Restart services
./start_pose_estimation_services.sh restart

# View status
./start_pose_estimation_services.sh status

# View logs
./start_pose_estimation_services.sh logs
```

## 📺 Access URLs

### RTSP Stream URLs
- **Input stream (raw video)**: `rtsp://localhost:8554/rawstream`
- **Processed stream**: `rtsp://localhost:8554/mystream`
- **Output stream (annotated)**: `rtsp://localhost:8554/outstream`

### MQTT Topics
- **MQTT Broker**: `localhost:1883`
- **Detection results**: `yolox`
- **Pose estimation**: `rtmpose`
- **Tracking results**: `bytetrack`

## 🐳 Docker File Structure

```
projects/
├── docker-compose.yml              # Main orchestration file
├── pose_estimation_config.yaml     # Configuration file (needs link from project root)
├── start_pose_estimation_services.sh  # Startup script
├── prj-yolox-onnx/
│   ├── Dockerfile.yaml            # YOLOX Docker file
│   ├── yolox_main_yaml.py         # YOLOX main program (YAML config version)
│   └── yolox_worker.py            # YOLOX worker process
├── prj-rtmpose-onnx/
│   ├── Dockerfile.yaml            # RTMPose Docker file
│   ├── rtmpose_main_yaml.py       # RTMPose main program (YAML config version)
│   └── rtmpose_worker.py          # RTMPose worker process
├── prj-bytetrack-cpu/
│   ├── Dockerfile.yaml            # ByteTrack Docker file
│   ├── bytetrack_main_yaml.py     # ByteTrack main program (YAML config version)
│   └── bytetrack_worker.py        # ByteTrack worker process
└── prj-annotation-cpu/
    ├── Dockerfile.yaml            # Annotation Docker file
    ├── annotation_main_yaml.py    # Annotation main program (YAML config version)
    └── annotation_worker.py       # Annotation worker process
```

## 📊 Performance Monitoring

### View Container Status
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### View Resource Usage
```bash
docker stats
```

### View GPU Usage
```bash
nvidia-smi
```

## 🔍 Troubleshooting

### Common Issues

1. **Service startup failure**
   ```bash
   # Check logs
   docker-compose logs [service-name]
   
   # Rebuild
   docker-compose build [service-name]
   ```

2. **GPU unavailable**
   ```bash
   # Check nvidia-docker
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Configuration file not found**
   ```bash
   # Ensure configuration file exists
   ls -la ../pose_estimation_config.yaml
   ```

### Debugging Steps

1. Check if base services are running properly
2. Start AI services one by one and check logs
3. Test message passing using MQTT client
4. Check if RTSP streams are accessible

## 🔄 Service Dependencies

```
mqtt-broker + rtsp-server (base services)
    ↓
mp4-rtsp-source + mp4-transcode-sei (video sources)
    ↓
yolox-service (object detection)
    ↓
rtmpose-service + bytetrack-service (pose estimation + tracking)
    ↓
annotation-service (visualization)
```

## 📝 Configuration Customization

### Modify Configuration Parameters

Edit the `../pose_estimation_config.yaml` file:

```yaml
# Modify device configuration
global:
  devices: "cpu"  # or "cuda"

# Modify model parameters
yolox:
  model_input_size: [640, 640]

# Modify MQTT configuration
mqtt:
  broker_host: "mqtt-broker"
  qos: 2
```

### Environment Variable Override

You can also override configuration through environment variables:

```bash
export DEVICES=cpu
export LOG_LEVEL=DEBUG
docker-compose up -d
```

