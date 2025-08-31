# Command Line Interface (CMD) Directory

This directory contains the command line utilities and service launchers for the Contanos AI Multi-Person Pose Estimation System.

## 📁 Files Overview

| File | Description | Purpose |
|------|-------------|---------|
| `unified_pose_estimation_service.py` | **Main Service Launcher** | Runs all AI services together in a coordinated pipeline |
| `annotation_main.py` | Annotation Service | Runs only the visualization/annotation service |
| `yolo_main.py` | YOLOX Service | Runs only the object detection service |
| `bytetrack_main.py` | ByteTrack Service | Runs only the object tracking service |
| `rtmpos_main.py` | RTMPose Service | Runs only the pose estimation service |
| `dev_pose_estimation_config.yaml` | Configuration File | Contains all service configurations |

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Dependencies**: Install required packages from `requirements.txt`
3. **CUDA (Optional)**: For GPU acceleration (recommended)
4. **Base Services**: MQTT Broker and RTSP Server (see setup options below)

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you're in the project root directory
cd /path/to/contanos
```

### 🐳 Setting Up Base Services (Docker - Recommended)

The easiest way to set up the required MQTT broker and RTSP server is using the provided Docker script:

```bash
# Start base services (MQTT broker + RTSP server + video sources)
./projects/start_pose_estimation_services.sh start_base_services

# This will start:
# - MQTT broker on localhost:1883
# - RTSP server on localhost:8554
# - Video source services for testing
```



### 🔧 Manual Setup (Alternative)

pass 

## 🎯 Running the Services

### Option 1: Unified Service (Recommended)

Run all services together in a coordinated pipeline:

```bash
# First, start base services with Docker
./projects/start_pose_estimation_services.sh start_base_services

# Then run the Python services
python cmd/unified_pose_estimation_service.py

# Or with custom configuration
python cmd/unified_pose_estimation_service.py --config cmd/dev_pose_estimation_config.yaml

# Run with specific devices
python cmd/unified_pose_estimation_service.py --devices cuda --log_level INFO

# Run with some services disabled
python cmd/unified_pose_estimation_service.py --skip_services yolox bytetrack

# Run with custom startup delay
python cmd/unified_pose_estimation_service.py --startup_delay 3
```

#### Service Startup Order
1. **YOLOX** - Object Detection (RTSP input → MQTT output)
2. **RTMPose** - Pose Estimation (RTSP+MQTT input → MQTT output)  
3. **ByteTrack** - Object Tracking (MQTT input → MQTT output)
4. **Annotation** - Visualization Output (RTSP+2×MQTT input → RTSP output)

### Option 2: Individual Services

Run each service separately for development or debugging:

#### YOLOX Object Detection Service
```bash
# Ensure base services are running first
./projects/start_pose_estimation_services.sh start_base_services

# Run YOLOX service only
python cmd/yolo_main.py

# The service will:
# - Read from RTSP stream (default: rtsp://localhost:8554/mystream)
# - Detect human objects in video frames
# - Publish detection results to MQTT (default: localhost:1883/yolox)
```

#### RTMPose Pose Estimation Service
```bash
# Run RTMPose service only
python cmd/rtmpos_main.py

# The service will:
# - Read from RTSP stream AND MQTT detection results
# - Estimate human poses from detected bounding boxes
# - Publish pose estimation results to MQTT (default: localhost:1883/rtmpose)
```

#### ByteTrack Object Tracking Service
```bash
# Run ByteTrack service only
python cmd/bytetrack_main.py

# The service will:
# - Read detection results from MQTT (default: localhost:1883/yolox)
# - Track objects across video frames
# - Publish tracking results to MQTT (default: localhost:1883/bytetrack)
```

#### Annotation Visualization Service
```bash
# Run Annotation service only
python cmd/annotation_main.py

# The service will:
# - Read from RTSP stream AND multiple MQTT topics
# - Combine video with pose/tracking annotations
# - Output annotated video to RTSP (default: rtsp://localhost:8554/outstream)
```

## 🔗 Stream URLs (Docker Setup)

When using the Docker-based setup, these streams are available:

```
📺 RTSP Stream URLs:
   Input stream (raw video): rtsp://localhost:8554/rawstream
   Processed stream: rtsp://localhost:8554/mystream  
   Output stream (annotated): rtsp://localhost:8554/outstream

📡 MQTT Broker:
   Address: localhost:1883
   Topics:
     - yolox: YOLOX detection results
     - rtmpose: RTMPose pose estimation results
     - bytetrack: ByteTrack tracking results
```

You can test the streams with:
```bash
# View input stream
ffplay rtsp://localhost:8554/mystream

# View annotated output
ffplay rtsp://localhost:8554/outstream
```

## ⚙️ Configuration

### Using Configuration File

All services use the `dev_pose_estimation_config.yaml` configuration file by default. You can modify this file to change:

- **Input/Output streams**: RTSP and MQTT endpoints
- **Model parameters**: Input sizes, thresholds
- **Device settings**: CPU/CUDA preferences
- **Service behavior**: Enable/disable individual services

### Command Line Options

The unified service supports extensive command line configuration:

```bash
# View all available options
python cmd/unified_pose_estimation_service.py --help

# Common options:
--config CONFIG                    # Configuration file path
--devices DEVICES                  # Computing device (cuda/cpu)
--log_level LOG_LEVEL             # Logging level (DEBUG/INFO/WARN/ERROR)
--skip_services [SERVICE ...]     # Skip specific services
--startup_delay SECONDS           # Delay between service startups
--backend BACKEND                 # Inference backend (onnxruntime/openvino/tensorrt)
--num_workers_per_device N        # Number of workers per device

# Service-specific input/output configuration:
--yolox_in_rtsp CONFIG            # YOLOX input RTSP configuration
--yolox_out_mqtt CONFIG           # YOLOX output MQTT configuration
--rtmpose_in_rtsp CONFIG          # RTMPose input RTSP configuration
--rtmpose_in_mqtt CONFIG          # RTMPose input MQTT configuration
--rtmpose_out_mqtt CONFIG         # RTMPose output MQTT configuration
# ... and many more
```

## 📊 System Monitoring

The unified service provides real-time monitoring information:

```
📊 System status: YOLOX: Queue=5 | RTMPose: Queue=12 DataDict=8 | ByteTrack: OutputQueue=3 | Annotation: Queue=15
```

This shows:
- **Queue sizes**: Number of pending items in input/output queues
- **DataDict sizes**: Internal data structure sizes
- **Processing status**: Real-time processing statistics

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'contanos'
```
**Solution**: Ensure you're running from the project root directory:
```bash
cd /path/to/contanos
python cmd/unified_pose_estimation_service.py
```

#### 2. Configuration File Not Found
```bash
Failed to load config file: No such file or directory
```
**Solution**: Specify the full path to config file:
```bash
python cmd/unified_pose_estimation_service.py --config cmd/dev_pose_estimation_config.yaml
```

#### 3. MQTT Connection Failed
```bash
Error connecting to MQTT broker
```
**Solution**: Ensure MQTT broker is running:
```bash
# Install and start mosquitto MQTT broker
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

#### 4. RTSP Stream Issues
```bash
Failed to connect to RTSP stream
```
**Solution**: Ensure RTSP server is running and stream exists:
```bash
# Check if stream is available
ffplay rtsp://localhost:8554/mystream
```



#### 6. Service Startup Failures
If individual services fail to start, check:
- Required model files are downloaded
- Input streams are available
- Output destinations are accessible
- No port conflicts exist

### Debug Mode

For detailed debugging information:

```bash
# Enable debug logging
python cmd/unified_pose_estimation_service.py --log_level DEBUG

# Run individual services for isolated testing
python cmd/yolo_main.py  # Test YOLOX only
python cmd/rtmpos_main.py  # Test RTMPose only
```

## 📝 Logging

Logs are written to:
- **Console**: Real-time status and error information
- **Log files**: Located in `logs/` directory (if configured)

Log levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARN`: Warning messages
- `ERROR`: Error messages

## 🔄 Service Dependencies

When running individual services, ensure proper startup order:

1. **YOLOX** (no dependencies)
2. **RTMPose** (depends on YOLOX output)
3. **ByteTrack** (depends on YOLOX output)
4. **Annotation** (depends on RTMPose and ByteTrack output)

## 📈 Performance Tips

1. **Use GPU**: Enable CUDA for better performance
   ```bash
   python cmd/unified_pose_estimation_service.py --devices cuda
   ```

2. **Adjust Worker Count**: Increase workers for better throughput
   ```bash
   python cmd/unified_pose_estimation_service.py --num_workers_per_device 2
   ```

3. **Monitor Queue Sizes**: Watch for bottlenecks in system status logs

4. **Optimize Input Resolution**: Balance quality vs. performance in config file

## 🛑 Stopping Services

- **Graceful shutdown**: Press `Ctrl+C` once and wait for cleanup
- **Force stop**: Press `Ctrl+C` multiple times (not recommended)

The unified service automatically handles:
- Service cleanup
- Interface disconnection
- Resource deallocation
- Graceful shutdown of all components

## 🔗 Related Documentation

- **Configuration Guide**: See `dev_pose_estimation_config.yaml` for detailed parameter documentation
- **API Documentation**: Check individual service modules for API details
- **Docker Setup**: See `projects/README_DOCKER_SERVICES.md` for containerized deployment

---

**For additional help or issues, check the project documentation or contact the development team.** 