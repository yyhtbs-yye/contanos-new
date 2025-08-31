# Contanos

**Contanos** is a lightweight Python framework that streamlines the containerization of machine learning (ML), deep learning (DL), and data visualization components using Docker. It provides a structured way to wrap various ML/DL tools into Docker containers with clear input/output interfaces, making it easier to deploy complex AI pipelines across cloud and edge environments. By using Contanos, developers can focus on their ML/DL logic while the framework handles container orchestration, standardized I/O handling, and inter-component communication.

## Project Description

Contanos is designed to simplify how ML and DL modules are packaged and run as microservices. The framework defines a common structure (using base classes and interfaces) for building containerized ML/DL components that can communicate seamlessly. Whether you are deploying a neural network on an edge device or orchestrating multiple AI services in the cloud, Contanos helps by:

- **Containerizing ML/DL Tools**: Each component (e.g., an object detector, a pose estimator, a data annotator) runs in its own Docker container with all dependencies included.
- **Standardizing Interfaces**: Every container uses clear input and output interfaces (such as MQTT messages or RTSP video streams) so that components can be easily connected in a pipeline.
- **Cloud-Edge Deployment**: The framework is lightweight and suitable for resource-constrained edge devices while remaining scalable for cloud deployments. It’s built to handle typical cloud-edge use cases, like an AI service on a server processing data from edge IoT devices.

In essence, Contanos provides the boilerplate and structure needed to turn your ML/DL code into a portable containerized service with minimal effort.

## Features

- **Modular Design**: Contanos encourages a modular architecture where each functional component (e.g., a detector, tracker, or visualizer) is a self-contained module. This modularity makes it easy to develop, test, and deploy components independently or as part of a larger system.
- **Base Classes (`base_\*` Modules)**: The core of the framework is the **`base_\*` series of modules**:
  - **`base_service.py`** – A base class for service components (often called "main" services) that coordinate input and output flow and manage one or more workers.
  - **`base_worker.py`** – A base class for worker components that carry out the heavy ML/DL processing (e.g., running inference on a model). Workers handle reading input data, running the prediction, and writing outputs. The framework supports running multiple workers in parallel for scalability.
  - **`base_processor.py`** – A base class for processing units that can be used for intermediate processing or custom logic. (This can be extended if you want a component that processes data in-line without the full service/worker split.)
- **Standard Input/Output Interfaces**: Contanos comes with a set of pre-built I/O interface modules under `contanos/io`:
  - **MQTT messaging** – Interfaces for reading from and writing to MQTT topics (useful for sensor data, IoT messaging, or chaining modules via a message broker).
  - **RTSP video streams** – Interfaces to ingest video streams via RTSP (for camera input) and to output processed video or data to RTSP or other endpoints.
  - **Multi-source input** – A `MultiInputInterface` that can combine multiple input sources or streams and feed them into the processing pipeline.
  - These interfaces parse standardized connection URIs (e.g., an MQTT source might be given as a URI like `mqtt://broker:1883?topic=...`, and RTSP streams with optional parameters). This uniform approach makes configuring sources and destinations very straightforward.
- **Built-in Monitoring and Utilities**: The framework includes utility classes and tools to help manage and debug deployments:
  - **QueueMonitor** – Monitors internal queues and worker performance (helpful to diagnose bottlenecks in the pipeline, e.g., if a worker is slower than input rate).
  - **Logging and Config Parsing** – Helpers for setting up logging and parsing configuration strings for inputs/outputs are provided (making it easy to specify complex configs via environment variables or CLI arguments).
  - **Visualization Tools** – Under `contanos/visualizer`, you’ll find utilities like `box_drawer.py`, `skeleton_drawer.py`, and `trajectory_drawer.py` for drawing bounding boxes, skeletons (pose estimation), and trajectories on images. These can be used in modules that produce visual output (for example, an annotation module that overlays detection results on video frames).
- **Example Projects**: Contanos comes with several example modules (in the `projects/` directory) that demonstrate how to use the framework for real-world tasks:
  - **YOLOX Object Detection (`prj-yolox-onnx`)** – Detect objects in images or video using a YOLOX model (with ONNX Runtime). This example shows a service that reads frames (e.g., from an RTSP camera), uses a worker to run the YOLOX model, and publishes detections to an MQTT topic.
  - **ByteTrack Object Tracking (`prj-bytetrack-cpu`)** – Track objects across frames on CPU. This module can subscribe to detection results (for example, from YOLOX) via MQTT, then assign consistent IDs to objects as they move between frames, publishing tracking results.
  - **RTMPose Pose Estimation (`prj-rtmpose-onnx`)** – Estimate human poses from video. This service can take a video stream and produce pose keypoints. It can be configured to work alongside an object detector (e.g., only run pose on detected persons) by receiving detections via MQTT.
  - **Annotation/Visualization (`prj-annotation-cpu`)** – Annotate frames with results (e.g., draw bounding boxes or keypoints). This module can consume detection or tracking data and overlay the information on the original images or video, effectively creating a visualized output stream.
  - Each of these projects is built using the Contanos base classes and demonstrates how to extend them for specific use cases. They also come with Dockerfiles for containerization.
- **Lightweight and Flexible**: The framework itself is lightweight (primarily Python code with minimal overhead). You can use Contanos for simple one-container deployments or orchestrate multiple containers together. Because communication is via standard protocols (like MQTT or HTTP/RTSP), Contanos components can be mixed and matched and even integrated with non-Contanos services.

## Installation and Setup

To get started with Contanos, follow these steps:

1. **Prerequisites**: Make sure you have **Docker** installed on your system (Docker Engine for Linux or Docker Desktop for Windows/Mac). No specific Python installation is required on the host, since all Python code runs inside containers.

2. **Clone the Repository**:

   ```
   git clone https://github.com/yyhtbs-yye/contanos.git
   cd contanos
   ```

3. **Build the Docker Images**: The repository includes Dockerfiles for each example project (and a base image for ONNX). You can build whichever component you need. For example:

   - *Optional:* Build the base ONNX GPU image (if you plan to use GPU-accelerated projects like YOLOX or RTMPose):

     ```
     cd projects/base-onnx-gpu
     docker build -t contanos:base-onnx-gpu .
     cd ../../  # return to repository root
     ```

     This image includes the common dependencies (CUDA, cuDNN, ONNX Runtime, etc.) for GPU-based models.

   - Build a project image. For instance, to build the YOLOX object detection service:

     ```
     cd projects/prj-yolox-onnx
     docker build -t contanos-yolox .
     ```

     Similarly, you can build other projects:

     - `projects/prj-bytetrack-cpu` (CPU-based tracking) – e.g. `docker build -t contanos-bytetrack .`
     - `projects/prj-rtmpose-onnx` (pose estimation) – e.g. `docker build -t contanos-rtmpose .`
     - `projects/prj-annotation-cpu` (annotation/visualization) – e.g. `docker build -t contanos-annotation .`
       Each Dockerfile will pull in the Contanos framework and the required dependencies for that project. The build process may take some time, especially for the GPU images, as it will install ML frameworks and models.

4. **Verify Installation** (optional): After building, you can list your Docker images to ensure they were created:

   ```
   docker images | grep contanos
   ```

   You should see entries for the images you built (e.g., `contanos-yolox`, etc.).

With the images ready, you can proceed to run the containers as needed.

## Usage

Using Contanos containers typically involves running one or more Docker containers and configuring their input/output connections via environment variables. The framework is designed so that each container will start the appropriate service or worker automatically on launch, based on how the Dockerfile’s command is configured.

**General Usage Pattern:**

- **Configure Input Source**: Decide how the module will get its input:
  - For video or image streams, you might use an RTSP URL (for example, from an IP camera or streaming server).
  - For data or detection results, you might use an MQTT topic (with a broker URL and topic name).
- **Configure Output Destination**: Decide where the module will send its output:
  - Many modules output results to an MQTT topic (which could be consumed by another module or a dashboard).
  - Some modules (like the annotation module) might output an RTSP stream or save images/video with annotations, depending on configuration.

**Environment Variables for Configuration:**

Contanos uses environment variables to pass configuration parameters into the container at runtime. Some common variables (supported by the example projects) include:

- **`IN_RTSP`** – RTSP input source for video frames. This is typically a URI of the format:
  `rtsp://<camera_ip>:<port>/<stream_path>,topic=<name>`
  Example: `rtsp://192.168.1.10:8554/live.sdp,topic=camera1`
  The `topic` parameter in the URI can be used to tag the stream (useful if a service publishes frames or needs to identify the source).
- **`IN_MQTT`** – MQTT input source for data. Given as:
  `mqtt://<broker_host>:<port>,topic=<topic_name>[,qos=<qos_level>]`
  Example: `mqtt://localhost:1883,topic=yolox/detections,qos=1`
  A module with `IN_MQTT` set will subscribe to the specified MQTT topic to receive incoming data (e.g., detection results from another module).
- **`OUT_MQTT`** – MQTT output destination. Format is similar to IN_MQTT:
  `mqtt://<broker_host>:<port>,topic=<topic_name>[,qos=<qos_level>][,queue_max_len=<N>]`
  Example: `mqtt://localhost:1883,topic=yolox/detections,qos=1,queue_max_len=50`
  When a module produces results (detections, tracking info, etc.), it will publish messages to this MQTT topic. The optional `queue_max_len` can define an internal buffer length if needed.
- **`DEVICES`** – Which device(s) to use for computation. For example `"cuda:0"` to use the first GPU, or `"cpu"` to force CPU execution. Some projects support multiple devices (for running multiple workers), e.g., `"cuda:0,cuda:1"` if you want to utilize two GPUs.
- **`MODEL_INPUT_SIZE`** – The expected input resolution for the model (if applicable). For example `"640 640"` might be used to resize input images to 640x640 for a YOLOX model.
- **`BACKEND`** – The inference backend to use, such as `"onnxruntime"` or `"tensorflow"` etc., depending on what the project supports. (Most provided examples use ONNX runtime or simple CPU processing.)

Each project’s Dockerfile sets default values for these variables (often pointing to example addresses or common defaults), but you will likely need to override them to match your environment.

**Running a Container:**

To run a Contanos container, use `docker run` with the appropriate environment variables. Here are a few examples:

- *YOLOX Object Detection Service:* Run the YOLOX container to read from an RTSP camera stream and publish detections to MQTT:

  ```
  docker run -d --rm \
    -e IN_RTSP="rtsp://<CAMERA_IP>:8554/live,topic=mycam" \
    -e OUT_MQTT="mqtt://localhost:1883,topic=yolox/detections,qos=1" \
    -e DEVICES="cuda:0" \
    contanos-yolox
  ```

  In this example, replace `<CAMERA_IP>` with your RTSP camera address, and `<BROKER_IP>` with your MQTT broker address. The container will start the YOLOX main service which connects to the camera, performs object detection using the GPU (on `cuda:0`), and publishes results to the `yolox/detections` topic on the MQTT broker.

- *ByteTrack Tracking Service:* Suppose you have YOLOX producing detections as above. You can run the ByteTrack container to subscribe to those detections and output tracked objects:

  ```
  docker run -d --rm \
    -e IN_MQTT="mqtt://<BROKER_IP>:1883,topic=yolox/detections,qos=1" \
    -e OUT_MQTT="mqtt://<BROKER_IP>:1883,topic=bytetrack/tracks,qos=1" \
    contanos-bytetrack
  ```

  This will start the tracking service on CPU, reading incoming detection messages and publishing annotated tracking results (each object gets an ID and trajectory) to the `bytetrack/tracks` topic.

- *RTMPose Pose Estimation Service:* You can use the RTMPose container either directly on a video stream or in conjunction with detection data:

  - **Direct video input**: Provide an `IN_RTSP` (video of people) and it will output poses to an MQTT topic.

  - **Using detection input**: Provide both `IN_RTSP` (for the video frames) and `IN_MQTT` (for detection boxes of persons in those frames). The service will use the detection input to focus pose estimation on those regions. For example:

    ```
    docker run -d --rm \
      -e IN_RTSP="rtsp://<CAMERA_IP>:8554/live,topic=mycam" \
      -e IN_MQTT="mqtt://<BROKER_IP>:1883,topic=yolox/detections" \
      -e OUT_MQTT="mqtt://<BROKER_IP>:1883,topic=rtmpose/poses" \
      -e DEVICES="cuda:0" \
      contanos-rtmpose
    ```

    Here, the RTMPose container will subscribe to both the camera stream and the detection results from YOLOX, then publish pose keypoints to `rtmpose/poses`.

- *Annotation/Visualization Service:* After you have detection or tracking data, you might want to visualize it. The annotation module can subscribe to a topic (e.g., detections or tracks) and overlay the information on the video frames:

  ```
  docker run -d --rm \
    -e IN_RTSP="rtsp://<CAMERA_IP>:8554/live,topic=mycam" \
    -e IN_MQTT="mqtt://<BROKER_IP>:1883,topic=bytetrack/tracks" \
    -e OUT_MQTT="mqtt://<BROKER_IP>:1883,topic=annotation/frames" \
    contanos-annotation
  ```

  In this scenario, the annotation service takes the original camera stream and the tracking results, draws boxes and labels on each frame, and publishes the annotated frames (perhaps as images or a video stream) to a new topic. You could have a simple subscriber that pulls those frames to display on a UI, or stream them out via RTSP (depending on implementation).

**Connecting Modules Together:** As shown in the examples, you can chain modules by aligning the output of one to the input of another. Because MQTT is used as a medium, modules don’t need to know about each other explicitly – one just needs to publish to a topic that the other is listening on. This decoupling makes it easy to add or remove components or even swap one implementation for another (for example, you could replace the YOLOX detector with another detector, as long as it publishes results in the expected format to the same topic).

**Scaling and Multi-Worker Setup:** By design, the base service class (`BaseService`) can manage multiple worker processes. In practice, this means you could configure a service to launch several worker instances (for example, if you want to utilize multiple CPU cores or GPUs for parallel processing). The details of configuring multiple workers will depend on the specific project (some may auto-scale based on the `DEVICES` list or other parameters). For cloud deployments, you might also run multiple container instances behind a load balancer or orchestrate with Kubernetes – Contanos is flexible to accommodate these patterns thanks to its stateless, message-driven design.

**Logs and Monitoring:** Each container will log its activity (to stdout by default, which you can view using `docker logs`). The logs include informational messages and warnings (for example, if an input read is slow or if a worker crashes). If you enabled the `QueueMonitor` in a service, you will also see periodic logs of queue lengths and processing timings, which help in debugging performance issues. These logs can guide you to tune parameters like number of workers or queue sizes.

## Directory Structure

The repository is organized into the following key directories and files:

```
contanos/             # Core framework package
├── base_processor.py    # Base class for processing units (common logic for modules)
├── base_service.py      # Base class for service (main orchestrator that manages I/O and workers)
├── base_worker.py       # Base class for worker (handles model inference loop)
├── helpers/             # Helper scripts to simplify creating processes
│   ├── create_a_processor.py  # Utility to create a processor instance from config
│   └── start_a_service.py     # Utility to launch a service with given parameters
├── io/                  # Input/Output interface implementations
│   ├── mqtt_input_interface.py        # MQTT input (subscribe to topic for data)
│   ├── mqtt_output_interface.py       # MQTT output (publish results to topic)
│   ├── mqtt_sorted_input_interface.py # MQTT input that maintains order or sorting
│   ├── multi_input_interface.py       # Interface to combine multiple inputs
│   ├── rtsp_input_interface.py        # RTSP input (pull video frames from an RTSP source)
│   └── rtsp_output_interface.py       # RTSP output (stream out video/frames via RTSP or related sink)
├── utils/               # Utility functions and classes
│   ├── create_args.py       # Define and parse command-line arguments for modules
│   ├── create_configs.py    # Helpers to create config objects from raw inputs
│   ├── format_results.py    # Utility to format model results into serializable output (e.g., JSON)
│   ├── parse_config_string.py  # Parser for config strings like the MQTT/RTSP URIs
│   └── setup_logging.py     # Convenient setup for logging format and levels
└── visualizer/         # Visualization utilities for drawing on images/frames
    ├── box_drawer.py       # Draw bounding boxes and labels on images
    ├── skeleton_drawer.py  # Draw skeletal keypoints (for pose estimation)
    └── trajectory_drawer.py# Draw trajectories for tracked objects
projects/             # Example project modules built on Contanos
├── prj-annotation-cpu/    # Annotation/visualization module (CPU-based)
│   ├── annotation_main.py    # Service that subscribes to data and original frames
│   ├── annotation_worker.py  # Worker that draws annotations on frames
│   └── Dockerfile            # Dockerfile for containerizing this module
├── prj-bytetrack-cpu/    # ByteTrack tracking module (CPU-based)
│   ├── bytetrack_main.py     # Service that subscribes to detections and publishes tracks
│   ├── bytetrack_worker.py   # Worker that performs tracking computations
│   └── Dockerfile            # Dockerfile for this module
├── prj-rtmpose-onnx/     # RTMPose pose estimation module (uses ONNX, can use GPU)
│   ├── rtmpose_main.py       # Service coordinating video frames and (optionally) detections for pose
│   ├── rtmpose_worker.py     # Worker that runs the pose estimation model
│   └── Dockerfile            # Dockerfile for this module
├── prj-yolox-onnx/       # YOLOX object detection module (ONNX model, GPU-supported)
│   ├── yolox_main.py         # Service that reads frames and sends them to worker(s)
│   ├── yolox_worker.py       # Worker that runs the YOLOX inference on frames
│   └── Dockerfile            # Dockerfile for this module
└── base-onnx-gpu/ # Base Docker image definition for ONNX and GPU support 
    └── Dockerfile            # Dockerfile for the base image (CUDA, ONNX Runtime, etc.)
analyzer/             # Utility scripts for development and debugging
├── mqtt_sniffer.py        # Script to subscribe to all MQTT topics ( "#" ) and log messages (useful for debugging message flows)
├── queue_monitor.py       # Script or module to monitor internal queue lengths and worker status in a running service
└── test_rtsp.py           # Simple tester for RTSP streams (to verify a camera feed is accessible)
```

Other files in the repository include:

- **`pyproject.toml`** – Build configuration for the project (used if you want to install contanos as a Python package or for development).
- **`requirements.txt`** – Python dependencies listed for the project (these are installed in the Docker images; it’s not typically needed on the host).
- **`.gitignore`** – Git ignore file for the repository.
  *(At the moment, the project does not include a separate `LICENSE` file, but the license is stated below.)*

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute this software in your own projects, including for commercial purposes, as long as the terms of the MIT License are met. (The MIT license is very permissive – it basically requires including a copy of the license in any redistributed versions of this software.)

See the MIT License text below for details:

```
swiftCopyEditMIT License

Copyright (c) 2025 yyhtbs-yye

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

*(Full license text would be included here or in a separate LICENSE file.)*



## Contributing and Contact Info

Contributions to Contanos are welcome! If you have ideas for improvements, find a bug, or want to add new features (such as support for additional input/output types or new example modules), please consider contributing:

## Acknowledgement
This work has received funding as an Open Call project under the aerOS project (”Autonomous, scalablE, tRustworthy, intelligent European meta Operating System for the IoT edgecloud continuum”), funded by the European Union’s Horizon Europe programme under grant agreement No. 101069732.


- **Submit Issues**: If you encounter any problems or have questions, please open an issue on the GitHub repository. This helps track known issues and discussions.
- **Pull Requests**: Feel free to fork the repository and submit pull requests. Whether it’s a bug fix, new feature, or improved documentation, we will review PRs and integrate them if they align with the project goals. Before starting large changes, it might be good to open an issue to discuss your plans.
- **Project Style**: Try to follow the coding style of the project for consistency (PEP8 for Python code, and similar structure to existing modules if creating new ones). Write clear commit messages and document new code as needed.

**Contact**: For any inquiries, you can reach out by opening an issue (which is actively monitored by the maintainer). If you need to contact the maintainer directly, you can find contact links on the maintainer’s GitHub profile (`yyhtbs-yye` on GitHub).
