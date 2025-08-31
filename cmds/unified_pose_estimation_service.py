#!/usr/bin/env python3
"""
Unified AI Multi-Person Pose Estimation Service Launcher
Combines YOLOX detection, RTMPose pose estimation, ByteTrack tracking and Annotation visualization services
"""
import os
import sys
import asyncio
import logging
import argparse
import traceback
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import signal

# Set project paths - Fixed to use correct project root
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from cmd directory
sys.path.insert(0, str(PROJECT_ROOT))

# Add service paths
YOLOX_PATH = PROJECT_ROOT / "projects" / "prj-yolox-onnx"
RTMPOSE_PATH = PROJECT_ROOT / "projects" / "prj-rtmpose-onnx" 
BYTETRACK_PATH = PROJECT_ROOT / "projects" / "prj-bytetrack-cpu"
ANNOTATION_PATH = PROJECT_ROOT / "projects" / "prj-annotation-cpu"

for path in [YOLOX_PATH, RTMPOSE_PATH, BYTETRACK_PATH, ANNOTATION_PATH]:
    sys.path.insert(0, str(path))

# Import common modules
from contanos.utils.create_args import  add_service_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string

class ServiceManager:
    """Service manager responsible for managing the lifecycle of all AI services"""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(__name__)
        self.shutdown_event = asyncio.Event()
        
    async def start_yolox_service(self, args) -> bool:
        """Start YOLOX object detection service"""
        service_name = "YOLOX"
        try:
            self.logger.info(f"🚀 Starting {service_name} service...")
            
            # Switch to service directory
            original_cwd = os.getcwd()
            os.chdir(YOLOX_PATH)
            
            # Import service modules
            from yolox_worker import YOLOXWorker
            from contanos.io.rtsp_input_interface import RTSPInput
            from contanos.io.mqtt_output_interface import MQTTOutput
            from contanos.helpers.create_a_processor import create_a_processor
            from contanos.helpers.start_a_service import start_a_service
            
            # Parse configuration
            in_rtsp_config = parse_config_string(args.yolox_in_rtsp)
            out_mqtt_config = parse_config_string(args.yolox_out_mqtt)
            
            # Create interfaces
            input_interface = RTSPInput(config=in_rtsp_config)
            output_interface = MQTTOutput(config=out_mqtt_config)
            
            await input_interface.initialize()
            await output_interface.initialize()
            
            # Model configuration
            model_config = dict(
                onnx_model=('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                           'yolox_m_8xb8-300e_humanart-c2c7a14a.zip'),
                model_input_size=args.model_input_size,
                backend=args.backend,
            )
            
            # Create processor
            devices = args.devices.split(',') if isinstance(args.devices, str) else args.devices
            _, processor = create_a_processor(
                worker_class=YOLOXWorker,
                model_config=model_config,
                devices=devices,
                input_interface=input_interface,
                output_interface=output_interface,
                num_workers_per_device=args.num_workers_per_device,
            )
            
            # Start service - Fix: keep service running continuously
            service_task = asyncio.create_task(start_a_service(
                processor=processor,
                run_until_complete=False,
                daemon_mode=False,  # Changed to False to keep service running
            ))
            
            self.services[service_name] = {
                'service_task': service_task,
                'processor': processor,
                'input_interface': input_interface,
                'output_interface': output_interface
            }
            
            os.chdir(original_cwd)
            self.logger.info(f"✅ {service_name} service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {service_name} service startup failed: {e}")
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
            return False
    
    async def start_rtmpose_service(self, args) -> bool:
        """Start RTMPose pose estimation service"""
        service_name = "RTMPose"
        try:
            self.logger.info(f"🚀 Starting {service_name} service...")
            
            original_cwd = os.getcwd()
            os.chdir(RTMPOSE_PATH)
            
            from rtmpose_worker import RTMPoseWorker
            from contanos.io.rtsp_input_interface import RTSPInput
            from contanos.io.mqtt_output_interface import MQTTOutput
            from contanos.io.mqtt_input_interface import MQTTInput
            from contanos.io.multi_input_interface import MultiInputInterface
            from contanos.helpers.create_a_processor import create_a_processor
            from contanos.helpers.start_a_service import start_a_service
            
            # Parse configuration
            in_rtsp_config = parse_config_string(args.rtmpose_in_rtsp)
            in_mqtt_config = parse_config_string(args.rtmpose_in_mqtt)
            out_mqtt_config = parse_config_string(args.rtmpose_out_mqtt)
            
            # Create interfaces
            input_video_interface = RTSPInput(config=in_rtsp_config)
            input_message_interface = MQTTInput(config=in_mqtt_config)
            input_interface = MultiInputInterface([input_video_interface, input_message_interface])
            output_interface = MQTTOutput(config=out_mqtt_config)
            
            await input_interface.initialize()
            await output_interface.initialize()
            
            # Model configuration
            model_config = dict(
                onnx_model=('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                           'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'),
                model_input_size=args.rtmpose_model_input_size,
                backend=args.backend,
            )
            
            # Create processor
            devices = args.devices.split(',') if isinstance(args.devices, str) else args.devices
            _, processor = create_a_processor(
                worker_class=RTMPoseWorker,
                model_config=model_config,
                devices=devices,
                input_interface=input_interface,
                output_interface=output_interface,
                num_workers_per_device=args.num_workers_per_device,
            )
            
            # Start service - Fix: keep service running continuously
            service_task = asyncio.create_task(start_a_service(
                processor=processor,
                run_until_complete=False,
                daemon_mode=False,  # Changed to False to keep service running
            ))
            
            self.services[service_name] = {
                'service_task': service_task,
                'processor': processor,
                'input_interface': input_interface,
                'output_interface': output_interface
            }
            
            os.chdir(original_cwd)
            self.logger.info(f"✅ {service_name} service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {service_name} service startup failed: {e}")
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
            return False
    
    async def start_bytetrack_service(self, args) -> bool:
        """Start ByteTrack tracking service"""
        service_name = "ByteTrack"
        try:
            self.logger.info(f"🚀 Starting {service_name} service...")
            
            original_cwd = os.getcwd()
            os.chdir(BYTETRACK_PATH)
            
            from bytetrack_worker import ByteTrackWorker
            from contanos.io.mqtt_sorted_input_interface import MQTTSortedInput
            from contanos.io.mqtt_output_interface import MQTTOutput
            from contanos.helpers.create_a_processor import create_a_processor
            from contanos.helpers.start_a_service import start_a_service
            
            # Parse configuration
            in_mqtt_config = parse_config_string(args.bytetrack_in_mqtt)
            out_mqtt_config = parse_config_string(args.bytetrack_out_mqtt)
            
            # Create interfaces
            input_interface = MQTTSortedInput(config=in_mqtt_config)
            output_interface = MQTTOutput(config=out_mqtt_config)
            
            await input_interface.initialize()
            await output_interface.initialize()
            
            # Model configuration
            model_config = dict(
                track_thresh=0.45,
                match_thresh=0.8,
                track_buffer=25,
                frame_rate=30,
                per_class=False
            )
            
            # Create processor
            devices = ['cpu']
            _, processor = create_a_processor(
                worker_class=ByteTrackWorker,
                model_config=model_config,
                devices=devices,
                input_interface=input_interface,
                output_interface=output_interface,
                num_workers_per_device=args.num_workers_per_device,
            )
            
            # Start service - Fix: keep service running continuously
            service_task = asyncio.create_task(start_a_service(
                processor=processor,
                run_until_complete=False,
                daemon_mode=False,  # Changed to False to keep service running
            ))
            
            self.services[service_name] = {
                'service_task': service_task,
                'processor': processor,
                'input_interface': input_interface,
                'output_interface': output_interface
            }
            
            os.chdir(original_cwd)
            self.logger.info(f"✅ {service_name} service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {service_name} service startup failed: {e}")
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
            return False
    
    async def start_annotation_service(self, args) -> bool:
        """Start Annotation visualization service"""
        service_name = "Annotation"
        try:
            self.logger.info(f"🚀 Starting {service_name} service...")
            
            original_cwd = os.getcwd()
            os.chdir(ANNOTATION_PATH)
            
            from annotation_worker import AnnotationWorker
            from contanos.io.rtsp_input_interface import RTSPInput
            from contanos.io.rtsp_output_interface import RTSPOutput
            from contanos.io.mqtt_input_interface import MQTTInput
            from contanos.io.multi_input_interface import MultiInputInterface
            from contanos.helpers.create_a_processor import create_a_processor
            from contanos.helpers.start_a_service import start_a_service
            
            # Parse configuration
            in_rtsp_config = parse_config_string(args.annotation_in_rtsp)
            in_mqtt1_config = parse_config_string(args.annotation_in_mqtt1)
            in_mqtt2_config = parse_config_string(args.annotation_in_mqtt2)
            out_rtsp_config = parse_config_string(args.annotation_out_rtsp)
            
            # Create input interfaces
            input_video_interface = RTSPInput(config=in_rtsp_config)
            input_message_interface1 = MQTTInput(config=in_mqtt1_config)
            input_message_interface2 = MQTTInput(config=in_mqtt2_config)
            input_interface = MultiInputInterface([input_video_interface, input_message_interface1, input_message_interface2])
            
            # Initialize input interfaces
            await input_interface.initialize()
            
            # Auto-detect video dimensions and update output configuration
            output_interface = RTSPOutput(config=out_rtsp_config)
            await output_interface.initialize()
            
            # Model configuration
            model_config = dict()
            
            # Create processor
            devices = ['cpu']  # Annotation service usually runs on CPU
            _, processor = create_a_processor(
                worker_class=AnnotationWorker,
                model_config=model_config,
                devices=devices,
                input_interface=input_interface,
                output_interface=output_interface,
                num_workers_per_device=args.num_workers_per_device,
            )
            
            # Start service - Fix: keep service running continuously
            service_task = asyncio.create_task(start_a_service(
                processor=processor,
                run_until_complete=False,
                daemon_mode=False,  # Changed to False to keep service running
            ))
            
            self.services[service_name] = {
                'service_task': service_task,
                'processor': processor,
                'input_interface': input_interface,
                'output_interface': output_interface
            }
            
            os.chdir(original_cwd)
            self.logger.info(f"✅ {service_name} service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {service_name} service startup failed: {e}")
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
            return False
    
    async def start_monitoring_task(self):
        """Start monitoring task"""
        self.logger.info("🔍 Starting system monitoring task...")
        
        async def monitor():
            while not self.shutdown_event.is_set():
                try:
                    status_info = []
                    for service_name, service_data in self.services.items():
                        if 'input_interface' in service_data:
                            input_interface = service_data['input_interface']
                            if hasattr(input_interface, '_queue'):
                                queue_size = input_interface._queue.qsize()
                                status_info.append(f"{service_name}: Queue={queue_size}")
                            if hasattr(input_interface, 'queue'):
                                queue_size = input_interface.queue.qsize()
                                status_info.append(f"{service_name}: Queue={queue_size}")
                            if hasattr(input_interface, '_data_dict'):
                                data_dict_size = len(input_interface._data_dict)
                                status_info.append(f"{service_name}: DataDict={data_dict_size}")
                        if 'output_interface' in service_data:
                            output_interface = service_data['output_interface']
                            if hasattr(output_interface, 'queue'):
                                queue_size = output_interface.queue.qsize()
                                status_info.append(f"{service_name}: OutputQueue={queue_size}")
                            if hasattr(output_interface, '_data_dict'):
                                data_dict_size = len(output_interface._data_dict)
                                status_info.append(f"{service_name}: OutputDataDict={data_dict_size}")
                    if status_info:
                        self.logger.info(f"📊 System status: {' | '.join(status_info)}")
                    
                    await asyncio.sleep(10)  # Monitor every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Monitoring task error: {e}")
                    await asyncio.sleep(5)
        
        self.tasks['monitor'] = asyncio.create_task(monitor())
    
    async def shutdown_all_services(self):
        """Shutdown all services"""
        self.logger.info("🔄 Starting to shutdown all services...")
        self.shutdown_event.set()
        
        # Cancel monitoring tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                self.logger.info(f"✅ Task cancelled: {task_name}")
        
        # Shutdown all services
        for service_name, service_data in self.services.items():
            try:
                # Cancel service tasks
                if 'service_task' in service_data and not service_data['service_task'].done():
                    service_data['service_task'].cancel()
                    try:
                        await service_data['service_task']
                    except asyncio.CancelledError:
                        pass
                
                # Cleanup interfaces
                if 'input_interface' in service_data:
                    await service_data['input_interface'].cleanup()
                if 'output_interface' in service_data:
                    await service_data['output_interface'].cleanup()
                self.logger.info(f"✅ Service shutdown: {service_name}")
            except Exception as e:
                self.logger.error(f"❌ Error shutting down service {service_name}: {e}")
        
        self.logger.info("🏁 All services have been shutdown")

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not config_file or not os.path.exists(config_file):
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load config file {config_file}: {e}")
        return {}

def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested config value using dot notation (e.g., 'yolox.input.config')"""
    keys = path.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified AI Multi-Person Pose Estimation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Service startup order:
1. YOLOX - Object Detection (RTSP input -> MQTT output)
2. RTMPose - Pose Estimation (RTSP+MQTT input -> MQTT output)  
3. ByteTrack - Object Tracking (MQTT input -> MQTT output)
4. Annotation - Visualization Output (RTSP+2×MQTT input -> RTSP output)

Example:
  python unified_pose_estimation_service.py --config dev_pose_estimation_config.yaml
  python unified_pose_estimation_service.py --devices cuda --log_level INFO
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, default='dev_pose_estimation_config.yaml',
                       help='YAML configuration file path')
    
    # Load config file first to get defaults
    temp_args, _ = parser.parse_known_args()
    config = load_config(temp_args.config)
    
    # YOLOX service configuration
    parser.add_argument('--yolox_in_rtsp', 
                       default=get_config_value(config, 'yolox.input.config', 'rtsp://localhost:8554,topic=mystream'),
                       help='YOLOX input RTSP configuration')
    parser.add_argument('--yolox_out_mqtt', 
                       default=get_config_value(config, 'yolox.output.config', 'mqtt://localhost:1883,topic=yolox,qos=2,queue_max_len=50'),
                       help='YOLOX output MQTT configuration')
    
    # RTMPose service configuration
    parser.add_argument('--rtmpose_in_rtsp', 
                       default=get_config_value(config, 'rtmpose.input.rtsp.config', 'rtsp://localhost:8554,topic=mystream'),
                       help='RTMPose input RTSP configuration')
    parser.add_argument('--rtmpose_in_mqtt', 
                       default=get_config_value(config, 'rtmpose.input.mqtt.config', 'mqtt://localhost:1883,topic=yolox,qos=2,queue_max_len=100'),
                       help='RTMPose input MQTT configuration')
    parser.add_argument('--rtmpose_out_mqtt', 
                       default=get_config_value(config, 'rtmpose.output.config', 'mqtt://localhost:1883,topic=rtmpose,qos=2,queue_max_len=100'),
                       help='RTMPose output MQTT configuration')
    
    # ByteTrack service configuration
    parser.add_argument('--bytetrack_in_mqtt', 
                       default=get_config_value(config, 'bytetrack.input.config', 'mqtt://localhost:1883,topic=yolox,qos=2,buffer_threshold=100'),
                       help='ByteTrack input MQTT configuration')
    parser.add_argument('--bytetrack_out_mqtt', 
                       default=get_config_value(config, 'bytetrack.output.config', 'mqtt://localhost:1883,topic=bytetrack,qos=2,queue_max_len=100'),
                       help='ByteTrack output MQTT configuration')
    
    # Annotation service configuration
    parser.add_argument('--annotation_in_rtsp', 
                       default=get_config_value(config, 'annotation.input.rtsp.config', 'rtsp://localhost:8554,topic=mystream'),
                       help='Annotation input RTSP configuration')
    parser.add_argument('--annotation_in_mqtt1', 
                       default=get_config_value(config, 'annotation.input.mqtt1.config', 'mqtt://localhost:1883,topic=bytetrack,qos=2,queue_max_len=100'),
                       help='Annotation input MQTT1 configuration')
    parser.add_argument('--annotation_in_mqtt2', 
                       default=get_config_value(config, 'annotation.input.mqtt2.config', 'mqtt://localhost:1883,topic=rtmpose,qos=2,queue_max_len=100'),
                       help='Annotation input MQTT2 configuration')
    parser.add_argument('--annotation_out_rtsp', 
                       default=get_config_value(config, 'annotation.output.config', 'rtsp://localhost:8554,topic=outstream,width=1920,height=1080,fps=25'),
                       help='Annotation output RTSP configuration')
    
    # Model configuration
    parser.add_argument('--model_input_size', type=int, nargs=2, 
                       default=get_config_value(config, 'yolox.model_input_size', [640, 640]),
                       help='YOLOX model input size [width, height]')
    parser.add_argument('--rtmpose_model_input_size', type=int, nargs=2, 
                       default=get_config_value(config, 'rtmpose.model_input_size', [192, 256]),
                       help='RTMPose model input size [width, height]')
    
    # Device configuration
    parser.add_argument('--devices', 
                       default=get_config_value(config, 'global.devices', 'cuda'), 
                       help='Computing device (cuda/cpu)')
    
    # Service startup control
    skip_services_default = []
    for service in ['yolox', 'rtmpose', 'bytetrack', 'annotation']:
        if not get_config_value(config, f'{service}.enabled', True):
            skip_services_default.append(service)
    
    parser.add_argument('--skip_services', nargs='*', choices=['yolox', 'rtmpose', 'bytetrack', 'annotation'],
                       default=skip_services_default, help='Skip specified services')
    parser.add_argument('--startup_delay', type=int, 
                       default=get_config_value(config, 'global.startup_delay', 1), 
                       help='Delay between service startups (seconds)')
    
    # Backend configuration
    parser.add_argument('--backend',
                       default=get_config_value(config, 'global.backend', 'onnxruntime'),
                       help='Inference backend')
    
    # Number of workers configuration
    parser.add_argument('--num_workers_per_device', type=int,
                       default=get_config_value(config, 'global.num_workers_per_device', 1),
                       help='Number of workers per device')
    
    # Log level configuration

    parser.add_argument('--log_level',
                       default=get_config_value(config, 'global.log_level', 'INFO'),
                       help='Logging level')

    
    add_service_args(parser)

    
    args = parser.parse_args()
    
    # Store config for reference
    args.config_data = config
    
    return args

async def main():
    """Main function"""
    # Parse arguments

    args = parse_args()

    # Setup logging

    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    
    # Output startup information
    logger.info("=" * 80)
    logger.info("🤖 Unified AI Multi-Person Pose Estimation Service")
    logger.info("=" * 80)
    logger.info(f"Startup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Project root directory: {PROJECT_ROOT}")
    logger.info(f"Computing device: {args.devices}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Skipped services: {args.skip_services if args.skip_services else 'None'}")
    
    # Create service manager
    service_manager = ServiceManager()
    
    # Setup signal handling
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, preparing to shutdown services...")
        asyncio.create_task(service_manager.shutdown_all_services())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start services in order
        services_to_start = [
            ('yolox', service_manager.start_yolox_service),
            ('rtmpose', service_manager.start_rtmpose_service),
            ('bytetrack', service_manager.start_bytetrack_service), 
            ('annotation', service_manager.start_annotation_service),
        ]
        
        startup_success = True
        
        for service_name, start_func in services_to_start:
            if service_name in args.skip_services:
                logger.info(f"⏭️  Skipping {service_name.upper()} service")
                continue
                
            success = await start_func(args)
            if not success:
                logger.error(f"❌ {service_name.upper()} service startup failed, stopping subsequent service startups")
                startup_success = False
                break
            
            if args.startup_delay > 0:
                logger.info(f"⏳ Waiting {args.startup_delay} seconds before starting next service...")
                await asyncio.sleep(args.startup_delay)
        
        if not startup_success:
            logger.error("💥 Some services failed to start, system cannot run normally")
            return
        
        # Start monitoring task
        await service_manager.start_monitoring_task()
        
        logger.info("🎉 All services started successfully! System is running...")
        logger.info("📝 Log level: " + args.log_level)
        logger.info("🔧 Use Ctrl+C to stop services")
        
        # Keep main loop running
        while not service_manager.shutdown_event.is_set():
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("👋 Received interrupt signal, preparing to shutdown...")
    except Exception as e:
        logger.error(f"💥 Unexpected error in main program: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
    finally:
        await service_manager.shutdown_all_services()
        logger.info("🏁 Program has completely exited")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
    except Exception as e:
        print(f"💥 Program startup failed: {e}")
        sys.exit(1) 