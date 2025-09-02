#!/usr/bin/env python3
"""
YOLOX service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from yolox_worker import YOLOXWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.mqtt_output_interface import MQTTOutput
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenMMPose YOLOX for Bounding Box Detection"
    )

    add_argument(parser, 'in_rtsp', 'IN_RTSP_URL', 'rtsp://localhost:8554,topic=mystream')
    add_argument(parser, 'out_mqtt', 'OUT_MQTT_URL', 'mqtt://localhost:1883,topic=yolox,qos=2,queue_max_len=50')
    add_argument(parser, 'devices', 'DEVICES', 'cuda:3')
    add_argument(parser, 'model_input_size', 'MODEL_INPUT_SIZE', '640,640')
    add_argument(parser, 'model_url', 'MODEL_URL', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip')

    add_service_args(parser)
    add_compute_args(parser)
    
    return parser.parse_args()

async def main():
    """Main function to create and start the service."""
    args = parse_args()
    
    # Get configuration values (CLI args override YAML)
    in_rtsp = args.in_rtsp
    out_mqtt = args.out_mqtt
    devices = args.devices
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    backend = args.backend
    
    # Model input size from YAML or CLI
    model_input_size = args.model_input_size.split(',') if isinstance(args.model_input_size, str) else args.model_input_size
    model_input_size = [int(size) for size in model_input_size]
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting YOLOX service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  out_mqtt: {out_mqtt}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  model_input_size: {model_input_size}")
    logger.info(f"  backend: {backend}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        out_mqtt_config = parse_config_string(out_mqtt)
        
        # Create input/output interfaces
        input_interface = RTSPInput(config=in_rtsp_config)
        output_interface = MQTTOutput(config=out_mqtt_config)
        
        await input_interface.initialize()
        await output_interface.initialize()

        # Create model configuration
        model_config = dict(
            onnx_model=args.model_url if hasattr(args, 'model_url') and args.model_url is not None else \
                'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
            model_input_size=model_input_size,
            backend=backend,
        )

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]
        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=YOLOXWorker,
            model_config=model_config,
            devices=devices,
            input_interface=input_interface,
            output_interface=output_interface,
            num_workers_per_device=args.num_workers_per_device,
        )
        
        # Start the service
        service = await start_a_service(
            processor=processor,
            run_until_complete=args.run_until_complete,
            daemon_mode=False,
        )
        
        logger.info("YOLOX service started successfully")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting YOLOX service: {e}")
        raise
    finally:
        logger.info("YOLOX service shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 