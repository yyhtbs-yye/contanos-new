#!/usr/bin/env python3
"""
JerseyOCR service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from jerseyocr_worker import JerseyOCRWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.mqtt_output_interface import MQTTOutput
from contanos.io.mqtt_input_interface import MQTTInput
from contanos.io.multi_input_interface import MultiInputInterface
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pytorch Mobilenet Jersey OCR GPU Service",
    )

    add_argument(parser, 'in_rtsp', 'IN_RTSP_URL', None) # 'rtsp://localhost:8554,topic=mystream'
    add_argument(parser, 'in_mqtt', 'IN_MQTT_URL', None) # 'mqtt://localhost:1883,topic=bytetrack,qos=2,queue_max_len=100'
    add_argument(parser, 'out_mqtt', 'OUT_MQTT_URL', None) # 'mqtt://localhost:1883,topic=jerseyocr,qos=2,queue_max_len=100'
    add_argument(parser, 'devices', 'DEVICES', 'cuda:1')
    add_argument(parser, 'model_input_size', 'MODEL_INPUT_SIZE', '192,256')
    add_argument(parser, 'use_small', 'USE_SMALL', True)

    add_service_args(parser)
    add_compute_args(parser)

    return parser.parse_args()

async def main():
    global input_interface
    """Main function to create and start the service."""
    args = parse_args()
    
    # Get configuration values (CLI args override YAML)
    if args.in_rtsp:
        in_rtsp = args.in_rtsp
    else:
        # throw error if not provided
        raise ValueError("IN_RTSP_URL must be provided as CLI argument.")
    
    if args.in_mqtt:
        in_mqtt = args.in_mqtt
    else:
        # throw error if not provided
        raise ValueError("IN_MQTT_URL must be provided as CLI argument.")
    
    out_mqtt = args.out_mqtt
    devices = args.devices
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    
    # Model input size from YAML or CLI
    if args.model_input_size:
        model_input_size = args.model_input_size.split(',') if isinstance(args.model_input_size, str) else args.model_input_size
        model_input_size = [int(size) for size in model_input_size]
    else:
        # throw error if not provided
        raise ValueError("MODEL_INPUT_SIZE must be provided as CLI argument.")
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting JerseyOCR service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  in_mqtt: {in_mqtt}")
    logger.info(f"  out_mqtt: {out_mqtt}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  model_input_size: {model_input_size}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        in_mqtt_config = parse_config_string(in_mqtt)
        out_mqtt_config = parse_config_string(out_mqtt)

        # Create input/output interfaces
        input_video_interface = RTSPInput(config=in_rtsp_config)
        input_message_interface = MQTTInput(config=in_mqtt_config)
        input_interface = MultiInputInterface([input_video_interface, input_message_interface])
        output_interface = MQTTOutput(config=out_mqtt_config)
        
        await input_interface.initialize()
        await output_interface.initialize()

        if isinstance(args.use_small, str):
            use_small = args.use_small.lower() in ('true', '1', 'yes')
        elif isinstance(args.use_small, bool):
            use_small = args.use_small
        else:
            raise ValueError("USE_SMALL must be a boolean or string representing a boolean.")
        
        # Create model configuration
        model_config = dict(
            weights_path='jersey_ocr_best.pth',
            model_input_size=model_input_size,
            use_small=use_small,
        )

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=JerseyOCRWorker,
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
        
        logger.info("JerseyOCR service started successfully")
        
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting JerseyOCR service: {e}")
        raise
    finally:
        logger.info("JerseyOCR service shutdown complete")

# Debug monitoring function
async def quick_debug():
    while True:
        main_q = input_interface._queue.qsize()
        sync_dict = len(input_interface._data_dict)
        
        rtsp_q = input_interface.interfaces[0].queue.qsize()  # RTSP queue
        mqtt_q = input_interface.interfaces[1].message_queue.qsize()  # MQTT queue
        
        logging.info(f"Main Q: {main_q}, Sync Dict: {sync_dict}, RTSP Q: {rtsp_q}, MQTT Q: {mqtt_q}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main()) 