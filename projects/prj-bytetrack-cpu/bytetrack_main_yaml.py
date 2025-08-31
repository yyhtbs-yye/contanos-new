#!/usr/bin/env python3
"""
ByteTrack service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from bytetrack_worker import ByteTrackWorker
from contanos.io.mqtt_sorted_input_interface import MQTTSortedInput
from contanos.io.mqtt_output_interface import MQTTOutput
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="BoxMOTLite ByteTrack Detection for Object Tracking"
    )
    
    # Optional overrides (these will override YAML config if provided)
    add_argument(parser, 'in_mqtt', 'IN_MQTT_URL', None)
    add_argument(parser, 'out_mqtt', 'OUT_MQTT_URL', None)
    add_argument(parser, 'devices', 'DEVICES', None)

    add_service_args(parser)
    add_compute_args(parser)

    return parser.parse_args()

async def main():
    global input_interface
    """Main function to create and start the service."""
    args = parse_args()
    
    # Load YAML configuration
    
    # Get configuration values (CLI args override YAML)
    in_mqtt = args.in_mqtt
    out_mqtt = args.out_mqtt
    devices = args.devices
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ByteTrack service with configuration:")
    logger.info(f"  in_mqtt: {in_mqtt}")
    logger.info(f"  out_mqtt: {out_mqtt}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_mqtt_config = parse_config_string(in_mqtt)
        out_mqtt_config = parse_config_string(out_mqtt)

        # Create input/output interfaces
        input_interface = MQTTSortedInput(config=in_mqtt_config)
        output_interface = MQTTOutput(config=out_mqtt_config)
        
        await input_interface.initialize()
        await output_interface.initialize()

        # Create model configuration with values from YAML
        model_config = dict(
            min_conf=0.1,
            track_thresh=0.45,
            match_thresh=0.8,
            track_buffer=25,
            frame_rate=30,
            per_class=False,
        )

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = ['cpu']  # ByteTrack typically runs on CPU

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=ByteTrackWorker,
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
        
        logger.info("ByteTrack service started successfully")
        
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting ByteTrack service: {e}")
        raise
    finally:
        logger.info("ByteTrack service shutdown complete")

# Debug monitoring function
async def quick_debug():
    while True:
        message_q = input_interface.message_queue.qsize()  # MQTT queue
        ordered_q = input_interface.ordered_queue.qsize()  # Ordered queue
        
        logging.info(f"MESSAGE Q: {message_q}, ORDERED Q: {ordered_q}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main()) 