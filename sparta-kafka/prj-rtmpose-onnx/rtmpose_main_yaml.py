#!/usr/bin/env python3
"""
RTMPose service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from rtmpose_worker import RTMPoseWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.kafka_output_interface import KafkaOutput
from contanos.io.kafka_input_interface import KafkaInput
from contanos.io.multi_input_interface import MultiInputInterface
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenMMPose RTMPose for Pose Estimation"
    )
    
    add_argument(parser, 'in_rtsp', 'IN_RTSP_URL', 'rtsp://localhost:8554,topic=mystream')
    add_argument(parser, 'in_kafka', 'IN_KAFKA_URL', 'kafka://localhost:9092,topic=yolox,group_id=yolox_group,queue_max_len=100')
    add_argument(parser, 'out_kafka', 'OUT_KAFKA_URL', 'kafka://localhost:9092,topic=rtmpose,group_id=rtmpose_group,queue_max_len=100')
    add_argument(parser, 'devices', 'DEVICES', 'cuda:1,cuda:2')
    add_argument(parser, 'model_input_size', 'MODEL_INPUT_SIZE', '192,256')
    add_argument(parser, 'model_url', 'MODEL_URL', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip')

    add_service_args(parser)
    add_compute_args(parser)

    return parser.parse_args()

async def main():
    global input_interface
    """Main function to create and start the service."""
    args = parse_args()
    
    # Get configuration values (CLI args override YAML)
    in_rtsp = args.in_rtsp
    in_kafka = args.in_kafka
    out_kafka = args.out_kafka
    devices = args.devices
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    backend = args.backend
    
    model_input_size = args.model_input_size.split(',') if isinstance(args.model_input_size, str) else args.model_input_size
    model_input_size = [int(size) for size in model_input_size]
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RTMPose service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  in_kafka: {in_kafka}")
    logger.info(f"  out_kafka: {out_kafka}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  model_input_size: {model_input_size}")
    logger.info(f"  backend: {backend}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        in_kafka_config = parse_config_string(in_kafka)
        out_kafka_config = parse_config_string(out_kafka)

        # Create input/output interfaces
        input_video_interface = RTSPInput(config=in_rtsp_config)
        input_message_interface = KafkaInput(config=in_kafka_config)
        input_interface = MultiInputInterface([input_video_interface, input_message_interface])
        output_interface = KafkaOutput(config=out_kafka_config)
        
        await input_interface.initialize()
        await output_interface.initialize()
        
        # Create model configuration
        model_config = dict(
            onnx_model=args.model_url if hasattr(args, 'model_url') and args.model_url is not None else \
                'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
            model_input_size=model_input_size,
            backend=backend,
        )

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=RTMPoseWorker,
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
        
        logger.info("RTMPose service started successfully")
        
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting RTMPose service: {e}")
        raise
    finally:
        logger.info("RTMPose service shutdown complete")

# Debug monitoring function
async def quick_debug():
    while True:
        main_q = input_interface._queue.qsize()
        sync_dict = len(input_interface._data_dict)
        
        rtsp_q = input_interface.interfaces[0].queue.qsize()  # RTSP queue
        kafka_q = input_interface.interfaces[1].message_queue.qsize()  # KAFKA queue
        
        logging.info(f"Main Q: {main_q}, Sync Dict: {sync_dict}, RTSP Q: {rtsp_q}, KAFKA Q: {kafka_q}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main()) 