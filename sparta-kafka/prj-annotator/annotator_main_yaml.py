#!/usr/bin/env python3
"""
Annotation service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from annotator_worker import AnnotatorWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.rtsp_output_interface import RTSPOutput
from contanos.io.kafka_input_interface import KafkaInput
from contanos.io.multi_input_interface import MultiInputInterface
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotation Visualization Service with CMC services"
    )
    
    add_argument(parser, 'in_rtsp', 'IN_RTSP_URL', 'rtsp://localhost:8554,topic=mystream')
    add_argument(parser, 'in_kafka1', 'IN_KAFKA_URL_1', 'kafka://localhost:9092,topic=bytetrack,qos=2,queue_max_len=100')
    add_argument(parser, 'in_kafka2', 'IN_KAFKA_URL_2', 'kafka://localhost:9092,topic=rtmpose,qos=2,queue_max_len=100')
    add_argument(parser, 'in_kafka3', 'IN_KAFKA_URL_3', 'kafka://localhost:9092,topic=cmc,qos=2,queue_max_len=100')
    add_argument(parser, 'in_kafka4', 'IN_KAFKA_URL_4', 'kafka://localhost:9092,topic=jerseyocr,qos=2,queue_max_len=100')
    add_argument(parser, 'out_rtsp', 'OUT_RTSP_URL', 'rtsp://0.0.0.0:5108,topic=annotated_stream,height=1080,width=1920,bitrate=7000k')
    
    add_service_args(parser)
    add_compute_args(parser)

    return parser.parse_args()

async def main():
    global input_interface
    """Main function to create and start the service."""
    args = parse_args()
    
    
    # Get configuration values (CLI args override YAML)
    in_kafka1 = args.in_kafka1    
    in_kafka2 = args.in_kafka2
    in_kafka3 = args.in_kafka3
    in_kafka4 = args.in_kafka4
    in_rtsp = args.in_rtsp
    out_rtsp = args.out_rtsp
    devices = 'cpu'
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Annotation service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  in_kafka1: {in_kafka1}")
    logger.info(f"  in_kafka2: {in_kafka2}")
    logger.info(f"  in_kafka3: {in_kafka3}")
    logger.info(f"  in_kafka4: {in_kafka4}")
    logger.info(f"  out_rtsp: {out_rtsp}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        in_kafka1_config = parse_config_string(in_kafka1)
        in_kafka2_config = parse_config_string(in_kafka2)
        in_kafka3_config = parse_config_string(in_kafka3)
        in_kafka4_config = parse_config_string(in_kafka4)
        out_rtsp_config = parse_config_string(out_rtsp)

        # Create input interfaces
        input_video_interface = RTSPInput(config=in_rtsp_config)
        input_message_interface1 = KafkaInput(config=in_kafka1_config)
        input_message_interface2 = KafkaInput(config=in_kafka2_config)
        input_message_interface3 = KafkaInput(config=in_kafka3_config)
        input_message_interface4 = KafkaInput(config=in_kafka4_config)

        # Combine multiple input interfaces
        input_interface = MultiInputInterface([input_video_interface, 
                                               input_message_interface1, 
                                               input_message_interface2, 
                                               input_message_interface3, 
                                               input_message_interface4])
        
        # Initialize input interface first
        await input_interface.initialize()
        output_interface = RTSPOutput(config=out_rtsp_config)
        await output_interface.initialize()
        
        # Create model configuration
        model_config = dict()

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=AnnotatorWorker,
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
        
        logger.info("Annotation service started successfully")
        
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting Annotation service: {e}")
        raise
    finally:
        logger.info("Annotation service shutdown complete")

# Debug monitoring function
async def quick_debug():
    while True:
        main_q = input_interface._queue.qsize()
        sync_dict = len(input_interface._data_dict)
        
        output_str = f"Main Q: {main_q}, Sync Dict: {sync_dict}"

        for i, kafka_q in enumerate(input_interface.interfaces):
            if hasattr(kafka_q, 'message_queue'):
                kafka_q_size = kafka_q.message_queue.qsize()
            else:
                kafka_q_size = 'N/A'
            output_str += f" Interface_{i} Q: {kafka_q_size} |"
            
        logging.info(output_str)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main()) 