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
from deblur_worker import DeblurWorker
from contanos.io.rtsp_sorted_input_interface import RTSPInput
from contanos.io.rtsp_output_interface import RTSPOutput
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string

def parse_args():
    parser = argparse.ArgumentParser(
        description="Yuhang's Lightweight Model for Video Motion Deblur"
    )

    add_argument(parser, 'in_rtsp', 'IN_RTSP_URL', 'rtsp://localhost:8554,topic=rawstream')
    add_argument(parser, 'out_rtsp', 'OUT_RTSP_URL', 'rtsp://localhost:8554,topic=xstream,height=540,width=960,bitrate=7000k')
    add_argument(parser, 'devices', 'DEVICES', 'cuda:8')

    add_service_args(parser)
    add_compute_args(parser)
    
    return parser.parse_args()

async def main():
    """Main function to create and start the service."""
    args = parse_args()
    
    # Get configuration values (CLI args override YAML)
    in_rtsp = args.in_rtsp
    out_rtsp = args.out_rtsp
    devices = args.devices
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Deblur service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  out_rtsp: {out_rtsp}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        out_rtsp_config = parse_config_string(out_rtsp)
        
        # Create input/output interfaces
        input_interface = RTSPInput(config=in_rtsp_config)
        output_interface = RTSPOutput(config=out_rtsp_config)
        
        await input_interface.initialize()
        await output_interface.initialize()

        # Create model configuration
        model_config = dict(
            weights_path='/home/admyyh/python_workspaces/contanos/projects/prj-deblur-gpu/lvr_arch_iter_30000.pth',
        )

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]
        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=DeblurWorker,
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
        
        logger.info("Deblur service started successfully")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting Deblur service: {e}")
        raise
    finally:
        logger.info("Deblur service shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 