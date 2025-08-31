import logging
from typing import Dict, List, Optional, Callable, Any, Type, Tuple

from contanos.base_worker import BaseWorker
from contanos.base_processor import BaseProcessor


def create_a_processor(
    worker_class: Type[BaseWorker],
    model_config,
    devices: List[str],
    input_interface, output_interface,
    num_workers_per_device: int = 1
) -> Tuple[List[BaseWorker], BaseProcessor]:
    """
    Helper function to create and initialize workers across multiple devices.
    
    Args:
        worker_class: The BaseWorker class to instantiate
        model_def, model_config # Model definition and configuration parameters
        devices: List of device strings (e.g., ['cuda:0', 'cuda:1', 'cpu'])
        input_interface: Shared input interface for all workers
        output_interface: Shared output interface for all workers
        num_workers_per_device: Number of workers to create per device
        
    Returns:
        Tuple of (workers list, processor instance)
    """
    workers = []
    worker_id = 0
    
    # Create workers for each device
    for device in devices:
        for _ in range(num_workers_per_device):
            # Create worker
            worker = worker_class(
                worker_id=worker_id,
                device=device,
                model_config=model_config,
                input_interface=input_interface,
                output_interface=output_interface
            )
            
            workers.append(worker)
            worker_id += 1
            
    logging.info(f"Created {len(workers)} workers across {len(devices)} devices")
    
    # Create processor with the workers
    processor = BaseProcessor(
        input_interface=input_interface,
        output_interface=output_interface,
        workers=workers
    )
    
    return workers, processor
