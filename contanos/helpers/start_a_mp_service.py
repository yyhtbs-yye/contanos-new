from contanos.multi_processor_service import MultiProcessorService
from contanos.base_processor import BaseProcessor

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any, List, Union

async def start_a_mp_service(
    processors: Union[BaseProcessor, List[BaseProcessor]],
    service_config: Optional[Dict[str, Any]] = None,
    run_until_complete: bool = True,
    daemon_mode: bool = False
) -> MultiProcessorService:
    """
    Start a MultiProcessorService with the given processor(s) and configuration.

    Args:
        processors: A single processor or a list of processors to monitor.
        service_config: Optional configuration dict with keys:
            - health_check_interval: Seconds between health checks (default: 5.0)
            - max_restart_attempts: Maximum restart attempts per worker (default: 3)
            - restart_cooldown: Seconds to wait before restarting (default: 30.0)
        run_until_complete: If True, wait for all processors to complete.
        daemon_mode: If True, return immediately after starting (service runs in background).

    Returns:
        The started MultiProcessorService instance.
    """
    # Normalize processors to a non-empty list
    if isinstance(processors, BaseProcessor):
        proc_list: List[BaseProcessor] = [processors]
    else:
        proc_list = list(processors or [])
    if not proc_list:
        raise ValueError("processors must be a non-empty BaseProcessor or list of BaseProcessor instances")

    # Default configuration
    config = {
        'health_check_interval': 5.0,
        'max_restart_attempts': 3,
        'restart_cooldown': 30.0
    }

    # Update with user config if provided
    if service_config:
        config.update(service_config)

    # Create the service
    service = MultiProcessorService(
        processors=proc_list,
        health_check_interval=config['health_check_interval'],
        max_restart_attempts=config['max_restart_attempts'],
        restart_cooldown=config['restart_cooldown']
    )

    logging.info(f"Starting service for {len(proc_list)} processor(s) with config: {config}")

    if daemon_mode:
        # Start in background and return immediately (no context manager here)
        for p in proc_list:
            await p.start()
        await service.start_monitoring()
        logging.info("Service started in daemon mode")
        return service

    if run_until_complete:
        # Run until all processors complete; ensure clean shutdown via context managers
        async with AsyncExitStack() as stack:
            for p in proc_list:
                await stack.enter_async_context(p)
            await stack.enter_async_context(service)
            await asyncio.gather(*(p.run_until_complete() for p in proc_list))
            logging.info("Service and all processors completed")
        return service

    # Just start and return the service (runs in background; caller manages lifecycle)
    for p in proc_list:
        await p.start()
    await service.start_monitoring()
    logging.info("Service started")
    return service


# Convenience wrapper for quick starts
async def quick_start_service(processors: Union[BaseProcessor, List[BaseProcessor]]) -> MultiProcessorService:
    """
    Quick start a service with default settings and run until complete.

    Args:
        processors: The processor or list of processors to monitor.

    Returns:
        The MultiProcessorService instance.
    """
    return await start_a_mp_service(processors, run_until_complete=True)
