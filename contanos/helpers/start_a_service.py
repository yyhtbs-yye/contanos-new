import logging
from typing import Optional, Dict, Any
from contanos.base_service import BaseService

async def start_a_service(
    processor,
    service_config: Optional[Dict[str, Any]] = None,
    run_until_complete: bool = True,
    daemon_mode: bool = False
) -> BaseService:
    """
    Start a BaseService with the given processor and configuration.
    
    Args:
        processor: The processor instance to monitor
        service_config: Optional configuration dict with keys:
            - health_check_interval: Seconds between health checks (default: 5.0)
            - max_restart_attempts: Maximum restart attempts per worker (default: 3)
            - restart_cooldown: Seconds to wait before restarting (default: 30.0)
        run_until_complete: If True, wait for processor to complete
        daemon_mode: If True, return immediately after starting (service runs in background)
        
    Returns:
        The started BaseService instance
    """
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
    service = BaseService(
        processor=processor,
        health_check_interval=config['health_check_interval'],
        max_restart_attempts=config['max_restart_attempts'],
        restart_cooldown=config['restart_cooldown']
    )
    
    logging.info(f"Starting service with config: {config}")
    
    if daemon_mode:
        # Start in background and return immediately
        async with processor, service:
            logging.info("Service started in daemon mode")
            # Return the service for external management
            return service
    
    elif run_until_complete:
        # Run until processor completes
        async with processor, service:
            await processor.run_until_complete()
            logging.info("Service and processor completed")
        return service
    
    else:
        # Just start and return the service
        await processor.start()
        await service.start_monitoring()
        logging.info("Service started")
        return service


# Convenience wrapper for quick starts
async def quick_start_service(processor) -> BaseService:
    """
    Quick start a service with default settings and run until complete.
    
    Args:
        processor: The processor to monitor
        
    Returns:
        The BaseService instance
    """
    return await start_a_service(processor, run_until_complete=True)