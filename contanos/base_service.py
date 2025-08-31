
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from contanos.base_processor import BaseProcessor

class BaseService:
    """A service that monitors and automatically restarts failed workers."""
    
    def __init__(self, processor: BaseProcessor, 
                 health_check_interval: float = 5.0,
                 max_restart_attempts: int = 3,
                 restart_cooldown: float = 30.0):
        """
        Initialize the self-healing service.
        
        Args:
            processor: The processor instance to monitor
            health_check_interval: Seconds between health checks
            max_restart_attempts: Maximum restart attempts per worker
            restart_cooldown: Seconds to wait before restarting a worker
        """
        self.processor = processor
        self.health_check_interval = health_check_interval
        self.max_restart_attempts = max_restart_attempts
        self.restart_cooldown = restart_cooldown
        
        self.restart_counts: Dict[int, int] = {}
        self.last_restart_times: Dict[int, datetime] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """Start monitoring the processor and its workers."""
        if self.is_monitoring:
            logging.warning("Self-healing service is already monitoring")
            return
            
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logging.info("Self-healing service started monitoring")
        
    async def stop_monitoring(self):
        """Stop monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logging.info("Self-healing service stopped monitoring")
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._check_and_heal_workers()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                
    async def _check_and_heal_workers(self):
        """Check worker health and restart failed workers."""
        if not self.processor.is_running:
            return
            
        status = self.processor.get_worker_status()
        
        for i, worker_info in enumerate(status['workers']):
            worker_id = worker_info['worker_id']
            
            # Check if worker task exists and is done (failed)
            if i < len(self.processor.worker_tasks):
                task = self.processor.worker_tasks[i]
                
                if task.done():
                    # Worker has stopped
                    try:
                        # Check if it raised an exception
                        exception = task.exception()
                        if exception:
                            logging.error(f"Worker {worker_id} failed with exception: {exception}")
                    except asyncio.CancelledError:
                        logging.info(f"Worker {worker_id} was cancelled")
                        continue
                        
                    # Check if we should restart this worker
                    if self._should_restart_worker(worker_id):
                        await self._restart_worker(i, worker_id)
                        
    def _should_restart_worker(self, worker_id: int) -> bool:
        """Determine if a worker should be restarted."""
        # Check restart count
        restart_count = self.restart_counts.get(worker_id, 0)
        if restart_count >= self.max_restart_attempts:
            logging.error(f"Worker {worker_id} has exceeded max restart attempts ({self.max_restart_attempts})")
            return False
            
        # Check cooldown period
        last_restart = self.last_restart_times.get(worker_id)
        if last_restart:
            time_since_restart = datetime.now() - last_restart
            if time_since_restart.total_seconds() < self.restart_cooldown:
                logging.info(f"Worker {worker_id} is in cooldown period")
                return False
                
        return True
        
    async def _restart_worker(self, task_index: int, worker_id: int):
        """Restart a specific worker."""
        logging.info(f"Restarting worker {worker_id}")
        
        # Update restart tracking
        self.restart_counts[worker_id] = self.restart_counts.get(worker_id, 0) + 1
        self.last_restart_times[worker_id] = datetime.now()
        
        # Get the worker instance
        worker = self.processor.workers[task_index]
        
        # Create new task for the worker
        new_task = asyncio.create_task(worker.run())
        self.processor.worker_tasks[task_index] = new_task
        
        logging.info(f"Worker {worker_id} restarted (attempt {self.restart_counts[worker_id]})")
        
    def reset_worker_stats(self, worker_id: int):
        """Reset restart statistics for a worker."""
        self.restart_counts.pop(worker_id, None)
        self.last_restart_times.pop(worker_id, None)
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()
