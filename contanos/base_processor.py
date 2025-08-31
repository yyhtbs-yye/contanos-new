import asyncio
from typing import Any, List
import logging

class BaseProcessor:
    def __init__(self, 
                 input_interface, 
                 output_interface,
                 workers: Any):
        """
        Initialize the processor with N workers.
        
        Args:
            input_interface: Interface for reading input data
            output_interface: Interface for writing output data
            workers: Precreated worker instances
        """
        self.input_interface = input_interface
        self.output_interface = output_interface
        
        # Track running tasks
        self.worker_tasks: List[asyncio.Task] = []
        self.is_running = False
        self.workers = workers

    async def start(self):
        """Start all workers."""
        if self.is_running:
            logging.warning("Processor is already running")
            return
        
        logging.info(f"Starting processor with {len(self.workers)} workers")
        self.is_running = True
        
        # Create and start tasks for all workers
        self.worker_tasks = []
        for worker in self.workers:
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)
        
        logging.info("All workers started")

    async def stop(self):
        """Stop all workers gracefully."""
        if not self.is_running:
            logging.warning("Processor is not running")
            return
        
        logging.info("Stopping processor...")
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete or be cancelled
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks = []
        logging.info("All workers stopped")

    async def run_until_complete(self):
        """Run all workers until they complete naturally (e.g., input exhausted)."""
        await self.start()
        
        try:
            # Wait for all worker tasks to complete
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        except Exception as e:
            logging.error(f"Error during processing: {e}")
        finally:
            self.is_running = False
            logging.info("Processor completed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def get_worker_status(self) -> dict:
        """Get status information about all workers."""
        status = {
            'num_workers': len(self.workers),
            'is_running': self.is_running,
            'workers': []
        }
        
        for i, worker in enumerate(self.workers):
            worker_info = {
                'worker_id': worker.worker_id,
                'device': worker.device,
                'task_done': self.worker_tasks[i].done() if i < len(self.worker_tasks) else None
            }
            status['workers'].append(worker_info)
        
        return status