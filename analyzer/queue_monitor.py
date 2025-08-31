import asyncio
import time
import logging
from typing import Dict, Any

class QueueMonitor:
    """Monitor queue lengths and worker wait times to diagnose bottlenecks."""
    
    def __init__(self, multi_input_interface, workers, log_interval=2.0):
        self.multi_input = multi_input_interface
        self.workers = workers
        self.log_interval = log_interval
        self.worker_wait_times = {}
        self.worker_last_read = {}
        self.is_monitoring = False
        self.monitor_task = None
        
    async def start_monitoring(self):
        """Start background monitoring task."""
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logging.info("Queue monitoring started")
        
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
    async def _monitor_loop(self):
        """Main monitoring loop that logs queue states."""
        while self.is_monitoring:
            try:
                await self._log_queue_status()
                await asyncio.sleep(self.log_interval)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                
    async def _log_queue_status(self):
        """Log current queue lengths and worker states."""
        # MultiInputInterface queue status
        main_queue_size = self.multi_input._queue.qsize()
        data_dict_size = len(self.multi_input._data_dict)
        
        # Individual interface queue sizes
        interface_queue_sizes = []
        for i, interface in enumerate(self.multi_input.interfaces):
            if hasattr(interface, 'queue'):
                interface_queue_sizes.append(f"Interface{i}: {interface.queue.qsize()}")
            elif hasattr(interface, 'message_queue'):
                interface_queue_sizes.append(f"Interface{i}: {interface.message_queue.qsize()}")
        
        # Worker status
        current_time = time.time()
        worker_status = []
        for worker in self.workers:
            worker_id = worker.worker_id
            last_read = self.worker_last_read.get(worker_id, current_time)
            time_since_read = current_time - last_read
            worker_status.append(f"Worker{worker_id}: {time_since_read:.1f}s")
        
        # Log comprehensive status
        logging.info(
            f"QUEUE STATUS - "
            f"Main queue: {main_queue_size}, "
            f"Sync dict: {data_dict_size}, "
            f"Interfaces: [{', '.join(interface_queue_sizes)}], "
            f"Workers: [{', '.join(worker_status)}]"
        )


class TimedMultiInputInterface:
    """Wrapper to add timing measurements to MultiInputInterface."""
    
    def __init__(self, multi_input_interface, queue_monitor):
        self.multi_input = multi_input_interface
        self.monitor = queue_monitor
        
    async def read_data(self, worker_id=None):
        """Timed version of read_data."""
        start_time = time.time()
        
        try:
            result = await self.multi_input.read_data()
            read_time = time.time() - start_time
            
            # Update monitor
            if worker_id is not None:
                self.monitor.worker_last_read[worker_id] = time.time()
                if read_time > 0.1:  # Log slow reads
                    logging.warning(f"Worker {worker_id} slow read: {read_time:.3f}s")
                    
            return result
            
        except Exception as e:
            read_time = time.time() - start_time
            logging.error(f"Worker {worker_id} read failed after {read_time:.3f}s: {e}")
            raise
    
    def __getattr__(self, name):
        """Forward other calls to original interface."""
        return getattr(self.multi_input, name)


# Modified BaseWorker to track timing
class TimedBaseWorker:
    """Modified worker that tracks timing and reports to monitor."""
    
    def __init__(self, original_worker, queue_monitor):
        self.original_worker = original_worker
        self.monitor = queue_monitor
        
    async def run(self):
        """Timed version of worker run loop."""
        worker_id = self.original_worker.worker_id
        logging.info(f"Timed Worker {worker_id} started on {self.original_worker.device}")
        
        while True:
            try:
                # Time the read operation
                read_start = time.time()
                input, metadata = await self.original_worker.input_interface.read_data()
                read_time = time.time() - read_start
                
                # Update monitor
                self.monitor.worker_last_read[worker_id] = time.time()
                if read_time > 0.1:
                    logging.warning(f"Worker {worker_id} read blocked for {read_time:.3f}s")
                
                # Time the prediction
                predict_start = time.time()
                results = self.original_worker._predict(input, metadata)
                predict_time = time.time() - predict_start
                
                # Time the write
                if results is not None:
                    write_start = time.time()
                    output = self.original_worker._format_results(results, metadata)
                    await self.original_worker.output_interface.write_data(output)
                    write_time = time.time() - write_start
                    
                    total_time = read_time + predict_time + write_time
                    logging.info(
                        f"Worker {worker_id} timing - "
                        f"Read: {read_time:.3f}s, "
                        f"Predict: {predict_time:.3f}s, "
                        f"Write: {write_time:.3f}s, "
                        f"Total: {total_time:.3f}s"
                    )
                    
            except asyncio.TimeoutError:
                logging.info(f"Worker {worker_id} timed out, stopping")
                break
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                break
                
        logging.info(f"Timed Worker {worker_id} finished")
    
    def __getattr__(self, name):
        """Forward other calls to original worker."""
        return getattr__(self.original_worker, name)


# Usage function
async def diagnose_performance(processor, multi_input_interface):
    """Run performance diagnosis on the processor."""
    
    # Create monitor
    monitor = QueueMonitor(multi_input_interface, processor.workers)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    try:
        # Run processor for a short time
        logging.info("Starting performance diagnosis...")
        await processor.start()
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop and analyze
        await processor.stop()
        
    finally:
        await monitor.stop_monitoring()
        
    logging.info("Performance diagnosis completed")


# Quick debug function to add to your existing code
def add_queue_monitoring(multi_input_interface, workers):
    """Quick function to add monitoring to existing setup."""
    
    async def log_status():
        while True:
            main_q = multi_input_interface._queue.qsize()
            sync_dict = len(multi_input_interface._data_dict)
            
            interface_qs = []
            for i, iface in enumerate(multi_input_interface.interfaces):
                if hasattr(iface, 'queue'):
                    interface_qs.append(f"{i}:{iface.queue.qsize()}")
                elif hasattr(iface, 'message_queue'):
                    interface_qs.append(f"{i}:{iface.message_queue.qsize()}")
            
            print(f"Main Q: {main_q}, Sync Dict: {sync_dict}, Interface Qs: {interface_qs}")
            await asyncio.sleep(1)
    
    return asyncio.create_task(log_status())