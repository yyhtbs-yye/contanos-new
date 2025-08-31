import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from contanos.base_processor import BaseProcessor

class MultiProcessorService:
    """A service that monitors multiple processors and automatically restarts failed workers."""

    def __init__(
        self,
        processors: List[BaseProcessor],
        health_check_interval: float = 5.0,
        max_restart_attempts: int = 3,
        restart_cooldown: float = 30.0,
    ):
        """
        Initialize the self-healing service.

        Args:
            processors: List of processor instances to monitor.
            health_check_interval: Seconds between health checks.
            max_restart_attempts: Maximum restart attempts per worker (per processor).
            restart_cooldown: Seconds to wait before restarting a worker.
        """
        if not isinstance(processors, list) or not processors:
            raise ValueError("processors must be a non-empty list of BaseProcessor instances")

        self.processors: List[BaseProcessor] = processors
        self.health_check_interval = health_check_interval
        self.max_restart_attempts = max_restart_attempts
        self.restart_cooldown = restart_cooldown

        # Track restarts per (processor index, worker id)
        self.restart_counts: Dict[Tuple[int, int], int] = {}
        self.last_restart_times: Dict[Tuple[int, int], datetime] = {}

        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    async def start_monitoring(self):
        """Start monitoring all processors and their workers."""
        if self.is_monitoring:
            logging.warning("Self-healing service is already monitoring")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logging.info("Self-healing service started monitoring %d processors", len(self.processors))

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
        """Check worker health and restart failed workers across all processors."""
        for p_idx, processor in enumerate(self.processors):
            # Skip processors that aren't running
            if not getattr(processor, "is_running", False):
                continue

            try:
                status = processor.get_worker_status()
            except Exception as e:
                logging.error("%s get_worker_status() failed: %s", self._proc_label(p_idx), e)
                continue

            workers = status.get("workers", [])
            for i, worker_info in enumerate(workers):
                worker_id = worker_info.get("worker_id", i)

                # Ensure task index exists
                if i >= len(getattr(processor, "worker_tasks", [])):
                    continue

                task = processor.worker_tasks[i]

                if task.done():
                    # Worker has stopped
                    try:
                        exception = task.exception()
                        if exception:
                            logging.error(
                                "%s worker %s failed with exception: %s",
                                self._proc_label(p_idx),
                                worker_id,
                                exception,
                            )
                    except asyncio.CancelledError:
                        logging.info(
                            "%s worker %s was cancelled",
                            self._proc_label(p_idx),
                            worker_id,
                        )
                        continue

                    key = (p_idx, int(worker_id))
                    if self._should_restart_worker(key):
                        await self._restart_worker(p_idx, i, int(worker_id))

    def _should_restart_worker(self, key: Tuple[int, int]) -> bool:
        """Determine if a worker should be restarted, keyed by (processor_index, worker_id)."""
        restart_count = self.restart_counts.get(key, 0)
        if restart_count >= self.max_restart_attempts:
            logging.error(
                "Processor %d worker %d has exceeded max restart attempts (%d)",
                key[0],
                key[1],
                self.max_restart_attempts,
            )
            return False

        last_restart = self.last_restart_times.get(key)
        if last_restart:
            time_since_restart = datetime.now() - last_restart
            if time_since_restart.total_seconds() < self.restart_cooldown:
                logging.info(
                    "Processor %d worker %d is in cooldown period",
                    key[0],
                    key[1],
                )
                return False

        return True

    async def _restart_worker(self, proc_index: int, task_index: int, worker_id: int):
        """Restart a specific worker on a specific processor."""
        processor = self.processors[proc_index]
        key = (proc_index, worker_id)
        label = self._proc_label(proc_index)

        logging.info("Restarting %s worker %s", label, worker_id)

        # Update restart tracking
        self.restart_counts[key] = self.restart_counts.get(key, 0) + 1
        self.last_restart_times[key] = datetime.now()

        # Get the worker instance and create a new task
        try:
            worker = processor.workers[task_index]
        except Exception as e:
            logging.error("Failed to access %s worker %s: %s", label, worker_id, e)
            return

        try:
            new_task = asyncio.create_task(worker.run())
            processor.worker_tasks[task_index] = new_task
            logging.info(
                "%s worker %s restarted (attempt %d)",
                label,
                worker_id,
                self.restart_counts[key],
            )
        except Exception as e:
            logging.error("Failed to restart %s worker %s: %s", label, worker_id, e)

    def reset_worker_stats(self, proc_index: int, worker_id: int):
        """Reset restart statistics for a worker on a specific processor."""
        key = (proc_index, int(worker_id))
        self.restart_counts.pop(key, None)
        self.last_restart_times.pop(key, None)

    def reset_all_stats(self):
        """Reset restart statistics for all workers across all processors."""
        self.restart_counts.clear()
        self.last_restart_times.clear()

    def _proc_label(self, proc_index: int) -> str:
        """Human-friendly label for logs."""
        proc = self.processors[proc_index]
        name = getattr(proc, "name", None)
        if isinstance(name, str) and name:
            return f"processor[{proc_index}]({name})"
        return f"processor[{proc_index}]({proc.__class__.__name__})"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()
