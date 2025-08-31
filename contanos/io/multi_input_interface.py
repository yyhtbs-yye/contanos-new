from typing import Any, Dict, Tuple, List
import asyncio
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class MultiInputInterface:
    """Wrapper for multiple input interfaces with synchronized multi-entry queue."""
    
    def __init__(self, interfaces: List[Any]):
        self.interfaces = interfaces
        self.is_running = False
        self._executor = ThreadPoolExecutor(max_workers=len(interfaces))
        self._producer_tasks = []
        self._data_dict = defaultdict(lambda: {})  # {frame_id_str: {interface_idx: (data, metadata)}}
        self._queue = asyncio.Queue(maxsize=1000)  # Stores completed {interface_idx: (data, metadata)}
        self._lock = asyncio.Lock()  # Protects _data_dict and _queue
        self._num_interfaces = len(interfaces)
    
    async def initialize(self) -> bool:
        """Initialize all input interfaces and start producers."""
        try:
            # Initialize all interfaces concurrently
            results = await asyncio.gather(
                *[interface.initialize() for interface in self.interfaces],
                return_exceptions=True
            )
            if not all(results):
                logging.error("One or more interfaces failed to initialize")
                return False
            
            self.is_running = True
            # Start producer tasks for each interface
            for idx, interface in enumerate(self.interfaces):
                task = asyncio.create_task(self._interface_producer(idx, interface))
                self._producer_tasks.append(task)
            
            logging.info("MultiInputInterface initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize MultiInputInterface: {e}")
            return False
    
    async def _interface_producer(self, interface_idx, interface):
        """Background task to fetch data from an interface and update dictionary."""
        while self.is_running:
            try:
                data, metadata = await interface.read_data()
                frame_id_str = metadata.get('frame_id_str')
                if frame_id_str is None:
                    logging.warning(f"Interface {interface_idx} provided no frame_id_str, skipping")
                    continue
                
                async with self._lock:
                    # Add data to dictionary
                    self._data_dict[frame_id_str][interface_idx] = (data, metadata)
                    # logging.info(f"Interface {interface_idx} added data for frame_id_str: {frame_id_str}")
                    
                    # Check if all interfaces have contributed
                    if len(self._data_dict[frame_id_str]) == self._num_interfaces:
                        # Push completed dictionary to queue
                        await self._queue.put(self._data_dict[frame_id_str])
                        # Clean up dictionary
                        del self._data_dict[frame_id_str]
            
            except Exception as e:
                logging.error(f"Interface {interface_idx} error: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error
    
    async def read_data(self) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Read synchronized data for a frame_id_str from the queue."""
        if not self.is_running:
            raise Exception("MultiInputInterface not initialized or stopped")
        
        try:
            # Get completed dictionary from queue
            frame_data = await self._queue.get()
            # Sort by interface_idx for consistent order
            # print(f"the current frame_id_str is: {frame_data[0][1]['frame_id_str']}")
            data_list = []
            metadata_list = []
            for idx in sorted(frame_data.keys()):
                data, metadata = frame_data[idx]
                data_list.append(data)
                metadata_list.append(metadata)
            self._queue.task_done()
            return data_list, metadata_list
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise Exception(f"Failed to read synchronized data: {e}")
    
    async def cleanup(self):
        """Clean up all interfaces and resources."""
        self.is_running = False
        # Cancel producer tasks
        for task in self._producer_tasks:
            task.cancel()
        try:
            await asyncio.gather(*self._producer_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        # Clean up interfaces
        await asyncio.gather(
            *[interface.cleanup() for interface in self.interfaces],
            return_exceptions=True
        )
        # Clear dictionary and queue
        async with self._lock:
            self._data_dict.clear()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
        # Shut down executor
        self._executor.shutdown(wait=True)
        logging.info("MultiInputInterface cleaned up")