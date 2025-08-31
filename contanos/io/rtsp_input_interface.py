from typing import Any, Dict, Tuple
import logging
import av
import time
import asyncio
from abc import ABC

from concurrent.futures import ThreadPoolExecutor

def _safe_next_packet(frame_generator):
    """Safely get next packet, converting StopIteration to None."""
    try:
        return next(frame_generator)
    except StopIteration:
        return None


class RTSPInput(ABC):
    """RTSP stream input implementation using asyncio.Queue."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.addr = config['addr']
        self.topic = config['topic']
        self.container = None
        self.video_stream = None
        self.frame_generator = None
        self.queue = asyncio.Queue(maxsize=100)  # Buffer up to 100 frames
        self.is_running = False
        self._producer_task = None
        self._executor = ThreadPoolExecutor(max_workers=1)  # For blocking AV operations
        self._retry_count = 0
    
    async def initialize(self) -> bool:
        """Initialize RTSP connection and start frame producer."""
        try:
            logging.info(f"Connecting to RTSP stream: {self.addr} on {self.topic}")
            # Run blocking AV initialization in executor
            loop = asyncio.get_event_loop()
            self.container = await loop.run_in_executor(self._executor, av.open, self.addr+'/'+self.topic)
            self.video_stream = next(s for s in self.container.streams if s.type == 'video')
            self.frame_generator = self.container.demux(self.video_stream)
            self.is_running = True
            # Start frame producer task
            self._producer_task = asyncio.create_task(self._frame_producer())
            self._retry_count = 0
            logging.info("RTSP connection established")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize RTSP input: {e}")
            return False
    
    async def _frame_producer(self):
        """Background task to fetch frames and push to queue."""
        loop = asyncio.get_event_loop()
        while self.is_running:
            try:
                # Run blocking packet fetching in executor with safe wrapper
                packet = await loop.run_in_executor(self._executor, _safe_next_packet, self.frame_generator)
                
                # Check if stream ended
                if packet is None:
                    logging.warning("RTSP stream ended")
                    self.is_running = False
                    break
                
                for frame in packet.decode():
                    # Extract frame ID from SEI data if present
                    frame_id_str = None
                    if frame.side_data:
                        for side_data in frame.side_data:
                            if side_data.type.name == "SEI_UNREGISTERED":
                                frame_id_str = bytes(side_data).decode('ascii', 'ignore')
                    
                    # Convert to BGR numpy array
                    frame_np = frame.to_ndarray(format='rgb24')
                    
                    metadata = {
                        'frame_id_str': frame_id_str,
                        'timestamp': time.time(),
                        'width': frame_np.shape[1],
                        'height': frame_np.shape[0]
                    }
                    
                    # Push to queue
                    await self.queue.put((frame_np, metadata))
                    logging.debug(f"Pushed frame to queue: {metadata['frame_id_str']}")
                    
            except Exception as e:
                logging.error(f"Error in frame producer: {e}")
                self._retry_count += 1
                if self._retry_count > 5:
                    logging.error("Max retries reached, stopping frame producer")
                    self.is_running = False
                    break
                await asyncio.sleep(1)  # Wait before retry
    
    async def read_data(self) -> Tuple[Any, Dict[str, Any]]:
        """Read frame from queue."""
        if not self.is_running:
            raise Exception("RTSP input not initialized or stopped")
        
        try:
            # Asynchronously get frame from queue
            frame_np, metadata = await self.queue.get()
            self.queue.task_done()
            return frame_np, metadata
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise Exception(f"Failed to read frame from queue: {e}")
    
    async def cleanup(self):
        """Clean up RTSP resources."""
        self.is_running = False
        if self._producer_task:
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass
        if self.container:
            await asyncio.get_event_loop().run_in_executor(self._executor, self.container.close)
        self._executor.shutdown(wait=True)
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        logging.info("RTSP input cleaned up")