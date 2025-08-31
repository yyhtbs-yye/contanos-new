from typing import Any, Dict
import logging
import asyncio
import subprocess
import numpy as np

from abc import ABC

class RTSPOutput(ABC):
    """RTSP output implementation using ffmpeg subprocess + asyncio.Queue."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.addr: str = config["addr"]
        self.topic: str = config["topic"]
        self.width: int = int(config.get("width", 640))
        self.height: int = int(config.get("height", 480))
        self.fps: int = config.get("fps", 30)
        self.bitrate: str = config.get("bitrate", "1000k")
        self.codec: str = config.get("codec", "libx264")
        self.preset: str = config.get("preset", "ultrafast")
        self.pixel_format: str = config.get("pixel_format", "yuv420p")

        self.process: subprocess.Popen | None = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.get("queue_max_len", 10))
        self.is_running: bool = False
        self._producer_task: asyncio.Task | None = None

    #
    # ───────────────────────────────  PUBLIC API  ────────────────────────────────
    #
    async def initialize(self) -> bool:
        """
        Configure and start the ffmpeg process for RTSP streaming.
        """
        try:
            logging.info(f"Starting RTSP stream to {self.addr} on {self.topic}")

            # --- 1. Build ffmpeg command --------------------------------------------
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',  # input from stdin
                '-c:v', self.codec,
                '-preset', self.preset,
                '-pix_fmt', self.pixel_format,
                '-b:v', self.bitrate,
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                self.addr+'/'+self.topic
            ]

            # --- 2. Start ffmpeg process -----------------------------------------
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            # --- 3. Kick-off the async producer ---------------------------------
            self.is_running = True
            self._producer_task = asyncio.create_task(self._output_producer())

            logging.info("RTSP output initialized")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize RTSP output: {e}")
            return False

    #
    # ────────────────────────────────  PRODUCER  ────────────────────────────────
    #
    async def _output_producer(self) -> None:
        """
        Async background task:
        • Waits for frame data in `self.queue`
        • Writes frames to ffmpeg stdin in a worker thread
        """
        assert self.process is not None  # mypy / type safety

        while self.is_running:
            try:
                frame_np = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                # Ensure frame is in correct format (RGB24)
                if frame_np.dtype != np.uint8:
                    frame_np = frame_np.astype(np.uint8)
                
                # Ensure frame has correct dimensions
                if frame_np.shape[:2] != (self.height, self.width):
                    logging.warning(f"Frame dimensions {frame_np.shape[:2]} don't match expected {(self.height, self.width)}")
                
                # Off-load the blocking write to a thread:
                await asyncio.to_thread(
                    self._write_frame_to_ffmpeg,
                    frame_np
                )
                
                logging.debug(f"Streamed frame with shape {frame_np.shape}")
                self.queue.task_done()

            except asyncio.TimeoutError:
                continue  # idle loop – no frame yet
            except Exception as e:
                logging.error(f"Unexpected error in output producer: {e}")
                break

    def _write_frame_to_ffmpeg(self, frame_np: np.ndarray) -> None:
        """Write a single frame to ffmpeg stdin (blocking operation)."""
        if self.process and self.process.stdin:
            try:
                # Convert frame to bytes and write to stdin
                frame_bytes = frame_np.tobytes()
                self.process.stdin.write(frame_bytes)
                self.process.stdin.flush()
            except BrokenPipeError:
                logging.error("FFmpeg process has terminated unexpectedly")
                self.is_running = False
            except Exception as e:
                logging.error(f"Error writing frame to ffmpeg: {e}")
                raise

    #
    # ────────────────────────────────  WRITE DATA  ────────────────────────────────
    #
    async def write_data(self, results: Dict[str, Any]) -> bool:
        """Put a frame into the outbound queue."""
        if not self.is_running:
            raise RuntimeError("RTSP output not initialized")

        try:
            frame_np = results['results']['img']
            await self.queue.put(frame_np)
            return True
        except Exception as e:
            logging.error(f"Failed to queue frame: {e}")
            raise RuntimeError(f"Failed to write RTSP frame: {e}") from e

    #
    # ────────────────────────────────  CLEAN-UP  ────────────────────────────────
    #
    async def cleanup(self) -> None:
        """Flush queue, stop producer task, and terminate ffmpeg process."""
        self.is_running = False

        # 1. Stop producer gracefully
        if self._producer_task:
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass

        # 2. Close ffmpeg stdin and terminate process
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                
                # Wait for process to terminate with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait), 
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logging.warning("FFmpeg process didn't terminate gracefully, killing it")
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait)
                    
            except Exception as e:
                logging.error(f"Error during ffmpeg cleanup: {e}")
            finally:
                self.process = None

        # 3. Drain queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except:
                break

        logging.info("RTSP output cleaned up")