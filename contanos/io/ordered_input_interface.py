from typing import Any, Dict, Tuple, Optional, List
import logging
import time
import asyncio
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
import heapq

class OrderedInputInterface:

    def __init__(self, interface, config: Dict[str, Any] = {}):
        super().__init__()
        self.interface = interface

        # Public state
        self.is_running = False
        self.is_connected = False

        # Output queue (already ordered)
        self.ordered_queue: Queue[Tuple[List[Any], List[Dict[str, Any]]]] = Queue(maxsize=100)

        # Ordering configuration
        # Max number of out-of-order frames to buffer before skipping ahead.
        self.buffer_threshold: int = int(config.get("buffer_threshold", 10))
        # Optional starting frame id; if None, we'll infer from the first seen frame.
        self._start_frame_id: Optional[int] = config.get("start_frame_id")

        # Internals for ordering
        self._heap: List[Tuple[int, int, List[Any], List[Dict[str, Any]]]] = []  # (frame_id, seq, data, meta)
        self._seq_counter: int = 0  # disambiguates ties in heap
        self._expected_frame_id: Optional[int] = self._start_frame_id
        self._lock = asyncio.Lock()

        # Async task/executor management
        self._ordering_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _extract_frame_id(self, payload_data: Any) -> Optional[int]:
        """Extract frame ID from a metadata dict containing 'frame_id_str' with 'FRAME:<int>'."""
        if not isinstance(payload_data, dict):
            return None

        frame_id_str = payload_data.get("frame_id_str")
        if not frame_id_str or not isinstance(frame_id_str, str):
            return None

        try:
            # Extract frame ID using split('FRAME:')[-1]
            frame_id_part = frame_id_str.split("FRAME:")[-1]
            return int(frame_id_part)
        except (ValueError, IndexError):
            logging.warning(f"Failed to extract frame ID from: {frame_id_str}")
            return None

    async def initialize(self) -> bool:
        """
        Initialize the wrapped interface and start the background ordering loop.
        """
        try:
            ok = await self.interface.initialize()
            if not ok:
                logging.error("OrderedInputInterface: underlying interface failed to initialize.")
                return False

            self.is_connected = True
            self.is_running = True
            self._ordering_task = asyncio.create_task(self._ordering_loop())
            logging.info("OrderedInputInterface initialized.")
            return True
        except Exception as e:
            logging.exception(f"OrderedInputInterface.initialize() failed: {e}")
            return False

    async def _ordering_loop(self):
        """
        Continuously pull frames from the wrapped interface, buffer by frame_id,
        and push in-order frames to `ordered_queue`.
        """
        try:
            while self.is_running:
                try:
                    data_list, metadata_list = await self.interface.read_data()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logging.error(f"OrderedInputInterface: error reading from underlying interface: {e}")
                    await asyncio.sleep(0.01)
                    continue

                # Find a frame_id in the metadata list
                frame_id: Optional[int] = None
                for md in metadata_list:
                    frame_id = self._extract_frame_id(md)
                    if frame_id is not None:
                        break

                if frame_id is None:
                    logging.warning("OrderedInputInterface: no valid frame_id found; dropping frame.")
                    continue

                to_emit: List[Tuple[List[Any], List[Dict[str, Any]]]] = []

                async with self._lock:
                    # Push new frame into min-heap by frame_id
                    heapq.heappush(self._heap, (frame_id, self._seq_counter, data_list, metadata_list))
                    self._seq_counter += 1

                    # If we don't have an expected frame yet, anchor it to the smallest currently buffered.
                    if self._expected_frame_id is None:
                        self._expected_frame_id = self._heap[0][0]

                    # Drain all frames that match the expected id in ascending order.
                    while self._heap:
                        top_id, _, top_data, top_meta = self._heap[0]
                        if top_id < self._expected_frame_id:
                            # Late/duplicate frame; discard it to maintain strict ordering.
                            heapq.heappop(self._heap)
                            logging.debug(
                                f"OrderedInputInterface: discarding late frame {top_id} (< expected {self._expected_frame_id})."
                            )
                            continue
                        if top_id == self._expected_frame_id:
                            heapq.heappop(self._heap)
                            to_emit.append((top_data, top_meta))
                            self._expected_frame_id += 1
                            continue
                        # top_id > expected -> we must wait for the missing frame(s)
                        break

                    # If the buffer is too large and we're still missing the expected frame,
                    # skip ahead to the smallest available to avoid stalling forever.
                    if (
                        len(self._heap) >= self.buffer_threshold
                        and self._heap
                        and self._expected_frame_id is not None
                        and self._heap[0][0] > self._expected_frame_id
                    ):
                        skip_id, _, skip_data, skip_meta = heapq.heappop(self._heap)
                        logging.warning(
                            f"OrderedInputInterface: buffer size {len(self._heap)+1} exceeded threshold "
                            f"{self.buffer_threshold}; skipping ahead from expected {self._expected_frame_id} to {skip_id}."
                        )
                        # Emit the skipped-ahead frame and set new expected accordingly.
                        to_emit.append((skip_data, skip_meta))
                        self._expected_frame_id = skip_id + 1

                        # After skipping, we may be able to drain more consecutive frames.
                        while self._heap:
                            top_id, _, top_data, top_meta = self._heap[0]
                            if top_id == self._expected_frame_id:
                                heapq.heappop(self._heap)
                                to_emit.append((top_data, top_meta))
                                self._expected_frame_id += 1
                            elif top_id < self._expected_frame_id:
                                heapq.heappop(self._heap)  # drop late/duplicate
                            else:
                                break

                # Perform the (potentially blocking) queue puts outside the lock.
                for item in to_emit:
                    await self.ordered_queue.put(item)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.exception(f"OrderedInputInterface._ordering_loop crashed: {e}")

    async def read_data(self) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Return the next in-order (data_list, metadata_list).
        Raises if not initialized/running.
        """
        if not self.is_running:
            raise RuntimeError("OrderedInputInterface not initialized or already stopped.")

        try:
            item = await self.ordered_queue.get()
            self.ordered_queue.task_done()
            return item
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"OrderedInputInterface.read_data failed: {e}")

    async def cleanup(self):
        """
        Stop background tasks, drain buffers, and cleanup the wrapped interface.
        """
        self.is_running = False

        # Cancel ordering task
        if self._ordering_task:
            self._ordering_task.cancel()
            try:
                await asyncio.gather(self._ordering_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            self._ordering_task = None

        # Cleanup wrapped interface
        try:
            await self.interface.cleanup()
        except Exception as e:
            logging.error(f"OrderedInputInterface: error cleaning up wrapped interface: {e}")

        # Drain local buffers
        async with self._lock:
            self._heap.clear()
            while not self.ordered_queue.empty():
                try:
                    self.ordered_queue.get_nowait()
                    self.ordered_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        # Shutdown executor
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            pass

        self.is_connected = False
        logging.info("OrderedInputInterface cleaned up.")
