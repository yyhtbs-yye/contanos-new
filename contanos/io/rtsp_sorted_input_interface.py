from typing import Any, Dict, Tuple
import logging
import av
import time
import asyncio
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
import heapq
import collections

def _safe_next_packet(frame_generator):
    try:
        return next(frame_generator)
    except StopIteration:
        return None

class RTSPInput(ABC):
    """RTSP stream input implementation using asyncio.Queue with order-forced prebuffer."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.addr = config['addr']
        self.topic = config['topic']
        self.container = None
        self.video_stream = None
        self.frame_generator = None
        self.queue = asyncio.Queue(maxsize=100)  # Consumer queue
        self.is_running = False
        self._producer_task = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._retry_count = 0

        # >>> ORDERING / PREBUFFER SETTINGS <<<
        # Warm up this many frames before emitting (helps reorder small inversions)
        self.prebuffer_warmup = int(config.get('prebuffer_warmup', 3))
        # If the prebuffer grows beyond this, drop the earliest *received* frame
        self.max_prebuffer = int(config.get('max_prebuffer', 5))

        # Prebuffer internals
        self._arrival_seq = 0  # strictly increasing arrival counter
        self._buf_heap = []    # min-heap of (fid, arrival_seq, frame_np, metadata)
        self._buf_arrivals = collections.deque()  # deque of (arrival_seq, fid)
        self._present = set()  # arrival_seq values currently in buffer (for quick membership)
        self._removed = set()  # arrival_seq lazily removed from heap
        self._next_expected = None  # next expected frame_id (int), contiguous if possible

    async def initialize(self) -> bool:
        try:
            logging.info(f"Connecting to RTSP stream: {self.addr} on {self.topic}")
            loop = asyncio.get_event_loop()
            # Prefer TCP to reduce reordering/loss
            self.container = await loop.run_in_executor(
                self._executor,
                lambda: av.open(self.addr + '/' + self.topic, options={'rtsp_transport': 'tcp'})
            )
            self.video_stream = next(s for s in self.container.streams if s.type == 'video')
            self.frame_generator = self.container.demux(self.video_stream)
            self.is_running = True
            self._producer_task = asyncio.create_task(self._frame_producer())
            self._retry_count = 0
            logging.info("RTSP connection established")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize RTSP input: {e}")
            return False

    async def _frame_producer(self):
        """Fetch, decode, order by frame_id_str (int), and push to queue."""
        loop = asyncio.get_event_loop()

        async def _emit_ready_in_order():
            """
            Emit frames in strictly ascending frame_id (allowing gaps if missing).
            Uses contiguous order when possible; never blocks forever on a missing frame.
            """
            # Warmup: don't emit until we have at least prebuffer_warmup frames (if not started yet)
            if self._next_expected is None:
                # Prune heap top if its entry was lazily removed
                while self._buf_heap and self._buf_heap[0][1] in self._removed:
                    heapq.heappop(self._buf_heap)
                if len(self._present) < self.prebuffer_warmup:
                    return
                if self._buf_heap:
                    # Start from the smallest available id
                    self._next_expected = self._buf_heap[0][0]

            # Try to emit as long as the smallest available id equals next_expected
            made_progress = True
            while made_progress:
                made_progress = False

                # Clean up lazy removals
                while self._buf_heap and self._buf_heap[0][1] in self._removed:
                    heapq.heappop(self._buf_heap)

                if not self._buf_heap:
                    return

                smallest_fid, smallest_seq, f_np, meta = self._buf_heap[0]

                if smallest_fid < self._next_expected:
                    # Too-late frame (older than what we already emitted or skipped) → drop
                    heapq.heappop(self._buf_heap)
                    if smallest_seq in self._present:
                        self._present.remove(smallest_seq)
                    continue

                if smallest_fid == self._next_expected:
                    # Perfect: emit in strict order
                    heapq.heappop(self._buf_heap)
                    if smallest_seq in self._present:
                        self._present.remove(smallest_seq)
                    # Also drop from arrival deque head if matches; else it’ll be skipped later
                    while self._buf_arrivals and self._buf_arrivals[0][0] not in self._present:
                        self._buf_arrivals.popleft()
                    await self.queue.put((f_np, meta))
                    self._next_expected += 1
                    made_progress = True
                    continue

                # smallest_fid > next_expected → we're missing next_expected.
                # Only skip (advance) when the buffer has grown too large (handled below).
                break

        def _drop_earliest_received_if_needed(reason: str):
            """If the prebuffer is too large, drop the earliest *received* frame (FIFO)."""
            dropped = False
            while len(self._present) > self.max_prebuffer:
                # Pop the earliest arrival that is still present
                while self._buf_arrivals and self._buf_arrivals[0][0] not in self._present:
                    self._buf_arrivals.popleft()
                if not self._buf_arrivals:
                    break
                seq_to_drop, fid_to_drop = self._buf_arrivals.popleft()
                if seq_to_drop in self._present:
                    self._present.remove(seq_to_drop)
                    self._removed.add(seq_to_drop)  # lazy removal from heap
                    dropped = True
                    logging.warning(
                        f"Prebuffer>{self.max_prebuffer} ({reason}). "
                        f"Dropping earliest received frame_id={fid_to_drop} (seq={seq_to_drop})."
                    )
                    # If we were waiting on this exact id, declare it lost and move on.
                    if self._next_expected is not None and fid_to_drop == self._next_expected:
                        self._next_expected += 1
            return dropped

        while self.is_running:
            try:
                packet = await loop.run_in_executor(self._executor, _safe_next_packet, self.frame_generator)
                if packet is None:
                    logging.warning("RTSP stream ended")
                    self.is_running = False
                    break

                for frame in packet.decode():
                    # Extract frame_id_str (from SEI UNREGISTERED if present)
                    frame_id_str = None
                    if frame.side_data:
                        for side_data in frame.side_data:
                            if side_data.type.name == "SEI_UNREGISTERED":
                                try:
                                    frame_id_str = bytes(side_data).decode('ascii', 'ignore').strip()
                                except Exception:
                                    frame_id_str = None

                    # Convert to numpy
                    frame_np = frame.to_ndarray(format='rgb24')

                    # Build metadata
                    metadata = {
                        'frame_id_str': frame_id_str,
                        'timestamp': time.time(),
                        'width': frame_np.shape[1],
                        'height': frame_np.shape[0],
                    }

                    # If we cannot parse an integer frame id, bypass ordering (push immediately)
                    try:
                        fid = int(frame_id_str) if frame_id_str is not None else None
                    except Exception:
                        fid = None

                    if fid is None:
                        await self.queue.put((frame_np, metadata))
                        logging.debug("Bypassed ordering (no/invalid frame_id_str); pushed directly.")
                        continue

                    # Insert into prebuffer (both heap and arrival deque)
                    self._arrival_seq += 1
                    seq = self._arrival_seq
                    heapq.heappush(self._buf_heap, (fid, seq, frame_np, metadata))
                    self._buf_arrivals.append((seq, fid))
                    self._present.add(seq)

                    # Try to emit what we can in strict order
                    await _emit_ready_in_order()

                    # If we're stuck missing next_expected and buffer is too large, drop earliest
                    if self._next_expected is not None:
                        # Check if smallest fid is greater than next_expected -> missing
                        smallest_valid = None
                        while self._buf_heap and self._buf_heap[0][1] in self._removed:
                            heapq.heappop(self._buf_heap)
                        if self._buf_heap:
                            smallest_valid = self._buf_heap[0][0]

                        if smallest_valid is not None and smallest_valid > self._next_expected:
                            # We're blocked by a missing frame. If buffer too large, drop earliest & advance.
                            dropped = _drop_earliest_received_if_needed(reason="waiting for missing next_expected")
                            if dropped:
                                # After dropping, we may have skipped the missing id; try emitting again.
                                await _emit_ready_in_order()

                    # Also enforce max size generally (e.g., bursty arrivals)
                    if len(self._present) > self.max_prebuffer:
                        _drop_earliest_received_if_needed(reason="general overflow")
                        await _emit_ready_in_order()

            except Exception as e:
                logging.error(f"Error in frame producer: {e}")
                self._retry_count += 1
                if self._retry_count > 5:
                    logging.error("Max retries reached, stopping frame producer")
                    self.is_running = False
                    break
                await asyncio.sleep(1)

    async def read_data(self) -> Tuple[Any, Dict[str, Any]]:
        if not self.is_running:
            raise Exception("RTSP input not initialized or stopped")
        try:
            frame_np, metadata = await self.queue.get()
            self.queue.task_done()
            return frame_np, metadata
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise Exception(f"Failed to read frame from queue: {e}")

    async def cleanup(self):
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
