from typing import Any, Dict, Tuple, Optional
import logging
import json
import time
import asyncio
import threading
from abc import ABC
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.client as mqtt
import heapq

class MQTTSortedInput(ABC):
    """MQTT message input implementation with frame ordering capability using paho-mqtt."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.addr: str = config["addr"]
        proto, rest = self.addr.split("://", 1)
        self.broker_host, broker_port = rest.split(":")
        self.broker_port: int = int(broker_port)
        self.topic = config['topic']
        self.username = config.get('username')
        self.password = config.get('password')
        self.client_id = config.get('client_id', f"mqtt_input_{int(time.time())}")
        self.qos = int(config.get('qos', 0))
        self.keepalive = config.get('keepalive', 60)
        
        # Ordering configuration
        self.buffer_threshold = int(config.get('buffer_threshold', 50))  # Buffer size threshold for skipping frames
        
        self.client = None
        self.message_queue = Queue(maxsize=100)  # Raw messages from MQTT
        self.ordered_queue = Queue(maxsize=100)  # Ordered messages for output
        self.frame_buffer = []  # Min-heap for sorting frames
        # Expected frame ID will be filled with the first frame we actually receive
        self.expected_frame_id: Optional[int] = None
        
        self.is_running = False
        self.is_connected = False
        self._ordering_task = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._mqtt_thread = None
        self._connection_event = threading.Event()
        self._loop = None
    
    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    
    def _extract_frame_id(self, payload_data: Any) -> Optional[int]:
        """Extract frame ID from payload data."""
        if not isinstance(payload_data, dict):
            return None
        
        frame_id_str = payload_data.get('frame_id_str')
        if not frame_id_str or not isinstance(frame_id_str, str):
            return None
        
        try:
            # Extract frame ID using split('FRAME:')[-1]
            frame_id_part = frame_id_str.split('FRAME:')[-1]
            return int(frame_id_part)
        except (ValueError, IndexError):
            logging.warning(f"Failed to extract frame ID from: {frame_id_str}")
            return None
    
    # ---------------------------------------------------------------------
    # MQTT callbacks
    # ---------------------------------------------------------------------
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client receives a CONNACK response from the server."""
        if rc == 0:
            logging.info("Connected to MQTT broker successfully")
            self.is_connected = True
            client.subscribe(self.topic, qos=self.qos)
            logging.info(f"Subscribed to topic: {self.topic}")
            self._connection_event.set()
        else:
            logging.error(f"Failed to connect to MQTT broker, return code {rc}")
            self.is_connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the server."""
        logging.info(f"Disconnected from MQTT broker with result code {rc}")
        self.is_connected = False
        self._connection_event.clear()
    
    def _on_message(self, client, userdata, msg):
        """Callback for when a PUBLISH message is received from the server."""
        try:
            # Process message in a thread-safe way
            message = self._process_message_sync(msg)

            if message is None:
                return
            
            # Put message into asyncio queue from thread
            if self._loop and self.is_running:
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(message), 
                    self._loop
                )
                logging.debug(f"Pushed raw message to queue: {message['timestamp']}")
        except Exception as e:
            logging.error(f"Error processing MQTT message in callback: {e}")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """Callback for when the broker responds to a subscribe request."""
        logging.info(f"Subscribed with QoS: {granted_qos}")
    
    # ---------------------------------------------------------------------
    # Internal processing helpers
    # ---------------------------------------------------------------------
    
    def _process_message_sync(self, msg):
        """Process MQTT message synchronously."""
        try:
            # Decode payload
            if isinstance(msg.payload, bytes):
                payload_str = msg.payload.decode('utf-8')
            else:
                payload_str = str(msg.payload)

            # Skip x264 metadata
            if 'x264' in payload_str:
                return None

            # Try to parse as JSON, fall back to string
            try:
                payload_data = json.loads(payload_str)
            except json.JSONDecodeError:
                payload_data = payload_str
            
            return {
                'frame_id_str': payload_data['frame_id_str'],
                'topic': msg.topic,
                'payload': payload_data,
                'qos': msg.qos,
                'retain': msg.retain,
                'timestamp': time.time()
            }
        except UnicodeDecodeError:
            # If can't decode as text, keep as bytes
            return {
                'frame_id_str': msg.payload['frame_id_str'],
                'topic': msg.topic,
                'payload': msg.payload,
                'qos': msg.qos,
                'retain': msg.retain,
                'timestamp': time.time()
            }
    
    # ---------------------------------------------------------------------
    # MQTT lifecycle helpers
    # ---------------------------------------------------------------------
    
    def _mqtt_loop_thread(self):
        """Thread function to run MQTT client loop."""
        try:
            # Create MQTT client
            self.client = mqtt.Client(client_id=self.client_id)
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_subscribe = self._on_subscribe
            
            # Set credentials if provided
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # Connect to broker
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            
            # Start the network loop
            self.client.loop_forever()
            
        except Exception as e:
            logging.error(f"Error in MQTT thread: {e}")
            self.is_connected = False
    
    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    
    async def initialize(self) -> bool:
        """Initialize MQTT connection and start message ordering task."""
        try:
            logging.info(f"Connecting to MQTT broker: {self.broker_host}:{self.broker_port}")
            
            # Store current event loop
            self._loop = asyncio.get_event_loop()
            self.is_running = True
            
            # Start MQTT client in separate thread
            self._mqtt_thread = threading.Thread(target=self._mqtt_loop_thread, daemon=True)
            self._mqtt_thread.start()
            
            # Wait for connection (with timeout)
            connection_timeout = 10  # seconds
            await asyncio.get_event_loop().run_in_executor(
                None, self._connection_event.wait, connection_timeout
            )
            
            if not self.is_connected:
                logging.error("Failed to connect to MQTT broker within timeout")
                return False
            
            # Start message ordering task
            self._ordering_task = asyncio.create_task(self._message_ordering())
            
            logging.info(f"MQTT connection established, subscribed to topic: {self.topic}")
            return True
            
        except Exception as e:
            logging.error(f"Unexpected error during MQTT initialization: {e}")
            return False
    
    # ---------------------------------------------------------------------
    # Ordering logic
    # ---------------------------------------------------------------------
    
    async def _message_ordering(self):
        """Background task to order messages by frame ID."""
        while self.is_running:
            try:
                # Get message from raw queue
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                
                # Extract frame ID
                frame_id = self._extract_frame_id(message['payload'])
                
                # Initialise expected_frame_id on first frame
                if self.expected_frame_id is None and frame_id is not None:
                    self.expected_frame_id = frame_id
                    logging.debug(f"Initial expected_frame_id set to {self.expected_frame_id}")
                
                if frame_id is None or self.expected_frame_id is None:
                    # If we don't yet know what to expect, or message has no frame ID, pass through
                    await self.ordered_queue.put(message)
                    logging.debug("Passed through message without frame ID or before initialisation")
                    self.message_queue.task_done()
                    continue

                if frame_id < self.expected_frame_id:
                    logging.warning(f"Dropping stale frame {frame_id}, already expecting {self.expected_frame_id}")
                    self.message_queue.task_done()
                    continue

                # Add to buffer with frame ID as priority
                heapq.heappush(self.frame_buffer, (frame_id, message))
                logging.debug(f"Buffered frame {frame_id}, buffer size: {len(self.frame_buffer)}")
                
                # Try to release ordered messages
                await self._release_ordered_messages()
                
                # Mark raw message as done
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in message ordering: {e}")
                await asyncio.sleep(0.1)
    
    async def _release_ordered_messages(self):
        """Release consecutive ordered messages from buffer."""
        # If we haven't seen any valid frames yet, bail early
        if self.expected_frame_id is None:
            return
        
        while self.frame_buffer:
            # Check if the next expected frame is available
            if self.frame_buffer[0][0] == self.expected_frame_id:
                frame_id, message = heapq.heappop(self.frame_buffer)
                await self.ordered_queue.put(message)
                self.expected_frame_id += 1
                logging.debug(f"Released frame {frame_id}, next expected: {self.expected_frame_id}")
            else:
                # Next frame not available, check if buffer exceeds threshold
                if len(self.frame_buffer) > self.buffer_threshold:
                    await self._skip_to_next_available_frame()
                break
    
    async def _skip_to_next_available_frame(self):
        """Skip missing frames when buffer exceeds threshold."""
        # If expected_frame_id has not been set yet, there is nothing to skip
        if self.expected_frame_id is None:
            return
        
        while len(self.frame_buffer) > self.buffer_threshold:
            # Skip the current expected frame and move to next
            self.expected_frame_id += 1
            logging.warning(f"Skipped missing frame, now expecting frame {self.expected_frame_id}")
            
            # Check if the new expected frame is available
            if self.frame_buffer and self.frame_buffer[0][0] == self.expected_frame_id:
                # Found the next expected frame, release it and continue normal processing
                frame_id, message = heapq.heappop(self.frame_buffer)
                await self.ordered_queue.put(message)
                self.expected_frame_id += 1
                logging.debug(f"Released frame {frame_id} after skip, next expected: {self.expected_frame_id}")
                break
            
            # If buffer is still above threshold, continue skipping
            # If buffer is now at or below threshold, stop skipping and wait
    
    # ---------------------------------------------------------------------
    # Consumer API
    # ---------------------------------------------------------------------
    
    async def read_data(self) -> Tuple[Any, Dict[str, Any]]:
        """Read ordered message from queue."""
        if not self.is_running:
            raise Exception("MQTT input not initialized or stopped")
        
        try:
            # Get message from ordered queue
            message = await asyncio.wait_for(self.ordered_queue.get(), timeout=5.0)
            payload_data = message['payload']
            
            # Extract frame info for metadata
            frame_id = self._extract_frame_id(payload_data)
            
            metadata = {
                'frame_id_str': payload_data.get('frame_id_str') if isinstance(payload_data, dict) else None,
                'frame_id': frame_id,
                'topic': message['topic'],
                'qos': message['qos'],
                'retain': message['retain'],
                'timestamp': message['timestamp'],
                'payload_type': type(payload_data).__name__
            }
            
            self.ordered_queue.task_done()
            return payload_data, metadata
            
        except asyncio.TimeoutError:
            raise Exception("No ordered MQTT message received within timeout")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise Exception(f"Failed to read ordered MQTT message: {e}")
    
    async def cleanup(self):
        """Clean up MQTT resources."""
        self.is_running = False
        
        # Cancel ordering task
        if self._ordering_task:
            self._ordering_task.cancel()
            try:
                await self._ordering_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect MQTT client
        if self.client and self.is_connected:
            self.client.disconnect()
        
        # Wait for MQTT thread to finish
        if self._mqtt_thread and self._mqtt_thread.is_alive():
            self._mqtt_thread.join(timeout=5)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Clear queues
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        while not self.ordered_queue.empty():
            try:
                self.ordered_queue.get_nowait()
                self.ordered_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Clear frame buffer
        self.frame_buffer.clear()
        
        logging.info("MQTT input cleaned up")