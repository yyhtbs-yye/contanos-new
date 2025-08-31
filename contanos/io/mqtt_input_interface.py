from typing import Any, Dict, Tuple
import logging
import json
import time
import asyncio
import uuid
from abc import ABC
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.client as mqtt

class MQTTInput(ABC):
    """MQTT message input implementation using asyncio.Queue and paho-mqtt."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.addr: str = config["addr"]
        proto, rest = self.addr.split("://", 1)
        self.broker_host, broker_port = rest.split(":")
        self.broker_port: int = int(broker_port)
        self.topic = config['topic']
        self.username = config.get('username')
        self.password = config.get('password')
        base_id = config.get('client_id', f"mqtt_input_{int(time.time())}")
        unique_suffix = str(uuid.uuid4())[:8]  # Add unique suffix
        self.client_id = f"{base_id}_{unique_suffix}"
        self.qos = int(config.get('qos', 0))
        
        # Asyncio constructs
        self.message_queue: Queue = Queue(maxsize=100)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mqtt_{self.client_id}")
        self._loop: asyncio.AbstractEventLoop | None = None   # ← not yet
        
        # Paho client
        self.client = mqtt.Client(client_id=self.client_id)
        if self.username:
            self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.is_running = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Connected to MQTT broker: {self.broker_host}:{self.broker_port}")
            client.subscribe(self.topic, qos=self.qos)
            self.is_running = True
        else:
            logging.error(f"MQTT connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        # Called in network thread; schedule processing in asyncio loop
        try:
            # asyncio.run_coroutine_threadsafe(self._process_and_queue(msg), self._loop)
            # fut = asyncio.run_coroutine_threadsafe(self._process_and_queue(msg), self._loop)
            # fut.result()
            if not self._loop:
                logging.warning("Received MQTT message before initialise completed")
                return

            fut = asyncio.run_coroutine_threadsafe(
                self._process_and_queue(msg), self._loop
            )
            # Log any exception once the coroutine finishes, but never block:
            fut.add_done_callback(
                lambda f: logging.error("MQTT handler error: %s", f.exception())
                if f.exception()
                else None
            )

        except Exception as e:
            logging.error(f"Error queuing MQTT message: {e}")

    async def _process_and_queue(self, msg):
        # Run blocking decode in executor
        message = await self._loop.run_in_executor(self._executor, self._process_message, msg)
        if message is not None:  # Only queue valid frame messages
            await self.message_queue.put(message)
            logging.debug(f"Pushed message to queue: {message['frame_id_str']}")

    def _process_message(self, msg):
        # Blocking message processing
        try:
            payload = msg.payload
            if isinstance(payload, bytes):
                payload_str = payload.decode('utf-8')
            else:
                payload_str = str(payload)

            # Skip x264 metadata
            if 'x264' in payload_str:
                return None

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
            return {
                'frame_id_str': msg.payload['frame_id_str'],
                'topic': msg.topic,
                'payload': msg.payload,
                'qos': msg.qos,
                'retain': msg.retain,
                'timestamp': time.time()
            }

    async def initialize(self) -> bool:
        """Initialize MQTT connection and start background loop."""
        try:

            self._loop = asyncio.get_running_loop()            # ← correct loop
            self._executor = ThreadPoolExecutor(max_workers=1) # recreate after we know the loop

            logging.info(f"Starting MQTT client for broker {self.broker_host}:{self.broker_port}")
            # Start network loop in background thread
            self.client.connect(self.broker_host, self.broker_port)
            self.client.loop_start()
            # Wait for connection
            timeout = 5
            start = time.time()
            while not self.is_running and time.time() - start < timeout:
                await asyncio.sleep(0.1)
            if not self.is_running:
                logging.error("Timeout waiting for MQTT connection")
                return False
            logging.info(f"MQTT connection established, subscribed to topic: {self.topic}")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize MQTT input: {e}")
            return False

    async def read_data(self) -> Tuple[Any, Dict[str, Any]]:
        """Read message from queue."""
        if not self.is_running:
            raise Exception("MQTT input not initialized or stopped")
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=5.0)
            # message = await self.message_queue.get()
            data = message['payload']
            metadata = {
                'frame_id_str': data.get('frame_id_str') if isinstance(data, dict) else None,
                'topic': message['topic'],
                'qos': message['qos'],
                'retain': message['retain'],
                'timestamp': message['timestamp'],
                'payload_type': type(data).__name__
            }
            self.message_queue.task_done()
            return data, metadata
        except asyncio.TimeoutError:
            raise Exception("No MQTT message received within timeout")
        except Exception as e:
            raise Exception(f"Failed to read MQTT message: {e}")

    async def cleanup(self):
        """Clean up MQTT resources."""
        self.is_running = False
        # Stop network loop and disconnect
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logging.error(f"Error during MQTT cleanup: {e}")
        # Shutdown executor
        self._executor.shutdown(wait=True)
        # Clear queue
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except asyncio.QueueEmpty:
                break
        logging.info("MQTT input cleaned up")