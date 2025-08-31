from typing import Any, Dict
import logging
import json
import asyncio

from abc import ABC
from paho.mqtt.client import Client

class MQTTOutput(ABC):
    """MQTT output implementation using a plain paho-mqtt client + asyncio.Queue."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.addr: str = config["addr"]
        proto, rest = self.addr.split("://", 1)
        self.broker_host, broker_port = rest.split(":")
        self.broker_port: int = int(broker_port)

        self.topic: str = config["topic"]
        self.username: str | None = config.get("username")
        self.password: str | None = config.get("password")
        self.client_id: str = config.get(
            "client_id", f"mqtt_output_{int(asyncio.get_event_loop().time())}"
        )
        self.qos: int = int(config.get("qos", 0))

        self.client: Client | None = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=int(config.get("queue_max_len", 100)))
        self.is_running: bool = False
        self._producer_task: asyncio.Task | None = None
    #
    # ───────────────────────────────  PUBLIC API  ────────────────────────────────
    #
    async def initialize(self) -> bool:
        """
        Configure and connect the MQTT client.
        Runs paho-mqtt in a background thread (`loop_start()`).
        """
        try:
            logging.info(f"Connecting to MQTT broker {self.broker_host}:{self.broker_port}")

            # --- 1. Build the client -------------------------------------------------
            client = Client(client_id=self.client_id, transport="tcp")
            if self.username:
                client.username_pw_set(self.username, self.password)

            # --- 2. Connect & start network loop ------------------------------------
            client.connect(self.broker_host, self.broker_port, keepalive=60)
            client.loop_start()            # network loop in a dedicated thread
            self.client = client

            # --- 3. Kick-off the async producer ------------------------------------
            self.is_running = True
            self._producer_task = asyncio.create_task(self._output_producer())

            logging.info("MQTT output initialised")
            return True

        except Exception as e:
            logging.error(f"Failed to initialise MQTT output: {e}")
            return False
    #
    # ────────────────────────────────  PRODUCER  ────────────────────────────────
    #
    async def _output_producer(self) -> None:
        """
        Async background task:
        • Waits for items in `self.queue`
        • Publishes them via paho-mqtt (`publish()` is run in a worker thread
          with `asyncio.to_thread` so we never block the event loop).
        """
        assert self.client is not None  # mypy / type safety

        while self.is_running:
            try:
                results = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                # Off-load the blocking publish to a thread:
                await asyncio.to_thread(
                    self.client.publish,
                    self.topic,
                    json.dumps(results, default=str),
                    qos=self.qos,
                )
                logging.debug(f"Published to {self.topic}: {results}")
                self.queue.task_done()

            except asyncio.TimeoutError:
                continue  # idle loop – no message yet
            except Exception as e:
                logging.error(f"Unexpected error in output producer: {e}")
    #
    # ────────────────────────────────  WRITE DATA  ────────────────────────────────
    #
    async def write_data(self, results: Dict[str, Any]) -> bool:
        """Put a results into the outbound queue."""
        if not self.is_running:
            raise RuntimeError("MQTT output not initialised")

        try:
            await self.queue.put(results)
            return True
        except Exception as e:
            logging.error(f"Failed to queue data: {e}")
            raise RuntimeError(f"Failed to write MQTT data: {e}") from e
    #
    # ────────────────────────────────  CLEAN-UP  ────────────────────────────────
    #
    async def cleanup(self) -> None:
        """Flush queue, stop producer task, and disconnect the MQTT client."""
        self.is_running = False

        # 1. Stop producer gracefully
        if self._producer_task:
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass

        # 2. Stop MQTT network loop & disconnect
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None

        # 3. Drain queue
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

        logging.info("MQTT output cleaned up")