#!/usr/bin/env python3
"""
mqtt_sniffer.py - subscribe to “#” and dump every message to the console.

Usage examples
--------------
# Quick run with defaults (localhost:1883)
python mqtt_sniffer.py

# Specify broker, credentials, and a client-ID
python mqtt_sniffer.py --host broker.hivemq.com --port 1883 \
                       --username myuser --password secret \
                       --client-id my_sniffer
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

topic_subscribe = "bytetrack"

def on_connect(client: mqtt.Client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[{ts()}] Connected OK - subscribing to {topic_subscribe}")
        # Subscribe to every topic, QoS 0
        client.subscribe(topic_subscribe, qos=2)
    else:
        print(f"[{ts()}] Bad connection (return-code={rc})")
        sys.exit(1)


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    payload: bytes = msg.payload
    try:
        # Attempt to decode as UTF-8; fall back to raw bytes if that fails
        text = payload.decode()
    except UnicodeDecodeError:
        text = repr(payload)

    print(f"[{ts()}] {msg.topic} ({msg.qos}) → {text}")


def ts() -> str:
    """Human-readable timestamp for log lines."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MQTT message sniffer")
    p.add_argument("--host", default="localhost", help="Broker hostname or IP")
    p.add_argument("--port", type=int, default=1883, help="Broker port")
    p.add_argument("--client-id", default="mqtt_sniffer", help="MQTT client-ID")
    p.add_argument("--username", help="Username (if broker requires auth)")
    p.add_argument("--password", help="Password (if broker requires auth)")
    p.add_argument("--keepalive", type=int, default=60, help="Keep-alive seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    client = mqtt.Client(client_id=args.client_id, transport="tcp")
    if args.username:
        client.username_pw_set(args.username, args.password)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.host, args.port, args.keepalive)

    try:
        client.loop_forever()          # blocking network loop
    except KeyboardInterrupt:
        print("\nInterrupted - shutting down")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
