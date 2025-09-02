# 1. Base off the official Mosquitto image
FROM eclipse-mosquitto:2.0

# 2. Copy our custom config into the container
COPY mosquitto.conf /mosquitto/config/mosquitto.conf

# 3. Expose MQTT port
EXPOSE 1883

# 4. Run Mosquitto with our config
CMD ["mosquitto", "-c", "/mosquitto/config/mosquitto.conf"]
