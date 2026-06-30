# Working with Apache Kafka

Apache Kafka is a distributed event streaming platform. It can publish, subscribe to, store, and process streams of records in real time.

## Key Concepts

- **Producer** — publishes messages to a topic
- **Consumer** — subscribes to topics and processes messages
- **Topic** — a named feed where messages are stored and categorized
- **Broker** — a Kafka server that stores data
- **Partition** — topics are split into partitions for parallel processing
- **Offset** — a unique ID for each message within a partition

## Installing Confluent Kafka

```bash
pip install confluent-kafka
```

## Kafka Producer

```python
import uuid
import json
from confluent_kafka import Producer

# Create Producer Instance
producer = Producer({"bootstrap.servers": "localhost:9092"})

# Sample Data
data = {
    "id": str(uuid.uuid4()),
    "user": "Freak",
    "item": "USB C Cable",
    "quantity": "5",
}

# Convert data to bytes for Kafka
data_str = json.dumps(data)
data_bytes = data_str.encode("utf-8")


# Callback on message delivery
def producer_delivery_report(err, msg):
    if err:
        print(f"Error occurred: {err}")
        return

    print(f"Delivered Message: {msg.value().decode('utf-8')}")
    print(f"  to Topic - {msg.topic()}")
    print(f"  on Partition - {msg.partition()}")
    print(f"  at Offset - {msg.offset()}")


# Send data to Kafka Broker
producer.produce(
    topic="orders", value=data_bytes, callback=producer_delivery_report
)

# Flush to force delivery
producer.flush()
```

## Kafka Consumer

```python
import json
from confluent_kafka import Consumer

# Create Consumer Instance
consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "auto.offset.reset": "earliest",
    "group.id": "order-tracker2"
})

# Subscribe to a topic
consumer.subscribe(["orders"])

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            print(f"Error occurred: {msg.error()}")
            continue

        # Process the message
        value = msg.value().decode("utf-8")
        order = json.loads(value)

        print(f"Received order: {order['quantity']} x {order['item']} from {order['user']}")

except KeyboardInterrupt:
    print("Consumer connection closed")

finally:
    consumer.close()
```

## Running the Example

1. Start Kafka (using Docker):

```bash
docker run -d \
    --name kafka \
    -p 9092:9092 \
    apache/kafka
```

Alternatively, use Docker Compose with Zookeeper:

```yaml
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

2. Start the consumer first to be ready for messages:

```bash
python consumer.py
```

3. In another terminal, run the producer:

```bash
python producer.py
```

## Consumer Groups

Consumers with the same `group.id` share the workload — each message is delivered to only one consumer in the group. This enables parallel processing.

```python
consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "order-processors",  # same group = load balancing
    "auto.offset.reset": "earliest"
})
```

## Common Configuration Options

### Producer

| Setting | Description |
|---------|-------------|
| `bootstrap.servers` | Kafka broker address |
| `acks` | Acknowledgement level (`0`, `1`, `all`) |
| `compression.type` | Message compression (`gzip`, `snappy`, `lz4`) |
| `retries` | Number of retries on failure |

### Consumer

| Setting | Description |
|---------|-------------|
| `auto.offset.reset` | Where to start when no offset exists (`earliest`, `latest`) |
| `enable.auto.commit` | Auto-commit offsets (default: `true`) |
| `session.timeout.ms` | Heartbeat timeout |
| `max.poll.interval.ms` | Max time between polls before rebalance |

## Error Handling

```python
def producer_delivery_report(err, msg):
    if err is not None:
        # Retry logic or dead-letter queue
        print(f"Failed to deliver: {err}")
        # Optionally write to a file or another topic
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}")
```
