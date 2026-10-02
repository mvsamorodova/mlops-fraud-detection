import json
import logging
import os

import psycopg
from confluent_kafka import Consumer, KafkaException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    consumer = Consumer({
        "bootstrap.servers": os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"
        ),
        "group.id": "postgres-score-writer",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    try:
        with psycopg.connect(
            os.environ["DATABASE_URL"],
            connect_timeout=10,
        ) as connection:
            consumer.subscribe([
                os.getenv("KAFKA_SCORING_TOPIC", "scores")
            ])
            logger.info("Score writer started")

            while True:
                message = consumer.poll(1.0)

                if message is None:
                    continue
                if message.error():
                    raise KafkaException(message.error())

                result = json.loads(
                    message.value().decode("utf-8")
                )

                connection.execute(
                    """
                    INSERT INTO transaction_scores (
                        transaction_id, score, fraud_flag
                    )
                    VALUES (%s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING
                    """,
                    (
                        result["transaction_id"],
                        float(result["score"]),
                        int(result["fraud_flag"]),
                    ),
                )
                connection.commit()

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                logger.info(
                    "Saved transaction: %s",
                    result["transaction_id"],
                )

    except KeyboardInterrupt:
        logger.info("Score writer stopped")
    except Exception:
        logger.exception("Score writer failed")
        raise
    finally:
        consumer.close()


if __name__ == "__main__":
    main()