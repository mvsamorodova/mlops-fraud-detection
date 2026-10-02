import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from confluent_kafka import Consumer, Producer, KafkaException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import run_preproc
from scorer import make_pred

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"
)
TRANSACTIONS_TOPIC = os.getenv(
    "KAFKA_TRANSACTIONS_TOPIC", "transactions"
)
SCORING_TOPIC = os.getenv(
    "KAFKA_SCORING_TOPIC", "scores"
)


class ProcessingService:
    def __init__(self):
        self.preprocessor = joblib.load(
            ROOT / "models" / "preprocessor.joblib"
        )

        self.consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "ml-scorer",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        })
        self.consumer.subscribe([TRANSACTIONS_TOPIC])

        self.producer = Producer({
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "acks": "all",
        })

    def process_messages(self):
        try:
            while True:
                msg = self.consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    raise KafkaException(msg.error())

                data = json.loads(msg.value().decode("utf-8"))
                transaction_id = data["transaction_id"]
                input_df = pd.DataFrame([data["data"]])

                processed_df = run_preproc(self.preprocessor, input_df)
                prediction = make_pred(processed_df).iloc[0]

                result = {
                    "transaction_id": transaction_id,
                    "score": float(prediction["score"]),
                    "fraud_flag": int(prediction["fraud_flag"]),
                }

                delivery_errors = []

                def on_delivery(error, message):
                    if error is not None:
                        delivery_errors.append(str(error))

                self.producer.produce(
                    SCORING_TOPIC,
                    key=str(transaction_id),
                    value=json.dumps(result, allow_nan=False),
                    callback=on_delivery,
                )

                remaining = self.producer.flush(10)

                if remaining or delivery_errors:
                    raise RuntimeError(
                        f"Ошибка отправки в Kafka: {delivery_errors}; "
                        f"неотправленных сообщений: {remaining}"
                    )

                # Подтверждаем входное сообщение после отправки результата.
                self.consumer.commit(message=msg, asynchronous=False)
                logger.info("Transaction scored: %s", transaction_id)

        except KeyboardInterrupt:
            logger.info("Service stopped")
        except Exception:
            logger.exception("Scoring service failed")
            raise
        finally:
            self.producer.flush(5)
            self.consumer.close()


if __name__ == "__main__":
    logger.info("Starting Kafka ML scoring service")
    ProcessingService().process_messages()