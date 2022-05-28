from collections import deque
from uuid import uuid4
import logging
import redis
import json

logger = logging.getLogger(__name__)


class TrainingNode:

    def __init__(self):
        logger.info("Training node startup")
        self.red = redis.Redis(host='localhost', port=6379, db=0, charset="utf-8",
                               decode_responses=True)
        self.sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub.subscribe("samples")
        self.sample_deque = deque(maxlen=10)
        self.sample_cnt = 0
        self.required_training_samples = 10
        self.model_id = str(uuid4())
        self.red.set("model_id", self.model_id)
        logger.info(f"Initial model ID: {self.model_id}")

    def run(self):
        logger.info("Training node running")
        while True:
            msg = self.sub.get_message()
            if not msg:
                continue
            sample = json.loads(msg["data"])
            if not self._check_sample(sample):
                continue
            print(sample.get("sample"))
            self.sample_deque.append(sample.get("sample"))
            self.sample_cnt += 1
            if self.sample_cnt >= self.required_training_samples:
                self.model_update()
                self.sample_cnt = 0

    def model_update(self):
        logger.info("Training model")
        ...
        self.model_id = str(uuid4())
        self.red.set("model_id", self.model_id)
        self.push_model_update()

    def push_model_update(self):
        logger.info("Publishing new model")
        self.red.publish("model_update", self.model_id)
        logger.info("Model update successful")

    def _check_sample(self, sample):
        if sample.get("model_id") == self.model_id:
            logging.info("Sample ID accepted")
            return True
        logging.info("Sample ID rejected")
        return False
