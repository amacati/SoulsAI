import logging
from collections import deque
from uuid import uuid4

logger = logging.getLogger(__name__)


def connect_to_server():
    logger.info("Connected to DummyServer")
    return DummyServer()


class DummyServer:

    def __init__(self):
        self.sample_deque = deque(maxlen=10)
        self.sample_cnt = 0
        self.model_id = uuid4()
        self.sub_msgs = [True, self.model_id]

    def push_sample(self, sample):
        if not self._check_sample(sample):
            return
        self.sample_cnt += 1
        self.sample_deque.append(sample)
        if self.sample_cnt == 10:
            self.model_update()
            self.sample_cnt = 0

    def get_model(self):
        return {"id": self.model_id, "model": None}

    def model_update(self):
        logger.info("Training model")
        ...
        self.model_id = uuid4()
        self.push_model_update()

    def push_model_update(self):
        logger.info("Pushing new model")
        self.sub_msgs[0] = True
        self.sub_msgs[1] = self.model_id

    def _check_sample(self, sample):
        if sample["model_id"] == self.model_id:
            logging.info("Sample ID accepted")
            return True
        logging.info("Sample ID rejected")
        return False
