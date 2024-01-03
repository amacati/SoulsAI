import time
import logging
import itertools
import queue
import multiprocessing
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

from redis import Redis

logger = logging.getLogger(__name__)


class DQNServerConnector:

    def __init__(self, redis_address, redis_password):
        ctx = multiprocessing.get_context('spawn')
        self._stop_event = ctx.Event()
        self._msg_queue = ctx.Queue()
        args = (self._msg_queue, redis_address, redis_password, self._stop_event)
        self.msg_consumer = ctx.Process(target=self._consume_msgs, args=args, daemon=True)
        self.msg_consumer.start()

    def msgs(self, timeout: float | None = None) -> list[bytes]:
        try:
            data = [self._msg_queue.get(timeout=timeout) for _ in range(self._msg_queue.qsize())]
            return list(itertools.chain.from_iterable(x for x in data if x is not None))
        except queue.Empty:
            logger.warning("empty")
            return []

    def _consume_msgs(self, msg_queue: Queue, redis_address: str, redis_password: str,
                      stop_event: Event):
        red = Redis(host=redis_address, port=6379, password=redis_password)
        last_log_time = time.time()
        msgs = None
        while not stop_event.is_set():
            try:
                # If msgs is not None, we have leftover messages from the last iteration that could
                # not be processed. We try to process them first before getting new messages from
                # Redis.
                msgs = msgs or red.rpop("samples", 10)
                if not msgs:
                    time.sleep(1e-6)
                msg_queue.put_nowait(msgs)
                msgs = None
            except queue.Full:
                if time.time() - last_log_time > 5:
                    logger.warning("Connector queue is full")
                    last_log_time = time.time()
                time.sleep(1e-6)
                continue

    def stop(self):
        self._stop_event.set()
        self.msg_consumer.join()
