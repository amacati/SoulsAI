"""Connector module for the DQNServer to receive messages from Redis while training.

When training a DQN model, the DQNServer alternates between training and receiving new samples from
the Redis server. To interleave these two tasks, the DQNServerConnector acts as a bridge between
the DQNServer and the Redis server. It asynchronously consumes messages from the Redis server and
puts them into an internal queue. The messages can then be retrieved from the queue using the `msgs`
method.

After a training step, the DQNServer does not have to wait for new samples to download from Redis,
but fetches them from the internal queue instead, which should be much faster.
"""
from __future__ import annotations

import time
import logging
import itertools
import queue
import multiprocessing
from typing import TYPE_CHECKING

from redis import Redis

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

logger = logging.getLogger(__name__)


class DQNServerConnector:
    """Connector to asynchronously receive messages from Redis.

    The connector continuously polls Redis for new samples. Once a sample is available, it is put
    into an internal queue. The messages can then be retrieved from the queue using the `msgs`
    method.
    """

    def __init__(self, redis_address: str, redis_password: str):
        """Create a new connector to asynchronously receive messages from Redis.

        Args:
            redis_address: The address of the Redis server.
            redis_password: The password of the Redis server.
        """
        ctx = multiprocessing.get_context('spawn')
        self._stop_event = ctx.Event()
        self._msg_queue = ctx.Queue()
        args = (self._msg_queue, redis_address, redis_password, self._stop_event)
        self.msg_consumer = ctx.Process(target=self._consume_msgs, args=args, daemon=True)
        self.msg_consumer.start()

    def msgs(self, timeout: float | None = None) -> list[bytes]:
        """Get messages from the connector.

        Args:
            timeout: The maximum time to wait for messages. If None, the method will wait
                indefinitely until messages are available.

        Returns:
            A list of messages from the connector. If no messages are available, an empty list is
            returned.
        """
        try:
            data = [self._msg_queue.get(timeout=timeout) for _ in range(self._msg_queue.qsize())]
            return list(itertools.chain.from_iterable(x for x in data if x is not None))
        except queue.Empty:
            logger.warning("empty")
            return []

    def _consume_msgs(self, msg_queue: Queue, redis_address: str, redis_password: str,
                      stop_event: Event):
        """Consume messages from Redis and put them into the queue.

        Args:
            msg_queue: The internal queue to put the messages into.
            redis_address: The address of the Redis server.
            redis_password: The password of the Redis server.
            stop_event: The event to signal the consumer to stop.
        """
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
        """Stop the connector."""
        self._stop_event.set()
        self.msg_consumer.join()
