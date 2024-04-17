"""The ``watchdog`` module allows the execution of functions under special surveillance."""

from __future__ import annotations

import logging
import time
from multiprocessing import Value, Process
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

logger = logging.getLogger(__name__)


class ClientWatchdog:
    """Watchdog to surveil the client sampling function.

    The watchdog starts an observation thread that periodically checks if the client's main script
    is still running as expected. If this is not the case, it restarts the script.
    """

    def __init__(self, watched_fn: Callable, minimum_samples_per_minute: int, external_args: Any):
        """Initialize the shared events and gauges.

        Args:
            watched_fn: The surveilled function.
            minimum_samples_per_minute: The minimum expected samples per minute.
            external_args: The external arguments used to call ``watched_fn``.
        """
        self.sample_gauge = Value("i", 1_000_000)
        self.minimum_samples_per_minute = minimum_samples_per_minute
        self._watched_fn = watched_fn
        self._fn_shutdown = Event()
        self._watchdog_fn_shutdown = Event()
        self._external_args = external_args
        self.shutdown = Event()
        self.watchdog_thread = Thread(target=self._watchdog, daemon=True)

    def start(self):
        """Start the watchdog thread and execute the watched function.

        Restart the function in case the watchdog thread determined a failure. If the function exits
        nominally without intervention from the watchdog thread, the watchdog shuts down.
        """
        logger.info("Watchdog startup")
        self.watchdog_thread.start()
        while not self.shutdown.is_set():
            try:
                kwargs = {"stop_flag": self._fn_shutdown, "sample_gauge": self.sample_gauge}
                p = Process(target=self._watched_fn, args=self._external_args, kwargs=kwargs)
                p.start()
                p.join()
            except Exception as e:
                logger.info(f"{type(e).__name__}: {e}")
                self._fn_shutdown.clear()
                time.sleep(30)  # Give the game time to reset
                continue
            if self._watchdog_fn_shutdown.is_set():  # Function was reset by watchdog, restart
                self._watchdog_fn_shutdown.clear()
                self._fn_shutdown.clear()
                time.sleep(30)
                continue
            self.shutdown.set()  # Function ended execution nominally, shut watchdog down
        logger.info("Watchdog shutdown successful")

    def _watchdog(self):
        """Watch the sample gauge and set the shutdown flag if it drops below the minimum value."""
        while not self.shutdown.is_set():
            # Check performance metric
            if self.sample_gauge.value < self.minimum_samples_per_minute:
                logger.warning(
                    (
                        f"Current sample count {self.sample_gauge.value}/m is less than"
                        " required. Resetting training"
                    )
                )
                self.sample_gauge.value = 1_000_000
                self._watchdog_fn_shutdown.set()
                self._fn_shutdown.set()
            else:
                logger.debug(f"Watchdog check passed ({self.sample_gauge.value} samples/min)")
            time.sleep(10.0)


class WatchdogGauge:
    """A simple gauge to measure the sample rate."""

    def __init__(self, sync_value: Synchronized, update_time: float = 60.0):
        """Create a wrapper around the synchronized shared value.

        Args:
            sync_value: The shared value to store the current sample rate.
            update_time: The time interval in seconds in which the sample rate is updated.
        """
        self.sync_value = sync_value
        self._cnt = -1
        self._t_last = 0
        self._update_time = update_time

    def inc(self, amount: int = 1):
        """Increment the gauge by a given amount.

        On each increment, the gauge checks if more than `update_time` seconds have passed since the
        last time update. If so, it updates the shared value with the current sample rate and resets
        the counter and timer.

        Args:
            amount: The amount to increment the gauge by.
        """
        if self._cnt == -1:
            self._cnt = amount
            self._t_start = time.time()
            return
        self._cnt += amount
        t_now = time.time()
        if t_now - self._t_start > self._update_time:
            self.sync_value.value = int(self._cnt * 60 / (t_now - self._t_start))
            self._cnt = 0
            self._t_start = t_now
