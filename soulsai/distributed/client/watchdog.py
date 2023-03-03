"""The ``watchdog`` module allows the execution of functions under special surveillance."""
import time
from threading import Thread, Event
import logging
from multiprocessing import Value
from typing import Callable, Any

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
                self._watched_fn(*self._external_args, stop_flag=self._fn_shutdown,
                                 sample_gauge=self.sample_gauge)
            except Exception as e:
                logger.info(e)
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
                logger.warning((f"Current sample count {self.sample_gauge.value}/m is less than"
                                " required. Resetting training"))
                self.sample_gauge.value = 1_000_000
                self._watchdog_fn_shutdown.set()
                self._fn_shutdown.set()
            else:
                logger.debug(f"Watchdog check passed ({self.sample_gauge.value} samples/min)")
            time.sleep(10.)
