import time
from threading import Thread, Event
import logging
from multiprocessing import Value

logger = logging.getLogger(__name__)


class ClientWatchdog:

    def __init__(self, watched_fn, minimum_samples_per_minute, external_args):
        self.sample_gauge = Value("i", 1_000_000)
        self.minimum_samples_per_minute = minimum_samples_per_minute
        self._watched_fn = watched_fn
        self._fn_shutdown = Event()
        self._watchdog_fn_shutdown = Event()
        self._external_args = external_args
        self.shutdown = Event()
        self.watchdog_thread = Thread(target=self.watchdog, daemon=True)

    def start(self):
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

    def watchdog(self):
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
