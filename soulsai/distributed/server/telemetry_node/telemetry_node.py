import logging
import json
from pathlib import Path
import time
import tempfile
import os
from threading import Lock
temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name  # Set matplotlib config dir 

import redis
from prometheus_client import start_http_server

from soulsai.utils import load_redis_secret, load_remote_config
from soulsai.utils.visualization import save_plots
from soulsai.distributed.server.telemetry_node.grafana_connector import GrafanaConnector

logger = logging.getLogger(__name__)


class TelemetryNode:

    stats = ["rewards", "rewards_av", "steps", "steps_av", "boss_hp", "boss_hp_av", "wins",
             "wins_av", "eps"]

    def __init__(self, config):
        logger.info("Telemetry node startup")
        # Read redis server secret
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = redis.Redis(host='redis', port=6379, password=secret, db=0,
                               decode_responses=True)
        self.sub_telemetry = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub_telemetry.subscribe("telemetry")
        self.config = load_remote_config(config.redis_address, secret)
        self.lock = Lock()

        self._shutdown = False
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.subscribe(shutdown=self.shutdown)
        self.cmd_sub.run_in_thread(sleep_time=1., daemon=True)

        self.rewards = []
        self.rewards_av = []
        self.steps = []
        self.steps_av = []
        self.boss_hp = []
        self.boss_hp_av = []
        self.wins = []
        self.wins_av = []
        self.eps = []
        
        self._best_reward = float("-inf")

        if self.config.monitoring.enable:
            logger.info("Starting Grafana connector server for live monitoring")
            self.grafana_con = GrafanaConnector(data_lock=self.lock)
            for stat in self.stats:
                self.grafana_con.data[stat] = getattr(self, stat)
            self.grafana_con.run()
            start_http_server(port=8080)
        else:
            logger.info("Skipping live monitoring")

        save_dir = Path(__file__).parents[4] / "saves" / self.config.save_dir
        self.figure_path = save_dir / "SoulsAIDashboard.png"
        self.stats_path = save_dir / "SoulsAIStats.json"

        if self.config.load_checkpoint:
            self._load_stats()
        logger.info("Telemetry node startup complete")

    def run(self):
        logger.info("Telemetry node running")
        while not self._shutdown:
            # read new samples
            msg = self.sub_telemetry.get_message()
            if not msg:
                time.sleep(1)
                continue
            sample = json.loads(msg["data"])
            # Appending automatically changes data in GrafanaConnector
            with self.lock:
                self.rewards.append(sample["reward"])
                self.rewards_av.append(self._latest_moving_av(self.rewards))
                self.steps.append(sample["steps"])
                self.steps_av.append(self._latest_moving_av(self.steps))
                self.boss_hp.append(sample["boss_hp"])
                self.boss_hp_av.append(self._latest_moving_av(self.boss_hp))
                self.wins.append(int(sample["win"]))
                self.wins_av.append(self._latest_moving_av(self.wins))
                self.eps.append(sample["eps"])
            n_rewards = len(self.rewards)
            if n_rewards % self.config.telemetry.update_interval == 0:
                self.update_stats_and_dashboard()
                logger.info((f"Dashboard updated, last av. reward: {self.rewards_av[-1]:.1f}"
                             f", last av. steps: {self.steps_av[-1]:.0f}"))
            if n_rewards % self.config.telemetry.save_best_interval == 0:
                if self.rewards_av[-1] > self._best_reward:
                    self.red.publish("save_best", "")
                    self._best_reward = self.rewards_av[-1]

    def update_stats_and_dashboard(self):
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        save_plots(self.rewards, self.steps, self.boss_hp, self.wins, self.figure_path, self.eps,
                   self.config.telemetry.moving_average)
        self._save_stats(self.stats_path)

    def _save_stats(self, path):
        with open(path, "w") as f:
            json.dump({stat: getattr(self, stat) for stat in self.stats}, f)

    def _load_stats(self, path):
        path = Path(__file__).parents[4] / "saves" / "checkpoint" / "SoulsAIStats.json"
        if path.exists() and path.is_file():
            with open(path, "r") as f:
                stats = json.load(f)
            for stat in stats:
                setattr(self, stat, stats.get(stat))
        else:
            logger.warning("Loading from checkpoint, but no previous telemetry found.")

    def shutdown(self, _):
        logger.info("Shutdown signaled")
        self._shutdown = True

    def _latest_moving_av(self, x):
        view = x[-self.config.telemetry.moving_average:]
        return sum(view) / len(view)  # len(view) can be smaller than N
