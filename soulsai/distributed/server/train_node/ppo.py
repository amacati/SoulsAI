import logging
from types import SimpleNamespace
from uuid import uuid4
from pathlib import Path
from threading import Lock
import json

from redis import Redis

from soulsai.core.agent import PPOAgent
from soulsai.utils import load_redis_secret, mkdir_date

logger = logging.getLogger(__name__)


class PPOTrainingNode:

    def __init__(self, config, decode_sample):
        logger.info("PPO training node startup")
        self.config = config
        self.decode_sample = decode_sample
        self._shutdown = False
        # Read redis server secret
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = Redis(host='redis', port=6379, password=secret, db=0, decode_responses=True)
        self.sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub = self.red.pubsub(ignore_subscribe_messages=True)
        self.cmd_sub.psubscribe(manual_save=self.quicksave, shutdown=self.shutdown)
        self.cmd_sub.run_in_thread(sleep_time=1., daemon=True)
        self.lock = Lock()

        # Create unique directory
        save_root_dir = Path(__file__).parents[4] / "saves"
        save_root_dir.mkdir(exist_ok=True)
        self.save_dir = mkdir_date(save_root_dir)

        self.sub.subscribe("samples")
        self.model_cnt = 0  # Track number of model iterations for checkpoint trigger

        self.agent = PPOAgent(self.config.network_type, self.config.network_kwargs, self.config.lr,
                              self.config.gamma, self.config.dqn_multistep, self.config.grad_clip,
                              self.config.q_clip)
        self.model_id = str(uuid4())
        self.agent.model_id = self.model_id
        logger.info(f"Initial model ID: {self.model_id}")
        self.buffer = TrajectoryBuffer(self.config.n)
        if self.config.load_checkpoint(save_root_dir / "checkpoint"):
            logger.info("Checkpoint loading complete")
        else:
            self.checkpoint(self.save_dir)  # Make config accessible for sanity checking
        self.config.save_dir = self.save_dir.name
        # Upload config to redis to share with client and telemetry node
        logger.info("Saving config to redis for synchronization")
        self.red.set("config", json.dumps(vars(self.config)))
        self.push_model_update()
        logger.info("Initial model upload successful, startup complete")

    def run(self):
        # Discover clients
        logger.info("Starting discovery phase")
        self.discover_clients()
        logger.info("Discovery complete, starting training")
        while not self._shutdown:
            msg = self.sub.get_message(0.01)
            if not msg:
                continue
            sample = json.loads(msg["data"])
            if not sample.get("model_id") == self.model_id:
                continue
            node_id, sample_id = sample.get("node_id"), sample.get("sample_id")
            sample = self.decode_sample(sample)
            self.buffer.append(vec_id=node_id, sample_id = sample_id)
            if self.buffer.complete():
                with self.lock:
                    self._model_update()
                logger.info("Model update complete")
        logger.info("Training node has shut down")

    def _model_update(self):
        ...

    def _client_watchdog(self):
        ...

    def checkpoint(self, path):
        logger.info("Checkpointing...")
        path.mkdir(exist_ok=True)
        self.agent.save(path)
        with open(path / "config.json", "w")  as f:
            json.dump(vars(self.config), f)
        logger.info("Checkpoint finished")

    def load_checkpoint(self, path):
        self.agent.load(path)
        if self.config.load_checkpoint_config:
            with open(path / "config.json", "r") as f:
                saved_config = SimpleNamespace(**json.load(f))
            assert saved_config.env == self.config.env, "Config environments do not match"
            assert saved_config.algorithm == self.config.algorithm, "Config algorithms do not match"
            self.config = saved_config

    def quicksave(self, _):
        with self.lock:
            self.checkpoint(self.save_dir)

    def shutdown(self, _):
        logger.info("Shutdown signaled")
        self._shutdown = True
