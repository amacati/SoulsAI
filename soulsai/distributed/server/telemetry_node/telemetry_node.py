import logging
import json
from pathlib import Path
import time

import redis
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials

from soulsai.utils import load_redis_secret, load_remote_config
from soulsai.utils.visualization import save_plots

logger = logging.getLogger(__name__)


class TelemetryNode:

    GSA_SCOPES = ["https://www.googleapis.com/auth/drive"]
    GSA_SECRET = "/home/SoulsAI/config/gsa.secret"

    def __init__(self, config):
        logger.info("Telemetry node startup")
        # Read redis server secret
        secret = load_redis_secret(Path(__file__).parents[4] / "config" / "redis.secret")
        self.red = redis.Redis(host='redis', port=6379, password=secret, db=0,
                               decode_responses=True)
        self.sub_telemetry = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub_telemetry.subscribe("telemetry")
        self.config = load_remote_config(config.redis_address, secret)

        self.rewards = []
        self.steps = []
        self.boss_hp = []
        self.wins = []
        self.eps = []

        save_dir = Path(__file__).parents[4] / "saves" / self.config.save_dir
        self.figure_path = save_dir / "SoulsAIDashboard.png"
        self.stats_path = save_dir / "SoulsAIStats.json"

        if self.config.load_checkpoint:
            path = Path(__file__).parents[4] / "saves" / "checkpoint" / "SoulsAIStats.json"
            if path.exists() and path.is_file():
                with open(path, "r") as f:
                    stats = json.load(f)
                self.rewards = stats["rewards"]
                self.steps = stats["steps"]
                self.boss_hp = stats["boss_hp"]
                self.wins = stats["wins"]
                self.eps = stats["eps"]
            else:
                logger.warning("Loading from checkpoint, but no previous telemetry found.")

        if self.config.gdrive_sync:
            logger.info("Authenticating with Google Drive for live telemetry")
            try:
                # Set up the Google Drive service and create the Dashboard file if it does not
                # already exist
                credentials = ServiceAccountCredentials.from_json_keyfile_name(self.GSA_SECRET,
                                                                               self.GSA_SCOPES)
                self.gdrive_srv = build("drive", 'v3', credentials=credentials)
                metadata = {"name": "SoulsAIDashboard.png", "parents": [self.config.gdrive_dir]}
                self.update_dashboard(drive_update=False)
                media = MediaFileUpload(f"{str(self.figure_path)}", mimetype="image/png")
                logger.info("Authentication successful")
                # During training the file is only updated, so we have to make sure the file exists
                q = ("name='SoulsAIDashboard.png' and mimeType='image/png'"
                    f"and '{self.config.gdrive_dir}' in parents")
                rsp = self.gdrive_srv.files().list(q=q).execute()
                if rsp["files"]:
                    self.file_id = rsp["files"][0]["id"]
                else:
                    logger.info("Telemetry file does not exist in Drive, creating new file")
                    rsp = self.gdrive_srv.files().create(body=metadata, media_body=media).execute()
                    self.file_id = rsp["id"]
                logger.info("Google Drive initialization complete")
            except Exception as e:
                self.gdrive_srv = None
                logger.warning(e)
                logger.warning("Google Drive initialization failed")
        else:
            logger.info("Skipping cloud sync initialization")
            self.gdrive_srv = None

        logger.info("Telemetry node startup complete")

    def run(self):
        logger.info("Telemetry node running")
        while True:
            # read new samples
            msg = self.sub_telemetry.get_message()
            if not msg:
                time.sleep(1)
                continue
            sample = json.loads(msg["data"])

            self.rewards.append(sample["reward"])
            self.steps.append(sample["steps"])
            self.boss_hp.append(sample["boss_hp"])
            self.wins.append(sample["win"])
            self.eps.append(sample["eps"])

            if len(self.rewards) % self.config.telemetry_epochs == 0:
                self.update_dashboard()

    def update_dashboard(self, drive_update=True):
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        save_plots(self.rewards, self.steps, self.boss_hp, self.wins, self.figure_path, self.eps,
                   self.config.moving_average)
        with open(self.stats_path, "w") as f:
            json.dump({"rewards": self.rewards, "steps": self.steps, "boss_hp": self.boss_hp,
                       "wins": self.wins, "eps": self.eps}, f)
        if self.gdrive_srv is not None and drive_update:
            try:
                media = MediaFileUpload(f"{str(self.figure_path)}", mimetype="image/png")
                self.gdrive_srv.files().update(fileId=self.file_id, media_body=media).execute()
                logger.info("Google Drive upload successful")
            except Exception as e:  # noqa: E722
                logger.warning(f"Dashboard upload to Google Drive failed. Error was: {e}")
        logger.info("Dashboard updated")
