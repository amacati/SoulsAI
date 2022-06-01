import logging
import json
from pathlib import Path
from collections import deque
from threading import Lock

from matplotlib import pyplot as plt
import numpy as np
import redis
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials

from soulsai.utils.visualization import save_plots

logger = logging.getLogger(__name__)


class TelemetryNode:

    GSA_SCOPES = ["https://www.googleapis.com/auth/drive"]
    GSA_SECRET = "/home/SoulsAI/soulsai/distributed/server/telemetry_node/gsa.secret"
    GDRIVE_FOLDER = "1DLtsqv3fUGIMj4moLfGAk7I8QYqh8RIb"

    def __init__(self):
        logger.info("Telemetry node startup")
        # Read redis server secret
        with open(Path(__file__).parents[1] / "redis.secret") as f:
            conf = f.readlines()
        secret = None
        for line in conf:
            if len(line) > 12 and line[0:12] == "requirepass ":
                secret = line[12:]
                break
        if secret is None:
            raise RuntimeError("Missing password configuration for redis in redis.secret")

        self.red = redis.Redis(host='redis', port=6379, password=secret, db=0,
                               decode_responses=True)

        self.rewards = []
        self.steps = []
        self.boss_hp = []
        self.wins = []

        self.sub_telemetry = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub_telemetry.subscribe("telemetry")
        self.figure_path = Path("/tmp") / "dashboard" / "SoulsAIDashboard.png"
        self.stats_path = Path("/tmp") / "dashboard" / "SoulsAIStats.json"

        logger.info("Authenticating with Google Drive for live telemetry")

        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(self.GSA_SECRET,
                                                                            self.GSA_SCOPES)
            self.gdrive_service = build("drive", 'v3', credentials=credentials)
            self.metadata = {"name": "SoulsAIDashboard.png", "parents": [self.GDRIVE_FOLDER]}
            self.media = MediaFileUpload(f"{str(self.figure_path)}", mimetype="image/png")
            logger.info("Authentication successful")
        except Exception as e:
            self.gdrive_service = None
            logger.warning("Authentication failed")

        logger.info("Telemetry node startup complete")

    def run(self):
        logger.info("Telemetry node running")
        while True:
            # read new samples
            msg = self.sub_telemetry.get_message()
            if not msg:
                continue
            sample = json.loads(msg["data"])

            self.rewards.append(sample["reward"])
            self.steps.append(sample["steps"])
            self.boss_hp.append(sample["boss_hp"])
            self.wins.append(sample["win"])

            if len(self.rewards) % 1 == 0:
                self.update_dashboard()

    def update_dashboard(self):
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        save_plots(self.rewards, self.steps, self.boss_hp, self.wins, self.figure_path)
        with open(self.stats_path, "w") as f:
            json.dump({"rewards": self.rewards, "steps": self.steps, "boss_hp": self.boss_hp,
                       "wins": self.wins}, f)
        if self.gdrive_service is not None:
            self.gdrive_service.files().create(body=self.metadata, media_body=self.media,
                                               fields="id").execute()
            logger.info("Google Drive upload successful")
        logger.info("Dashboard updated")
