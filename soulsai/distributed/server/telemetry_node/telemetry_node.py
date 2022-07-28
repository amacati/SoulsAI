import logging
import json
from pathlib import Path
import time
import datetime

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
        self.eps = []

        self.sub_telemetry = self.red.pubsub(ignore_subscribe_messages=True)
        self.sub_telemetry.subscribe("telemetry")

        save_dir = Path(__file__).parent / "save" / datetime.now().strftime("%Y_%m_%d_%H_%M")
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            t = 1
            while save_dir.is_dir():
                curr_date_unique = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_({t})"
                save_dir = self.PATH / "backup" / (curr_date_unique)
                t += 1
            self.save_path.mkdir(parents=True, exist_ok=True)
        self.figure_path = save_dir / "SoulsAIDashboard.png"
        self.stats_path = save_dir / "SoulsAIStats.json"

        logger.info("Authenticating with Google Drive for live telemetry")

        # Set up the Google Drive service and create the Dashboard file if it does not already exist
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(self.GSA_SECRET,
                                                                           self.GSA_SCOPES)
            self.gdrive_service = build("drive", 'v3', credentials=credentials)
            metadata = {"name": "SoulsAIDashboard.png", "parents": [self.GDRIVE_FOLDER]}
            self.update_dashboard(drive_update=False)
            media = MediaFileUpload(f"{str(self.figure_path)}", mimetype="image/png")
            logger.info("Authentication successful")
            # During training the file is only updated, so we have to make sure the file exists
            q = ("name='SoulsAIDashboard.png' and mimeType='image/png'"
                 f"and '{self.GDRIVE_FOLDER}' in parents")
            rsp = self.gdrive_service.files().list(q=q).execute()
            if rsp["files"]:
                self.file_id = rsp["files"][0]["id"]
            else:
                logger.info("Telemetry file does not exist in Drive, creating new file")
                rsp = self.gdrive_service.files().create(body=metadata, media_body=media).execute()
                self.file_id = rsp["id"]
            logger.info("Google Drive initialization complete")
        except Exception as e:
            self.gdrive_service = None
            logger.warning(e)
            logger.warning("Google Drive initialization failed")

        logger.info("Telemetry node startup complete")

    def run(self):
        logger.info("Telemetry node running")
        while True:
            # read new samples
            msg = self.sub_telemetry.get_message()
            if not msg:
                time.sleep(5)
                continue
            sample = json.loads(msg["data"])

            self.rewards.append(sample["reward"])
            self.steps.append(sample["steps"])
            self.boss_hp.append(sample["boss_hp"])
            self.wins.append(sample["win"])
            self.eps.append(sample["eps"])

            if len(self.rewards) % 5 == 0:
                self.update_dashboard()

    def update_dashboard(self, drive_update=True):
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        save_plots(self.rewards, self.steps, self.boss_hp, self.wins, self.figure_path, self.eps)
        with open(self.stats_path, "w") as f:
            json.dump({"rewards": self.rewards, "steps": self.steps, "boss_hp": self.boss_hp,
                       "wins": self.wins, "eps": self.eps}, f)
        if self.gdrive_service is not None and drive_update:
            try:
                media = MediaFileUpload(f"{str(self.figure_path)}", mimetype="image/png")
                self.gdrive_service.files().update(fileId=self.file_id, media_body=media).execute()
                logger.info("Google Drive upload successful")
            except Exception as e:  # noqa: E722
                logger.warning(f"Dashboard upload to Google Drive failed. Error was: {e}")
        logger.info("Dashboard updated")
