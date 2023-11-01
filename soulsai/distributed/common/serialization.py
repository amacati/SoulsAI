from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty
import logging

import numpy as np

import capnp

capnp.remove_import_hook()
logger = logging.getLogger(__name__)


class Serializer(ABC):

    @abstractmethod
    def serialize_sample(self, **kwargs: dict) -> bytes:
        ...

    @abstractmethod
    def deserialize_sample(self, data: bytes) -> dict:
        ...

    @abstractmethod
    def serialize_episode_info(self, **kwargs: dict) -> bytes:
        ...

    @abstractmethod
    def deserialize_episode_info(self, data: bytes) -> dict:
        ...

    @abstractmethod
    def serialize_telemetry(self, **kwargs: dict) -> bytes:
        ...

    @abstractmethod
    def deserialize_telemetry(self, data: bytes) -> dict:
        ...

    @abstractproperty
    def supported_envs(self) -> list:
        ...


class DQNSerializer(Serializer):

    supported_envs = ["SoulsGymIudex-v0", "SoulsGymIudexImg-v0", "LunarLander-v2", "ALE/Pong-v5"]

    def __init__(self, env_id: str):
        assert env_id in self.supported_envs
        self.env_id = env_id
        _env_id = env_id.replace("-", "_").replace("/", "_")
        self.capnp_msgs = capnp.load(str(Path(__file__).parent / "data" / f"{_env_id}_msgs.capnp"))
        # Load functions for serialization and deserialization and overwrite defaults
        self._serialize_sample = getattr(self, f"_serialize_{_env_id}_sample")
        self._deserialize_sample = getattr(self, f"_deserialize_{_env_id}_sample")
        self._serialize_episode_info = getattr(self, f"_serialize_{_env_id}_episode_info")
        self._deserialize_episode_info = getattr(self, f"_deserialize_{_env_id}_episode_info")
        self._serialize_telemetry = getattr(self, f"_serialize_{_env_id}_telemetry")
        self._deserialize_telemetry = getattr(self, f"_deserialize_{_env_id}_telemetry")

    def serialize_sample(self, sample: dict) -> bytes:
        return self._serialize_sample(sample)

    def deserialize_sample(self, data: bytes) -> dict:
        return self._deserialize_sample(data)

    def serialize_episode_info(self, data: dict) -> bytes:
        return self._serialize_episode_info(data)

    def deserialize_episode_info(self, data: bytes) -> dict:
        return self._deserialize_episode_info(data)

    def serialize_telemetry(self, tel: dict) -> bytes:
        return self._serialize_telemetry(tel)

    def deserialize_telemetry(self, data: bytes) -> dict:
        return self._deserialize_telemetry(data)

    def _serialize_SoulsGymIudex_v0_sample(self, sample: dict) -> bytes:
        sample["obs"] = sample["obs"].tolist()
        sample["nextObs"] = sample["nextObs"].tolist()
        sample["reward"] = float(sample["reward"])
        sample["info"] = {"allowedActions": sample["info"]["allowed_actions"]}
        return self.capnp_msgs.DQNSample.new_message(**sample).to_bytes()

    def _deserialize_SoulsGymIudex_v0_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.DQNSample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])
        x["info"] = {"allowed_actions": x["info"]["allowedActions"]}
        return x

    def _serialize_SoulsGymIudex_v0_episode_info(self, data: dict) -> bytes:
        return self.capnp_msgs.EpisodeInfo.new_message(**data).to_bytes()

    def _deserialize_SoulsGymIudex_v0_episode_info(self, data: bytes) -> dict:
        with self.capnp_msgs.EpisodeInfo.from_bytes(data) as episode_info:
            return episode_info.to_dict()

    def _serialize_SoulsGymIudex_v0_telemetry(self, tel: dict) -> bytes:
        tel["bossHp"] = float(tel["obs"][2])
        del tel["obs"]
        tel["win"] = bool(tel["bossHp"] == 0)
        tel["reward"] = float(tel["reward"])
        return self.capnp_msgs.Telemetry.new_message(**tel).to_bytes()

    def _deserialize_SoulsGymIudex_v0_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Telemetry.from_bytes(data) as sample:
            return sample.to_dict()

    def _serialize_SoulsGymIudexImg_v0_sample(self, sample: dict) -> bytes:
        sample["obs"] = sample["obs"].tolist()
        sample["nextObs"] = sample["nextObs"].tolist()
        sample["reward"] = float(sample["reward"])
        sample["info"] = {"allowedActions": sample["info"]["allowed_actions"]}
        return self.capnp_msgs.DQNSample.new_message(**sample).to_bytes()

    def _deserialize_SoulsGymIudexImg_v0_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.DQNSample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])
        x["info"] = {"allowed_actions": x["info"]["allowedActions"]}
        return x

    def _serialize_SoulsGymIudexImg_v0_episode_info(self, data: dict) -> bytes:
        return self.capnp_msgs.EpisodeInfo.new_message(**data).to_bytes()

    def _deserialize_SoulsGymIudexImg_v0_episode_info(self, data: bytes) -> dict:
        with self.capnp_msgs.EpisodeInfo.from_bytes(data) as episode_info:
            return episode_info.to_dict()

    def _serialize_SoulsGymIudexImg_v0_telemetry(self, tel: dict) -> bytes:
        msg = {
            "bossHp": float(tel["info"]["boss_hp"]) / 1034,  # 1034 is max boss hp
            "win": bool(tel["info"]["boss_hp"] == 0),
            "reward": float(tel["reward"]),
            "steps": int(tel["steps"]),
            "eps": float(tel["eps"])
        }
        return self.capnp_msgs.Telemetry.new_message(**msg).to_bytes()

    def _deserialize_SoulsGymIudexImg_v0_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Telemetry.from_bytes(data) as sample:
            return sample.to_dict()

    def _serialize_LunarLander_v2_sample(self, sample: dict) -> bytes:
        sample["obs"] = sample["obs"].tolist()
        sample["nextObs"] = sample["nextObs"].tolist()
        sample["reward"] = float(sample["reward"])
        return self.capnp_msgs.DQNSample.new_message(**sample).to_bytes()

    def _deserialize_LunarLander_v2_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.DQNSample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])
        return x

    def _serialize_LunarLander_v2_episode_info(self, data: dict) -> bytes:
        msg = {
            "bossHp": 0,
            "win": bool(data["epReward"] > 200),
            "epReward": float(data["epReward"]),
            "epSteps": data["epSteps"],
            "eps": float(data["eps"]),
            "modelId": data["modelId"]
        }
        return self.capnp_msgs.EpisodeInfo.new_message(**msg).to_bytes()

    def _deserialize_LunarLander_v2_episode_info(self, data: bytes) -> dict:
        with self.capnp_msgs.EpisodeInfo.from_bytes(data) as episode_info:
            return episode_info.to_dict()

    def _serialize_LunarLander_v2_telemetry(self, tel: dict) -> bytes:
        msg = {
            "bossHp": 0,
            "win": bool(tel["epReward"] > 200),
            "epReward": float(tel["epReward"]),
            "epSteps": int(tel["epSteps"]),
            "totalSteps": int(tel["totalSteps"]),
            "eps": float(tel["eps"])
        }
        return self.capnp_msgs.Telemetry.new_message(**msg).to_bytes()

    def _deserialize_LunarLander_v2_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Telemetry.from_bytes(data) as sample:
            return sample.to_dict()

    def _serialize_ALE_Pong_v5_sample(self, sample: dict) -> bytes:
        sample["obs"] = sample["obs"].tolist()
        sample["nextObs"] = sample["nextObs"].tolist()
        sample["reward"] = float(sample["reward"])
        sample["info"] = {}
        return self.capnp_msgs.DQNSample.new_message(**sample).to_bytes()

    def _deserialize_ALE_Pong_v5_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.DQNSample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"], np.uint8)
        x["nextObs"] = np.array(x["nextObs"], np.uint8)
        return x

    def _serialize_ALE_Pong_v5_episode_info(self, data: dict) -> bytes:
        msg = {
            "epReward": data["epReward"],
            "epSteps": data["epSteps"],
            "bossHp": 0,
            "win": data["epReward"] > 0,
            "eps": data["eps"],
            "modelId": data["modelId"]
        }
        return self.capnp_msgs.EpisodeInfo.new_message(**msg).to_bytes()

    def _deserialize_ALE_Pong_v5_episode_info(self, data: bytes) -> dict:
        with self.capnp_msgs.EpisodeInfo.from_bytes(data) as episode_info:
            return episode_info.to_dict()

    def _serialize_ALE_Pong_v5_telemetry(self, tel: dict) -> bytes:
        return self.capnp_msgs.Telemetry.new_message(**tel).to_bytes()

    def _deserialize_ALE_Pong_v5_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Telemetry.from_bytes(data) as sample:
            return sample.to_dict()


class PPOSerializer(Serializer):

    supported_envs = ["LunarLander-v2"]

    def __init__(self, env_id: str):
        assert env_id in self.supported_envs
        self.env_id = env_id
        _env_id = env_id.replace("-", "_").replace("/", "_")
        self.capnp_msgs = capnp.load(str(Path(__file__).parent / "data" / f"{_env_id}_msgs.capnp"))
        # Load functions for serialization and deserialization
        self._serialize_sample = getattr(self, f"_serialize_{_env_id}_sample")
        self._deserialize_sample = getattr(self, f"_deserialize_{_env_id}_sample")
        self._serialize_episode_info = getattr(self, f"_serialize_{_env_id}_episode_info")
        self._deserialize_episode_info = getattr(self, f"_deserialize_{_env_id}_episode_info")
        self._serialize_telemetry = getattr(self, f"_serialize_{_env_id}_telemetry")
        self._deserialize_telemetry = getattr(self, f"_deserialize_{_env_id}_telemetry")

    def serialize_sample(self, sample: dict) -> bytes:
        return self._serialize_sample(sample)

    def deserialize_sample(self, data: bytes) -> dict:
        return self._deserialize_sample(data)

    def serialize_episode_info(self, episode_info: dict) -> bytes:
        return self._serialize_episode_info(episode_info)

    def deserialize_episode_info(self, data: bytes) -> dict:
        return self._deserialize_episode_info(data)

    def serialize_telemetry(self, tel: dict) -> bytes:
        return self._serialize_telemetry(tel)

    def deserialize_telemetry(self, data: bytes) -> dict:
        return self._deserialize_telemetry(data)

    def _serialize_LunarLander_v2_sample(self, sample: dict) -> bytes:
        sample["obs"] = sample["obs"].tolist()
        sample["reward"] = float(sample["reward"])
        sample["prob"] = float(sample["prob"])
        return self.capnp_msgs.PPOSample.new_message(**sample).to_bytes()

    def _deserialize_LunarLander_v2_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.PPOSample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        return x

    def _serialize_LunarLander_v2_episode_info(self, episode_info: dict) -> bytes:
        episode_info["bossHp"] = 0
        del episode_info["obs"]
        episode_info["win"] = bool(episode_info["epReward"] > 200)
        episode_info["epReward"] = float(episode_info["epReward"])
        return self.capnp_msgs.EpisodeInfo.new_message(**episode_info).to_bytes()

    def _deserialize_LunarLander_v2_episode_info(self, data: bytes) -> dict:
        with self.capnp_msgs.EpisodeInfo.from_bytes(data) as sample:
            return sample.to_dict()

    def _serialize_LunarLander_v2_telemetry(self, tel: dict) -> bytes:
        return self.capnp_msgs.Telemetry.new_message(**tel).to_bytes()

    def _deserialize_LunarLander_v2_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Telemetry.from_bytes(data) as sample:
            return sample.to_dict()


def get_serializer_cls(algorithm: str) -> type[Serializer]:
    if algorithm.lower() == "dqn":
        return DQNSerializer
    elif algorithm.lower() == "ppo":
        return PPOSerializer
    raise NotImplementedError
