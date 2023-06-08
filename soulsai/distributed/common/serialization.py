from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty
from typing import Type
import logging

import numpy as np

import capnp

capnp.remove_import_hook()
logger = logging.getLogger(__name__)


class Serializer(ABC):

    @abstractmethod
    def serialize_sample(self, **kwargs) -> bytes:
        ...

    @abstractmethod
    def deserialize_sample(self, data: bytes) -> dict:
        ...

    @abstractmethod
    def serialize_telemetry(self, **kwargs) -> bytes:
        ...

    @abstractmethod
    def deserialize_telemetry(self, data: bytes) -> dict:
        ...

    @abstractproperty
    def supported_envs(self) -> list:
        ...


class DQNSerializer(Serializer):

    supported_envs = ["SoulsGymIudex-v0", "LunarLander-v2"]

    def __init__(self, env_id: str):
        assert env_id in self.supported_envs
        self.env_id = env_id
        self.capnp_msgs = capnp.load(str(Path(__file__).parent / "data" / f"{env_id}_msgs.capnp"))
        # Load functions for serialization and deserialization
        _env_id = env_id.replace("-", "_")
        self._serialize_sample = getattr(self, f"_serialize_{_env_id}_sample")
        self._deserialize_sample = getattr(self, f"_deserialize_{_env_id}_sample")
        self._serialize_telemetry = getattr(self, f"_serialize_{_env_id}_telemetry")
        self._deserialize_telemetry = getattr(self, f"_deserialize_{_env_id}_telemetry")

    def serialize_sample(self, sample: dict) -> bytes:
        return self._serialize_sample(sample)

    def deserialize_sample(self, data: bytes) -> dict:
        return self._deserialize_sample(data)

    def serialize_telemetry(self, tel: dict) -> bytes:
        return self._serialize_telemetry(tel)

    def deserialize_telemetry(self, data: bytes) -> dict:
        return self._deserialize_telemetry(data)

    def _serialize_SoulsGymIudex_v0_sample(self, sample: dict) -> bytes:
        sample["obs"] = sample["obs"].tolist()
        sample["nextObs"] = sample["nextObs"].tolist()
        return self.capnp_msgs.Sample.new_message(**sample).to_bytes()

    def _deserialize_SoulsGymIudex_v0_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.Sample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])
        return x

    def _serialize_SoulsGymIudex_v0_telemetry(self, tel: dict) -> bytes:
        tel["bossHp"] = float(tel["obs"][2])
        del tel["obs"]
        tel["win"] = bool(tel["bossHp"] == 0)
        return self.capnp_msgs.Telemetry.new_message(**tel).to_bytes()

    def _deserialize_SoulsGymIudex_v0_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Telemetry.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])

    def _serialize_LunarLander_v2_sample(self, sample: dict) -> bytes:
        sample["obs"] = sample["obs"].tolist()
        sample["nextObs"] = sample["nextObs"].tolist()
        sample["reward"] = float(sample["reward"])
        return self.capnp_msgs.Sample.new_message(**sample).to_bytes()

    def _deserialize_LunarLander_v2_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.Sample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])
        return x

    def _serialize_LunarLander_v2_telemetry(self, tel: dict) -> bytes:
        tel["bossHp"] = 0
        del tel["obs"]
        tel["win"] = bool(tel["reward"] > 200)
        tel["reward"] = float(tel["reward"])
        return self.capnp_msgs.Telemetry.new_message(**tel).to_bytes()

    def _deserialize_LunarLander_v2_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Telemetry.from_bytes(data) as sample:
            return sample.to_dict()


def get_serializer_cls(algorithm: str) -> Type[Serializer]:
    if algorithm.lower() == "dqn":
        return DQNSerializer
    else:
        raise NotImplementedError
