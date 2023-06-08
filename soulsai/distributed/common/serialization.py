from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty
import numpy as np

import capnp

capnp.remove_import_hook()


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

    supported_envs = ["SoulsGymIudex-v0"]

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

    def serialize_sample(self, **kwargs) -> bytes:
        return self._serialize_sample(**kwargs)

    def deserialize_sample(self, data: bytes) -> dict:
        return self._deserialize_sample(data)

    def serialize_telemetry(self, **kwargs) -> bytes:
        return self._serialize_telemetry(**kwargs)

    def deserialize_telemetry(self, data: bytes) -> dict:
        return self._deserialize_telemetry(data)

    def _serialize_SoulsGymIudex_v0_sample(self, **kwargs) -> bytes:
        kwargs["obs"] = kwargs["obs"].tolist()
        kwargs["nextObs"] = kwargs["nextObs"].tolist()
        return self.capnp_msgs.Sample.new_message(**kwargs).to_bytes()

    def _deserialize_SoulsGymIudex_v0_sample(self, data: bytes) -> dict:
        with self.capnp_msgs.Sample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])
        return x["obs"], x["action"], x["reward"], x["nextObs"], x["done"], x["info"]

    def _serialize_SoulsGymIudex_v0_telemetry(self, **kwargs) -> bytes:
        kwargs["bossHp"] = float(kwargs["obs"][2])
        del kwargs["obs"]
        kwargs["win"] = bool(kwargs["bossHp"] == 0)
        return self.capnp_msgs.Sample.new_message(**kwargs).to_bytes()

    def _deserialize_SoulsGymIudex_v0_telemetry(self, data: bytes) -> dict:
        with self.capnp_msgs.Sample.from_bytes(data) as sample:
            x = sample.to_dict()
        x["obs"] = np.array(x["obs"])
        x["nextObs"] = np.array(x["nextObs"])
