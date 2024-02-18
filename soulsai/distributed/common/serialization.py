import io
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict


def serialize(data: TensorDict) -> bytes:
    buff = io.BytesIO()
    torch.save(data, buff)
    buff.seek(0)
    return buff.read()


def deserialize(data: bytes) -> TensorDict:
    buff = io.BytesIO(data)
    buff.seek(0)
    return torch.load(buff)
