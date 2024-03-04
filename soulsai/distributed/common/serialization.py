"""Serialization and deserialization module for exchanging data in tensordicts across nodes.

For simplicity, all data that is serialized must be of type `TensorDict`. This allows us to easily
serialize and deserialize the data using PyTorch's `torch.save` and `torch.load` functions in
combination with `io.BytesIO` to convert the data to and from bytes.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict


def serialize(data: TensorDict) -> bytes:
    """Serialize the data into a byte string.

    Args:
        data: Data to be serialized.

    Returns:
        Byte string representation of the data.
    """
    buff = io.BytesIO()
    torch.save(data, buff)
    buff.seek(0)
    return buff.read()


def deserialize(data: bytes) -> TensorDict:
    """Deserialize the data from a byte string.

    Args:
        data: Byte string representation of the data.

    Returns:
        Deserialized data as a TensorDict.
    """
    assert isinstance(data, bytes), f"Expected bytes, got {type(data)}."
    buff = io.BytesIO(data)
    buff.seek(0)
    return torch.load(buff)
