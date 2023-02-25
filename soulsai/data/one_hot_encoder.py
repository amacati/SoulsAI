"""The ``one_hot_encoder`` module contains a lightweight implementation of a one-hot encoder.

We include it to avoid the dependency on heavy frameworks like ``sklearn`` for a trivial task.
"""
from typing import Iterable, Hashable
import logging

import numpy as np

logger = logging.getLogger(__name__)


class OneHotEncoder:
    """Encode categorical data as one hot numpy arrays.

    Just like the sklearn encoder (which this class imitates), the encoder first has to be fit to
    data before it can be used to convert between the representations.
    """

    def __init__(self, allow_unknown: bool = False):
        """Initialize the lookup dictionaries.

        Args:
            allow_unknown: Flag to allow unknown categories.
        """
        self._key_to_index_dict = dict()
        self._index_to_key_dict = dict()
        self.dim = None
        self.allow_unknown = allow_unknown

    def fit(self, data: Iterable[Hashable]):
        """Fit the encoder to the training data.

        Args:
            data: An iterable of hashable categories
        """
        for idx, key in enumerate(data):
            self._key_to_index_dict[key] = idx
            self._index_to_key_dict[idx] = key
        self.dim = len(data)

    def __call__(self, data: Hashable) -> np.ndarray:
        """Alias for :meth:`.OneHotEncoder.transform`.

        Args:
            data: A categorical data sample.

        Returns:
            The corresponding one hot encoded array.
        """
        return self.transform(data)

    def transform(self, data: Hashable) -> np.ndarray:
        """Transform categorical data to a one hot encoded array.

        Args:
            data: A categorical data sample.

        Returns:
            The corresponding one hot encoded array.

        Raises:
            ValueError: An unknown category was provided without setting allow_unknown to `True`.
        """
        if data not in self._key_to_index_dict.keys():
            if not self.allow_unknown:
                raise ValueError("OneHotEncoder received an unknown category.")
            logger.warning(f"Unknown key {data} encountered")
            return np.zeros(self.dim, dtype=np.float32)
        x = np.zeros(self.dim, dtype=np.float32)
        x[self._key_to_index_dict[data]] = 1
        return x

    def inverse_transform(self, data: np.ndarray) -> Hashable:
        """Transform one-hot encoded data to its corresponding category.

        Args:
            data: A one-hot encoded data sample.

        Returns:
            The corresponding category.

        Raises:
            ValueError: An unknown category was provided.
        """
        key = np.where(data == 1)
        if not len(key[0]) == 1 or len(data) != self.dim:
            raise ValueError("OneHotEncoder received an unknown category.")
        return self._index_to_key_dict[key[0][0]]
