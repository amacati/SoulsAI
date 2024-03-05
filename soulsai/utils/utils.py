"""The ``utils`` module contains various utility functions for conversions and config handling."""
import json
from types import SimpleNamespace
from typing import List, NewType, Callable
import logging
from pathlib import Path
from datetime import datetime
import time
import copy
import sys

import numpy as np
import yaml
import redis

from soulsai.exception import InvalidConfigError, MissingConfigError

logger = logging.getLogger(__name__)

RedisType = NewType("RedisType", redis.Redis)  # Avoid RtD error with Redis | None type


def running_mean(x: List, N: int) -> np.ndarray:
    """Compute the running mean of a list with a sliding window.

    The first N-1 values are left as is, since the sliding window does not have sufficient values.

    Args:
        x: A list of numeric values.
        N: The size of the sliding window.

    Returns:
        An array of the running mean.
    """
    y = np.copy(x)
    if len(x) >= N:
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    return y


def running_std(x: List, N: int) -> np.ndarray:
    """Compute the running standard deviation of a list with a sliding window.

    The first N-1 entries of the deviation are 0.

    Args:
        x: A list of numeric values.
        N: The size of the sliding window.

    Returns:
        An array of the running standard deviation.
    """
    std = np.zeros_like(x)
    if len(x) >= N:
        std[N - 1:] = np.std(np.lib.stride_tricks.sliding_window_view(x, N), axis=-1)
    return std


def module_type_from_string(module_name: str) -> Callable[[str], type]:
    """Create a function that gets attributes from a module by name.

    Example:
        >>> import torch
        >>> torch_type = module_type_from_string("torch")
        >>> tensor_type = torch_type("Tensor")
        >>> tensor_type
        <class 'torch.Tensor'>

    Args:
        module_name: The module name.

    Returns:
        The module attribute.
    """

    def _module_type_from_string(name: str) -> type:
        return getattr(sys.modules[module_name], name)

    return _module_type_from_string


def mkdir_date(path: Path) -> Path:
    """Make a unique directory within the given directory with the current time as name.

    Args:
        path: Parent folder path.
    """
    assert path.is_dir()
    save_dir = path / datetime.now().strftime("%Y_%m_%d_%H_%M")
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        t = 1
        while save_dir.is_dir():
            curr_date_unique = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_({t})"
            save_dir = path / (curr_date_unique)
            t += 1
        save_dir.mkdir(parents=True)
    return save_dir


def load_config(default_config_path: Path, config_path: Path | None = None) -> SimpleNamespace:
    """Load the training configuration from the specified paths.

    The ``default_config_path`` argument should point to a complete configuration with all necessary
    parameters. In order to overwrite the default parameters, another config file at ``config_path``
    can be specified. This configuration always superseeds the default configuration.

    Args:
        default_config_path: Path to the default configuration.
        config_path: Optional path to a custom configuration.

    Returns:
        The configuration as a ``SimpleNamespace``.

    Raises:
        InvalidConfigError: An invalid logging level has been specified.
    """
    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_path is not None:
        if config_path.is_file():
            with open(config_path, "r") as f:
                _config = yaml.safe_load(f)
            _overwrite_dicts(config, _config)  # Overwrite default config with keys from user config
        else:
            logger.warning(f"Config file specified at {config_path} does not exist. Using defaults")
    loglvl = config["loglevel"].lower()
    if loglvl == "debug":
        config["loglevel"] = logging.DEBUG
    elif loglvl == "info":
        config["loglevel"] = logging.INFO
    elif loglvl == "warning":
        config["loglevel"] = logging.WARNING
    elif loglvl == "error":
        config["loglevel"] = logging.ERROR
    else:
        raise InvalidConfigError(f"Loglevel {config['loglevel']} in config not supported!")
    return dict2namespace(config)


def _overwrite_dicts(target_dict: dict, source_dict: dict) -> dict:
    for key, value in target_dict.items():
        if key not in source_dict.keys():
            continue
        if key == "kwargs":  # Do not merge kwargs, just overwrite
            target_dict[key] = source_dict[key]
        elif isinstance(value, dict):
            _overwrite_dicts(target_dict[key], source_dict[key])
        else:
            target_dict[key] = source_dict[key]
    for key, value in source_dict.items():
        if key not in target_dict.keys():
            target_dict[key] = source_dict[key]
    return target_dict


def dict2namespace(ns_dict: dict) -> SimpleNamespace:
    """Convert a dictionary to a (possibly nested) ``SimpleNamespace``.

    All dictionary key value pairs are converted to namespace attributes. If the value is a
    dictionary, another namespace object is used.

    Args:
        ns_dict: Dictionary for conversion.

    Returns:
        A namespace equivalent of the dictionary.
    """
    ns = copy.deepcopy(SimpleNamespace(**ns_dict))
    for key, value in ns_dict.items():
        if isinstance(value, dict):
            setattr(ns, key, dict2namespace(value))  # Works recursively with nested dicts
    return ns


def namespace2dict(ns: SimpleNamespace | None) -> dict:
    """Convert a ``SimpleNamespace`` to a (possibly nested) dictionary.

    All namespace attributes are converted to dictionary key value pairs. If the attribute is
    another namespace, it is also converted to a dictionary.

    Args:
        ns: NameSpace object.

    Returns:
        A (possibly nested) dictionary of the namespace.
    """
    if ns is None:
        return {}
    ns_dict = copy.deepcopy(vars(ns))
    for key, value in ns_dict.items():
        if isinstance(value, SimpleNamespace):  # Works recursively with nested namespaces
            ns_dict[key] = namespace2dict(getattr(ns, key))
    return ns_dict


def load_remote_config(address: str,
                       secret: str,
                       red: RedisType | None = None,
                       local_config: SimpleNamespace | None = None) -> SimpleNamespace:
    """Load the training configuration from the training server.

    This function allows us to only specify the address of a training server and its credentials.
    All hyperparameters etc. are copied from the server.

    Args:
        address: Address of the training server.
        secret: Redis secret.
        red: Optional redis instance that is used to load the remote config.
        local_config: Optional local config that should not be overwritten by the remote config.

    Returns:
        The remote training configuration.
    """
    if red is None:
        red = redis.Redis(host=address, port=6379, password=secret, db=0, decode_responses=True)
    config = None
    while config is None:
        config = red.get("config")
        time.sleep(0.2)
        logger.debug("Waiting for remote config")
    remote_config = json.loads(config)
    config = dict2namespace(_overwrite_dicts(remote_config, namespace2dict(local_config)))
    config.redis_address = address
    return config


def load_redis_secret(path: Path, default_path: Path = Path("/run/secrets/redis_secret")) -> str:
    """Load the redis secret from a `.secret` file.

    The file is expected to contain the the line "requirepass XXX", where XXX is the redis secret.
    If no file is found, the default path is used.

    Warning:
        If a secret file is present under /run/secrets/redis.secret, the supplied path argument is
        ignored!

    Args:
        path: Path to the secret file.
        default_path: Default path that is checked if the secret is not found under ``path``.

    Returns:
        The secret.
    """
    if not path.is_file():
        logger.info(f"No secret detected under {path}. Using default path {default_path}")
        path = default_path
    if not path.is_file():
        raise MissingConfigError(f"Missing password configuration for redis in {path}")
    with open(path, "r") as f:
        conf = f.readlines()
    secret = None
    for line in conf:
        if len(line) > 12 and line[0:12] == "requirepass ":
            secret = line[12:]
            break
    if secret is None:
        raise MissingConfigError(f"Password configuration for redis in {path} is invalid")
    return secret
