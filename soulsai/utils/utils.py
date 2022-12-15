import json
from types import SimpleNamespace
import logging
from pathlib import Path
from datetime import datetime
import time
import copy

import numpy as np
import yaml
import redis

from soulsai.exception import InvalidConfigError, MissingConfigError

logger = logging.getLogger(__name__)


def running_mean(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def running_std(x, N):
    std = np.zeros_like(x)
    if len(x) >= N:
        std[N-1:] = np.var(np.lib.stride_tricks.sliding_window_view(x, N), axis=-1)
    return std


def mkdir_date(path: Path):
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


def load_config(default_config_path, config_path=None):
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


def _overwrite_dicts(target_dict, source_dict):
    for key, value in target_dict.items():
        if not key in source_dict.keys():
            continue
        if isinstance(value, dict):
            _overwrite_dicts(target_dict[key], source_dict[key])
        else:
            target_dict[key] = source_dict[key]
    for key, value in source_dict.items():
        if not key in target_dict.keys():
            target_dict[key] = source_dict[key]
    return target_dict


def dict2namespace(dict_, create_copy=True):
    # Works recursively with nested dicts
    ns = SimpleNamespace(**dict_)
    if create_copy:
        ns = copy.deepcopy(ns)
    for key, value in dict_.items():
        if isinstance(value, dict):
            setattr(ns, key, dict2namespace(value, create_copy=True))
    return ns


def namespace2dict(ns, create_copy=True):
    # Works recursively with nested namespaces
    dict_ = vars(ns)
    if create_copy:
        dict_ = copy.deepcopy(dict_)
    for key, value in dict_.items():
        if isinstance(value, SimpleNamespace):
            dict_[key] = namespace2dict(getattr(ns, key), create_copy=True)
    return dict_


def load_remote_config(address, secret):
    red = redis.Redis(host=address, port=6379, password=secret, db=0, decode_responses=True)
    config = None
    while config is None:
        config = red.get("config")
        time.sleep(3)
        logger.debug("Waiting for remote config")
    config = dict2namespace(json.loads(config))
    config.redis_address = address
    return config


def load_redis_secret(path):
    assert path.suffix == ".secret", "Secrets have to be stored as .secret files!"
    with open(path, "r") as f:
        conf = f.readlines()
    secret = None
    for line in conf:
        if len(line) > 12 and line[0:12] == "requirepass ":
            secret = line[12:]
            break
    if secret is None:
        raise MissingConfigError(f"Missing password configuration for redis in {path}")
    return secret
